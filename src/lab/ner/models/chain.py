"""Linear-chain CRF inference over BIO tags, for one head or several at once."""

from __future__ import annotations

import torch
import torch.nn as nn

from lab.ner.models.bio import bio_constraint_masks, normalize_id2label

INITIAL_SCORE_RANGE = 0.1


class ConstrainedChain(nn.Module):
    """
    Start, transition and end scores per head, and every computation over them.

    Inputs are packed chains (`pack_chains`): emissions `[batch, width, heads, tags]`
    and a mask `[batch, width, heads]` padded on the right. Each head is its own chain
    with its own scores; the BIO constraints are the same for all of them, derived from
    `id2label` — starting on `I-X`, `O → I-X` and `I-X` after another type are
    excluded as `-inf` when the potentials are built, so no path through them has any
    weight, in the loss, Viterbi and the marginals alike (D155). With
    `constrain_transitions=False` every transition is allowed.

    The constraint masks are buffers that are not saved: they follow from `id2label`,
    so a loaded model rebuilds them. Scores start uniform in ±0.1, as `pytorch-crf`
    started them. Everything is computed in float32 with autocast off, whatever the
    emissions' dtype.
    """

    def __init__(
        self,
        id2label: dict,
        num_heads: int = 1,
        constrain_transitions: bool = True,
    ) -> None:
        super().__init__()

        id2label = normalize_id2label(id2label)
        num_tags = len(id2label)
        invalid_start, invalid_transitions = bio_constraint_masks(id2label)

        if not constrain_transitions:
            invalid_start = [False] * num_tags
            invalid_transitions = [[False] * num_tags for _ in range(num_tags)]

        self.num_heads = num_heads
        self.num_tags = num_tags
        self.constrain_transitions = constrain_transitions

        self.register_buffer("invalid_start", torch.tensor(invalid_start), persistent=False)
        self.register_buffer(
            "invalid_transitions", torch.tensor(invalid_transitions), persistent=False
        )

        self.start_transitions = nn.Parameter(_initial_scores(num_heads, num_tags))
        self.transitions = nn.Parameter(_initial_scores(num_heads, num_tags, num_tags))
        self.end_transitions = nn.Parameter(_initial_scores(num_heads, num_tags))

    def loss(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        The negative log-likelihood of the gold tags, per labelled word, balanced over heads.

        Each head's chains' negative log-likelihoods are summed over the batch and divided
        by that head's labelled words, then averaged over the heads with at least one
        (D158) — the rule `head_balanced_loss` applies to cross-entropy. A batch with no
        labelled word gives a zero that still carries a gradient.
        """
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            nll = self.log_partition(emissions, mask) - self.path_score(emissions, tags, mask)

        words = mask.sum(dim=(0, 1))
        head_losses = nll.sum(dim=0) / words.clamp_min(1)
        loss = head_losses.sum() / (words > 0).sum().clamp_min(1)

        return loss + emissions.sum() * 0.0

    def log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Each chain's log total weight over every allowed tag path, `[batch, heads]`.

        An empty chain's is zero, the weight of its one empty path.
        """
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            alphas = self._forward_scores(emissions.float(), mask)

            if not alphas:
                return emissions.new_zeros(emissions.shape[0], self.num_heads, dtype=torch.float32)

            _, _, end = self._potentials()
            log_partition = torch.logsumexp(alphas[-1] + end, dim=-1)

        return torch.where(mask.any(dim=1), log_partition, torch.zeros_like(log_partition))

    def path_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Each chain's score for the tag path `tags`, `[batch, heads]`; an empty chain's is zero.

        A path the constraints exclude raises: it has no weight, so its loss would be
        infinite. Rows encoded by this library never hold one (D157), so it means the
        rows' labels were built some other way.
        """
        self._check_tags(tags, mask)

        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            heads = torch.arange(self.num_heads, device=tags.device)
            lengths = mask.sum(dim=1)
            has_words = lengths > 0

            emission_scores = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)
            score = torch.where(mask, emission_scores, 0.0).sum(dim=1)

            if tags.shape[1] == 0:
                return score

            start = self.start_transitions[heads[None, :], tags[:, 0]]
            score = score + torch.where(has_words, start, 0.0)

            transitions = self.transitions[heads[None, None, :], tags[:, :-1], tags[:, 1:]]
            score = score + torch.where(mask[:, 1:], transitions, 0.0).sum(dim=1)

            last_tags = tags.gather(1, (lengths - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
            end = self.end_transitions[heads[None, :], last_tags]

        return score + torch.where(has_words, end, 0.0)

    def viterbi(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Each chain's best allowed tag path, `[batch, width, heads]`, `-100` past its end.
        """
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            start, transitions, end = self._potentials()
            width = emissions.shape[1]
            tags = torch.full(mask.shape, -100, dtype=torch.long, device=emissions.device)

            if width == 0:
                return tags

            score = start + emissions[:, 0]
            backpointers = []

            for step in range(1, width):
                best, backpointer = (score.unsqueeze(-1) + transitions).max(dim=-2)
                stepped = best + emissions[:, step]
                score = torch.where(mask[:, step, :, None], stepped, score)
                backpointers.append(backpointer)

            state = (score + end).argmax(dim=-1)

            for step in range(width - 1, -1, -1):
                tags[:, step] = torch.where(mask[:, step], state, -100)

                if step == 0:
                    break

                previous = backpointers[step - 1].gather(-1, state.unsqueeze(-1)).squeeze(-1)
                state = torch.where(mask[:, step], previous, state)

        return tags

    def log_marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Each step's log-probability of each tag given the whole chain, by forward-backward.

        `[batch, width, heads, tags]`, in float32. A tag the constraints rule out at a
        step is `-inf`; a step past a chain's end is zero throughout, a finite filler no
        reader uses (D161).
        """
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            alphas = self._forward_scores(emissions, mask)

            if not alphas:
                return torch.zeros_like(emissions)

            _, transitions, end = self._potentials()
            log_partition = torch.logsumexp(alphas[-1] + end, dim=-1, keepdim=True)
            beta = end.expand_as(alphas[-1])
            marginals = []

            for step in range(emissions.shape[1] - 1, -1, -1):
                marginal = alphas[step] + beta - log_partition
                marginals.append(torch.where(mask[:, step, :, None], marginal, 0.0))

                following = (emissions[:, step] + beta).unsqueeze(-2)
                stepped = torch.logsumexp(transitions + following, dim=-1)
                beta = torch.where(mask[:, step, :, None], stepped, beta)

        return torch.stack(marginals[::-1], dim=1)

    def _potentials(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.start_transitions.float().masked_fill(self.invalid_start, -torch.inf)
        transitions = self.transitions.float().masked_fill(self.invalid_transitions, -torch.inf)

        return start, transitions, self.end_transitions.float()

    def _forward_scores(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        start, transitions, _ = self._potentials()

        if emissions.shape[1] == 0:
            return []

        alpha = start + emissions[:, 0]
        alphas = [alpha]

        for step in range(1, emissions.shape[1]):
            stepped = torch.logsumexp(alpha.unsqueeze(-1) + transitions, dim=-2)
            stepped = stepped + emissions[:, step]
            alpha = torch.where(mask[:, step, :, None], stepped, alpha)
            alphas.append(alpha)

        return alphas

    def _check_tags(self, tags: torch.Tensor, mask: torch.Tensor) -> None:
        unknown = mask & ((tags < 0) | (tags >= self.num_tags))

        if unknown.any():
            raise ValueError(
                f"Gold tags outside 0..{self.num_tags - 1} at [row, step, head] "
                f"{unknown.nonzero()[:10].tolist()}; the labels do not match this model's "
                "label2id."
            )

        if tags.shape[1] == 0:
            return

        excluded = torch.zeros_like(mask)
        excluded[:, 0] = mask[:, 0] & self.invalid_start[tags[:, 0]]
        excluded[:, 1:] = mask[:, 1:] & self.invalid_transitions[tags[:, :-1], tags[:, 1:]]

        if excluded.any():
            raise ValueError(
                f"Gold tag paths the BIO constraints exclude, at [row, step, head] "
                f"{excluded.nonzero()[:10].tolist()}: an I- with no B- or I- of its type "
                "before it. The library's encoder never produces one (D157); encode the "
                "rows with it, or pass constrain_transitions=False."
            )


def _initial_scores(*shape: int) -> torch.Tensor:
    return torch.empty(*shape).uniform_(-INITIAL_SCORE_RANGE, INITIAL_SCORE_RANGE)
