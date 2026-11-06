"""Exact constrained linear-chain inference, vectorized over batches and heads."""

from __future__ import annotations

import torch
from torch import nn

from lab.ner.models.bio import bio_constraint_masks


def reachable_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Reduce unreachable states without undefined gradients from all-negative infinity."""
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, values, torch.zeros_like(values))

    return torch.logsumexp(safe, dim=dim).masked_fill(~reachable.squeeze(dim), -torch.inf)


class ConstrainedChain(nn.Module):
    """Independent BIO2 chains with identical constraints for every inference operation.

    Inputs use [B,T,K,C], or [B,T,C] for one head. Masks must be right-padded
    [B,T,K] (or [B,T]). Empty chains contribute no loss and have absent tags -100
    and zero marginal probabilities. Unknown gold at an active position is rejected;
    this component does not implement partially observed supervision.
    """

    def __init__(
        self, id2label: dict[int, str], num_heads: int = 1, constrain_transitions: bool = True
    ) -> None:
        super().__init__()
        labels = {int(index): label for index, label in id2label.items()}
        if sorted(labels) != list(range(len(labels))) or not labels or num_heads < 1:
            raise ValueError("Use contiguous label IDs and num_heads >= 1.")
        if any(
            label not in ("O", "B", "I") and not (label.startswith(("B-", "I-")) and len(label) > 2)
            for label in labels.values()
        ):
            raise ValueError("ConstrainedChain supports BIO2 only; BIOES/BILOU is not implemented.")
        vocabulary = {
            index: label if label not in ("B", "I") else f"{label}-ENTITY"
            for index, label in labels.items()
        }
        start, transitions = bio_constraint_masks(vocabulary)
        if constrain_transitions and all(start):
            raise ValueError("BIO2 vocabulary has no legal start state; include O or B states.")
        self.register_buffer("invalid_start", torch.tensor(start) & constrain_transitions)
        self.register_buffer(
            "invalid_transitions", torch.tensor(transitions) & constrain_transitions
        )
        self.start_transitions = nn.Parameter(
            torch.zeros(num_heads, len(labels), dtype=torch.float32)
        )
        self.transitions = nn.Parameter(
            torch.zeros(num_heads, len(labels), len(labels), dtype=torch.float32)
        )
        self.end_transitions = nn.Parameter(
            torch.zeros(num_heads, len(labels), dtype=torch.float32)
        )
        self.num_heads = num_heads
        self.num_states = len(labels)

    def _inputs(self, emissions: torch.Tensor, mask: torch.Tensor):
        single = emissions.ndim == 3
        if single:
            emissions, mask = emissions.unsqueeze(2), mask.unsqueeze(2)
        if emissions.ndim != 4 or mask.shape != emissions.shape[:-1]:
            raise ValueError("Expected emissions [B,T,K,C] and matching mask [B,T,K].")
        if emissions.shape[2:] != (self.num_heads, self.num_states):
            raise ValueError("Emission head/state dimensions disagree with the chain.")
        mask = mask.bool()
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Chain masks must be right padded; compact eligible words first.")
        if not torch.isfinite(emissions).all():
            raise ValueError("Emissions must be finite.")

        return emissions.float(), mask, single

    def _potentials(self):
        return (
            self.start_transitions.float().masked_fill(self.invalid_start, -torch.inf),
            self.transitions.float().masked_fill(self.invalid_transitions, -torch.inf),
            self.end_transitions.float(),
        )

    def _forward(self, emissions, mask):
        start, transitions, end = self._potentials()
        alpha = start.unsqueeze(0).expand(emissions.shape[0], -1, -1)
        history = []
        for position in range(emissions.shape[1]):
            step = (
                start + emissions[:, position]
                if position == 0
                else (
                    reachable_logsumexp(alpha.unsqueeze(-1) + transitions, -2)
                    + emissions[:, position]
                )
            )
            alpha = torch.where(mask[:, position, :, None], step, alpha)
            history.append(alpha)
        partition = reachable_logsumexp(alpha + end, -1)
        active = mask.any(dim=1)
        partition = torch.where(active, partition, torch.zeros_like(partition))

        return partition, history

    def log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log total weight of legal paths, with empty-path score defined as zero."""
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            values, valid, single = self._inputs(emissions, mask)
            result, _ = self._forward(values, valid)

        return result[:, 0] if single else result

    def gold_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Score legal observed paths; report batch, head and position for invalid gold."""
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            values, valid, single = self._inputs(emissions, mask)
            tags = tags.unsqueeze(2) if single else tags
            if tags.shape != valid.shape:
                raise ValueError("Gold and mask shapes differ.")
            invalid_ids = valid & ((tags < 0) | (tags >= self.num_states))
            if invalid_ids.any():
                raise ValueError(
                    f"Unknown gold at [batch,position,head] {invalid_ids.nonzero()[:10].tolist()}."
                )
            safe_tags = tags.masked_fill(~valid, 0).long()
            if valid.shape[1] == 0:
                result = values.sum(dim=(1, 3)) * 0
                return result[:, 0] if single else result
            illegal = torch.zeros_like(valid)
            illegal[:, 0] = valid[:, 0] & self.invalid_start[safe_tags[:, 0]]
            if valid.shape[1] > 1:
                illegal[:, 1:] = (
                    valid[:, 1:] & self.invalid_transitions[safe_tags[:, :-1], safe_tags[:, 1:]]
                )
            if illegal.any():
                raise ValueError(
                    f"Illegal BIO2 gold transition at [batch,position,head] "
                    f"{illegal.nonzero()[:10].tolist()}. Repair gold before CRF training."
                )
            heads = torch.arange(self.num_heads, device=values.device)[None, :]
            result = self.start_transitions.float()[heads, safe_tags[:, 0]] * valid[:, 0]
            result = result + (values.gather(-1, safe_tags[..., None]).squeeze(-1) * valid).sum(1)
            for position in range(1, valid.shape[1]):
                edge = self.transitions.float()[
                    heads, safe_tags[:, position - 1], safe_tags[:, position]
                ]
                result = result + edge * valid[:, position]
            lengths = valid.sum(1)
            last = safe_tags.gather(1, (lengths - 1).clamp_min(0)[:, None, :]).squeeze(1)
            result = result + self.end_transitions.float()[heads, last] * (lengths > 0)

        return result[:, 0] if single else result

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        head_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Word-normalized NLL per head, averaged over active heads with explicit weights."""
        mask = mask.bool()
        nll = self.log_partition(emissions, mask) - self.gold_score(emissions, tags, mask)
        if nll.ndim == 1:
            nll, mask = nll[:, None], mask[:, :, None]
        counts = mask.sum(dim=(0, 1))
        weights = (
            torch.ones_like(counts, dtype=torch.float32) if head_weights is None else head_weights
        )
        if (
            weights.shape != counts.shape
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Head weights must be finite, nonnegative, nonempty and match heads.")
        active = weights * (counts > 0)

        return ((nll.sum(0) / counts.clamp_min(1)) * active).sum() / active.sum().clamp_min(
            torch.finfo(torch.float32).tiny
        )

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi tags with -100 outside the right-padded chain, without parameter mutation."""
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            values, valid, single = self._inputs(emissions, mask)
            start, transitions, end = self._potentials()
            score = start.unsqueeze(0).expand(values.shape[0], -1, -1)
            pointers = []
            for position in range(values.shape[1]):
                candidates = score.unsqueeze(-1) + transitions
                step, pointer = candidates.max(-2)
                step = start + values[:, position] if position == 0 else step + values[:, position]
                score = torch.where(valid[:, position, :, None], step, score)
                pointers.append(pointer)
            state = (score + end).argmax(-1)
            decoded = torch.full(valid.shape, -100, dtype=torch.long, device=values.device)
            for position in range(values.shape[1] - 1, -1, -1):
                decoded[:, position] = torch.where(valid[:, position], state, -100)
                previous = pointers[position].gather(-1, state[..., None]).squeeze(-1)
                state = torch.where(valid[:, position], previous, state)

        return decoded[:, :, 0] if single else decoded

    def log_marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalized log marginals; absent positions are all -inf, not confident O."""
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            values, valid, single = self._inputs(emissions, mask)
            partition, alphas = self._forward(values, valid)
            _, transitions, end = self._potentials()
            beta = end.unsqueeze(0).expand(values.shape[0], -1, -1)
            results = []
            for position in range(values.shape[1] - 1, -1, -1):
                result = alphas[position] + beta - partition[..., None]
                results.append(result.masked_fill(~valid[:, position, :, None], -torch.inf))
                previous = reachable_logsumexp(
                    transitions + (values[:, position] + beta).unsqueeze(-2), -1
                )
                beta = torch.where(valid[:, position, :, None], previous, beta)
            output = torch.stack(list(reversed(results)), 1) if results else values.clone()

        return output[:, :, 0] if single else output
