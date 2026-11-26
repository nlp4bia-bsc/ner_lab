"""Verify the CRF chain and packing against brute-force enumeration of every tag path."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

ID2LABEL = {0: "O", 1: "B-D", 2: "I-D"}
TOLERANCE = 1e-4


def brute_force(chain, emissions, head: int, length: int, constrained: bool) -> dict:
    """Every tag path of one chain, its score, and the partition, marginals and best path."""
    import torch

    start = chain.start_transitions[head].detach().double()
    transitions = chain.transitions[head].detach().double()
    end = chain.end_transitions[head].detach().double()
    emissions = emissions.double()

    scores = {}

    for path in itertools.product(range(3), repeat=length):
        if constrained and is_excluded(path):
            continue

        score = start[path[0]] + emissions[0, path[0]] + end[path[-1]]

        for step in range(1, length):
            score = score + transitions[path[step - 1], path[step]] + emissions[step, path[step]]

        scores[path] = float(score)

    log_partition = math.log(sum(math.exp(score) for score in scores.values()))
    marginals = torch.zeros(length, 3, dtype=torch.float64)

    for path, score in scores.items():
        for step, tag in enumerate(path):
            marginals[step, tag] += math.exp(score - log_partition)

    return {
        "scores": scores,
        "log_partition": log_partition,
        "marginals": marginals,
        "best": max(scores, key=scores.get),
    }


def is_excluded(path: tuple[int, ...]) -> bool:
    return path[0] == 2 or any(
        previous == 0 and tag == 2 for previous, tag in zip(path, path[1:])
    )


def random_chain(num_heads: int, constrained: bool = True):
    import torch

    from lab.ner.models.chain import ConstrainedChain

    chain = ConstrainedChain(ID2LABEL, num_heads=num_heads, constrain_transitions=constrained)

    with torch.no_grad():
        for parameter in chain.parameters():
            parameter.normal_(0.0, 1.0)

    return chain


def legal_path(length: int, generator) -> list[int]:
    import torch

    path = []

    for _ in range(length):
        tag = int(torch.randint(0, 3, (1,), generator=generator))

        if tag == 2 and (not path or path[-1] == 0):
            tag = 1

        path.append(tag)

    return path


def verify_against_brute_force(checks: Checks, constrained: bool) -> None:
    import torch

    generator = torch.Generator().manual_seed(7 if constrained else 11)
    lengths = [[5, 1, 3], [0, 4, 2], [3, 3, 0], [2, 5, 5]]
    num_heads = 3
    width = 5
    chain = random_chain(num_heads, constrained)

    emissions = torch.randn(len(lengths), width, num_heads, 3, generator=generator) * 2
    mask = torch.zeros(len(lengths), width, num_heads, dtype=torch.bool)
    tags = torch.zeros(len(lengths), width, num_heads, dtype=torch.long)

    for row, row_lengths in enumerate(lengths):
        for head, length in enumerate(row_lengths):
            mask[row, :length, head] = True
            tags[row, :length, head] = torch.tensor(legal_path(length, generator), dtype=torch.long)

    log_partition = chain.log_partition(emissions, mask).detach()
    path_score = chain.path_score(emissions, tags, mask).detach()
    viterbi = chain.viterbi(emissions, mask)
    marginals = chain.log_marginals(emissions, mask).exp()

    partition_ok = score_ok = viterbi_ok = marginals_ok = empty_ok = True

    for row, row_lengths in enumerate(lengths):
        for head, length in enumerate(row_lengths):
            if length == 0:
                empty_ok &= float(log_partition[row, head]) == 0.0
                empty_ok &= float(path_score[row, head]) == 0.0
                empty_ok &= bool((viterbi[row, :, head] == -100).all())
                continue

            expected = brute_force(chain, emissions[row, :length, head], head, length, constrained)
            gold = tuple(tags[row, :length, head].tolist())

            partition_ok &= abs(float(log_partition[row, head]) - expected["log_partition"]) < TOLERANCE
            score_ok &= abs(float(path_score[row, head]) - expected["scores"][gold]) < TOLERANCE
            viterbi_ok &= tuple(viterbi[row, :length, head].tolist()) == expected["best"]
            viterbi_ok &= bool((viterbi[row, length:, head] == -100).all())
            marginals_ok &= bool(
                torch.allclose(
                    marginals[row, :length, head].double(), expected["marginals"], atol=TOLERANCE
                )
            )

    name = "constrained" if constrained else "unconstrained"
    checks.check(f"{name}: log-partition matches enumeration over 3 heads, padded", partition_ok)
    checks.check(f"{name}: gold path score matches enumeration", score_ok)
    checks.check(f"{name}: Viterbi is the best enumerated path, -100 past the end", viterbi_ok)
    checks.check(f"{name}: forward-backward marginals match enumeration", marginals_ok)
    checks.check(f"{name}: an empty chain has zero partition and score and no tags", empty_ok)

    padded = chain.log_marginals(emissions, mask)[~mask]
    checks.check(f"{name}: steps past a chain's end carry a finite zero filler", bool((padded == 0).all()))

    if constrained:
        excluded = chain.log_marginals(emissions, mask)[:, 0, :, 2][mask[:, 0]]
        checks.check(
            "constrained: I- at a chain's first step has probability zero",
            bool(torch.isneginf(excluded).all()),
        )


def verify_loss(checks: Checks) -> None:
    import torch

    generator = torch.Generator().manual_seed(3)
    chain = random_chain(2)
    emissions = torch.randn(3, 4, 2, 3, generator=generator, requires_grad=True)
    mask = torch.tensor(
        [[[1, 0], [1, 0], [1, 0], [0, 0]], [[1, 0], [0, 0], [0, 0], [0, 0]], [[1, 0], [1, 0], [0, 0], [0, 0]]],
        dtype=torch.bool,
    )
    tags = torch.tensor(
        [[[1, 0], [2, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]], [[1, 0], [0, 0], [0, 0], [0, 0]]]
    )

    loss = chain.loss(emissions, tags, mask)
    nll = chain.log_partition(emissions, mask) - chain.path_score(emissions, tags, mask)
    expected = nll[:, 0].sum() / 6

    checks.check(
        "the loss is head 0's NLL over its 6 words; head 1, with none, is left out",
        abs(float(loss) - float(expected)) < TOLERANCE,
    )

    loss.backward()
    checks.check("the loss backpropagates finite gradients", bool(torch.isfinite(emissions.grad).all()))
    checks.check(
        "no gradient reaches a step outside every chain",
        bool((emissions.grad[0, 3] == 0).all() and (emissions.grad[:, :, 1] == 0).all()),
    )
    checks.check(
        "an excluded transition's score gets no gradient",
        float(chain.transitions.grad[0, 0, 2]) == 0.0 and float(chain.start_transitions.grad[0, 2]) == 0.0,
    )

    empty = torch.randn(2, 0, 1, 3, requires_grad=True)
    empty_loss = random_chain(1).loss(
        empty, torch.zeros(2, 0, 1, dtype=torch.long), torch.zeros(2, 0, 1, dtype=torch.bool)
    )
    empty_loss.backward()
    checks.check(
        "a batch with no labelled word gives a zero loss that carries a gradient",
        float(empty_loss) == 0.0 and empty.grad is not None,
    )

    half = random_chain(1)
    half_emissions = torch.randn(2, 3, 1, 3, generator=generator)
    half_mask = torch.ones(2, 3, 1, dtype=torch.bool)
    checks.check(
        "float16 emissions are computed in float32, matching float32 to rounding",
        torch.allclose(
            half.log_marginals(half_emissions.half(), half_mask),
            half.log_marginals(half_emissions.half().float(), half_mask),
        )
        and half.log_marginals(half_emissions.half(), half_mask).dtype == torch.float32,
    )


def verify_gold_checks(checks: Checks) -> None:
    import torch

    chain = random_chain(1)
    emissions = torch.zeros(1, 3, 1, 3)
    mask = torch.ones(1, 3, 1, dtype=torch.bool)

    checks.raises(
        "an orphan I- in gold raises, naming where",
        ValueError,
        chain.path_score,
        emissions,
        torch.tensor([[[0], [2], [2]]]),
        mask,
        match="[[0, 1, 0]]",
    )
    checks.raises(
        "gold starting on I- raises",
        ValueError,
        chain.path_score,
        emissions,
        torch.tensor([[[2], [2], [0]]]),
        mask,
        match="[[0, 0, 0]]",
    )
    checks.raises(
        "a tag outside the vocabulary raises",
        ValueError,
        chain.path_score,
        emissions,
        torch.tensor([[[0], [3], [0]]]),
        mask,
        match="outside 0..2",
    )
    checks.check(
        "without constraints the same orphan I- is scored",
        bool(
            torch.isfinite(
                random_chain(1, constrained=False).path_score(
                    emissions, torch.tensor([[[0], [2], [2]]]), mask
                )
            ).all()
        ),
    )
    checks.check(
        "the constraint masks are not saved with the weights",
        set(chain.state_dict()) == {"start_transitions", "transitions", "end_transitions"},
    )


def verify_packing(checks: Checks) -> None:
    import torch

    from lab.ner.models.packing import pack_chains, unpack_chains

    labels = torch.tensor(
        [
            [[-100, -100], [0, 1], [-100, -100], [1, 2], [2, -100], [-100, -100]],
            [[-100, -100], [-100, -100], [0, 0], [-100, -100], [-100, -100], [-100, -100]],
        ]
    )
    emissions = torch.arange(2 * 6 * 2 * 3, dtype=torch.float32).reshape(2, 6, 2, 3)
    packed = pack_chains(emissions, labels)

    checks.equal("the width is the longest chain", packed.mask.shape, (2, 3, 2))
    checks.equal(
        "chain steps are the labelled positions, in order, per head",
        packed.positions[0].T.tolist(),
        [[1, 3, 4], [1, 3, 0]],
    )
    checks.equal(
        "gold tags follow their positions, O past a chain's end",
        packed.tags[0].T.tolist(),
        [[0, 1, 2], [1, 2, 0]],
    )
    checks.equal(
        "the mask is padded on the right",
        packed.mask[1].T.tolist(),
        [[True, False, False], [True, False, False]],
    )
    checks.check(
        "packed emissions are the window's at each chain position",
        bool(torch.equal(packed.emissions[0, 2, 0], emissions[0, 4, 0])),
    )

    restored = unpack_chains(packed.emissions, packed, fill=-1.0)
    in_chain = labels != -100
    checks.check(
        "unpacking writes every chain step back to its position",
        bool(torch.equal(restored[in_chain], emissions[in_chain])),
    )
    checks.check(
        "every position outside the chain gets the fill",
        bool((restored[~in_chain] == -1.0).all()),
    )

    tags = unpack_chains(torch.ones(2, 3, 2, dtype=torch.long), packed, fill=-100)
    checks.equal("tags unpack to [batch, length, heads]", tuple(tags.shape), (2, 6, 2))

    none = pack_chains(torch.zeros(2, 4, 1, 3), torch.full((2, 4, 1), -100))
    checks.equal("windows with no labelled word pack to width zero", none.mask.shape, (2, 0, 1))

    checks.raises(
        "labels that do not match the emissions raise",
        ValueError,
        pack_chains,
        emissions,
        labels[:, :, :1],
        match="do not match emissions",
    )


def main() -> int:
    checks = Checks("CRF chain and packing")

    verify_against_brute_force(checks, constrained=True)
    verify_against_brute_force(checks, constrained=False)
    verify_loss(checks)
    verify_gold_checks(checks)
    verify_packing(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
