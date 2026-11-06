"""Exhaustive reference checks for exact constrained chain inference."""

from __future__ import annotations

import itertools

import torch
from _harness import Checks, run

from lab.ner.models.chain import ConstrainedChain


def main():
    torch.manual_seed(19)
    checks = Checks("constrained chain mathematics")
    for heads in (1, 2, 4):
        chain = ConstrainedChain({0: "O", 1: "B", 2: "I"}, heads)
        with torch.no_grad():
            for parameter in chain.parameters():
                parameter.normal_()
        for length in range(1, 6):
            emissions = torch.randn(2, length, heads, 3, requires_grad=True)
            mask = torch.ones(2, length, heads, dtype=torch.bool)
            partition = chain.log_partition(emissions, mask)
            decoded = chain.decode(emissions, mask)
            marginals = chain.log_marginals(emissions, mask).exp()
            gold = torch.zeros_like(mask, dtype=torch.long)
            for batch, head in itertools.product(range(2), range(heads)):
                paths = [
                    path
                    for path in itertools.product(range(3), repeat=length)
                    if path[0] != 2 and all(not (a == 0 and b == 2) for a, b in zip(path, path[1:]))
                ]
                scores = torch.stack(
                    [
                        chain.start_transitions[head, path[0]]
                        + chain.end_transitions[head, path[-1]]
                        + sum(emissions[batch, t, head, tag] for t, tag in enumerate(path))
                        + sum(chain.transitions[head, a, b] for a, b in zip(path, path[1:]))
                        for path in paths
                    ]
                )
                expected_z = scores.logsumexp(0)
                checks.check(
                    "partition matches enumeration",
                    torch.allclose(partition[batch, head], expected_z, atol=2e-5),
                )
                checks.equal(
                    "Viterbi matches enumeration",
                    decoded[batch, :, head].tolist(),
                    list(paths[scores.argmax()]),
                )
                probabilities = scores.softmax(0)
                expected = torch.zeros(length, 3)
                for path, probability in zip(paths, probabilities):
                    for t, tag in enumerate(path):
                        expected[t, tag] += probability
                checks.check(
                    "marginals match enumeration",
                    torch.allclose(marginals[batch, :, head], expected, atol=2e-5),
                )
                gold[batch, :, head] = torch.tensor(paths[0])
                reference_gradient = torch.autograd.grad(expected_z, emissions, retain_graph=True)[
                    0
                ][batch, :, head]
                checks.check(
                    "partition gradient equals marginal",
                    torch.allclose(reference_gradient, marginals[batch, :, head], atol=2e-5),
                )
            derivative = torch.autograd.grad(partition.sum(), emissions, retain_graph=True)[0]
            checks.check(
                "DP derivative equals marginals", torch.allclose(derivative, marginals, atol=2e-5)
            )
            reference_score = chain.start_transitions[:, 0] + chain.end_transitions[:, 0]
            reference_score = reference_score + emissions[..., 0].sum(1)
            reference_score = reference_score + (length - 1) * chain.transitions[:, 0, 0]
            checks.check(
                "gold score formula",
                torch.allclose(chain.gold_score(emissions, gold, mask), reference_score, atol=2e-5),
            )
            expected_loss = ((partition - reference_score).sum(0) / mask.sum((0, 1))).mean()
            loss = chain(emissions, gold, mask)
            checks.check("NLL word reduction", torch.allclose(loss, expected_loss, atol=2e-5))
            loss.backward()
            checks.check(
                "finite backward",
                all(p.grad is None or torch.isfinite(p.grad).all() for p in chain.parameters()),
            )
            checks.check("emission gradient finite", torch.isfinite(emissions.grad).all().item())
            checks.check(
                "normalization",
                torch.allclose(
                    marginals.sum(-1), torch.ones_like(mask, dtype=torch.float32), atol=2e-5
                ),
            )
    chain = ConstrainedChain({0: "O", 1: "B", 2: "I"})
    values = torch.randn(3, 5, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    tags = torch.zeros(3, 5, dtype=torch.long)
    loss = chain(values, tags, mask)
    loss.backward()
    checks.check("empty/padded chains finite", torch.isfinite(values.grad).all().item())
    checks.check(
        "absent marginals zero", (chain.log_marginals(values, mask).exp()[~mask] == 0).all().item()
    )
    checks.check(
        "absent decisions explicit", (chain.decode(values, mask)[~mask] == -100).all().item()
    )
    checks.raises(
        "START I rejected", ValueError, chain.gold_score, values, torch.full_like(tags, 2), mask
    )
    bad = tags.clone()
    bad[0, 1] = 2
    checks.raises("O I rejected", ValueError, chain.gold_score, values, bad, mask)
    holes = mask.clone()
    holes[0, 1] = False
    checks.raises("uncompacted holes rejected", ValueError, chain.log_partition, values, holes)
    empty = torch.zeros_like(mask)
    checks.equal("all empty loss zero", chain(values, tags, empty).item(), 0.0)
    for dtype in (torch.bfloat16, torch.float16):
        with torch.autocast("cpu", dtype=dtype):
            result = chain(values.to(dtype), tags, mask)
        checks.equal("DP FP32 under autocast", result.dtype, torch.float32)
        checks.check("autocast finite", torch.isfinite(result).item())
    adversarial = torch.zeros(1, 3, 3)
    adversarial[:, :, 2] = 1e6
    path = chain.decode(adversarial, torch.ones(1, 3, dtype=torch.bool))[0].tolist()
    checks.check("illegal START cannot win with huge emission", path[0] != 2)
    flat = ConstrainedChain({0: "O", 1: "B-X", 2: "I-X", 3: "B-Y", 4: "I-Y"})
    checks.check("cross-type continuation forbidden", flat.invalid_transitions[1, 4].item())
    if not torch.cuda.is_available():
        checks.skip("CUDA mathematics", "no CUDA device")
    return checks.report()


if __name__ == "__main__":
    run(main)
