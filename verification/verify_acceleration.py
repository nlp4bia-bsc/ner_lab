"""Device, vectorization and parallel-replica checks with explicit CUDA qualification gates."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from _harness import Checks, run
from fixtures import local_tokenizer, tiny_base_model
from verify_multihead import overlap_corpus

from lab.ner import Encoder, build_model, training_arguments
from lab.ner.evaluation.multilabel import binary_predictions
from lab.ner.inference import load_model, predict_logits, resolve_device
from lab.ner.models.binary_decoding import BinaryDecoding, decode_binary
from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_words
from lab.ner.training import model_encoding, train
from lab.ner.training.arguments import default_precision, for_inference
from lab.ner.training.collation import classification_collator
from lab.ner.training.dataset import to_dataset
from lab.ner.training.devices import apply_device_policy
from lab.ner.training.tracking import ResourceTracker
from lab.ner.training.trainer import BinaryPredictionTrainer


def verify_compaction(checks, device):
    for heads in (1, 2, 5):
        torch.manual_seed(52)
        values = torch.randn(3, 11, heads, 3, device=device, requires_grad=True)
        valid = torch.rand(3, 11, heads, device=device) > 0.4
        valid[1] = False
        packed = pack_words(values, valid)
        restored = packed.restore(packed.emissions)
        checks.check(
            "compaction preserves eligible emissions", torch.equal(restored[valid], values[valid])
        )
        checks.check("compaction absence explicit", (restored[~valid] == -100).all())
        packed.restore(packed.emissions, absent=0).sum().backward()
        checks.check(
            "scatter/gather backward",
            torch.equal(values.grad, valid[..., None].expand_as(values).float()),
        )
        chain = ConstrainedChain({0: "O", 1: "B", 2: "I"}, heads).to(device)
        partition, history = chain._forward(packed.emissions, packed.mask)
        checks.equal("loss partition does not retain alpha history", len(history), 0)
        checks.check("partition finite", torch.isfinite(partition).all())
        distribution = packed.restore(
            chain.log_marginals(packed.emissions, packed.mask), absent=-torch.inf
        )
        checks.check("absent marginals stay absent", torch.isneginf(distribution[~valid]).all())
        checks.check(
            "eligible marginals normalized",
            torch.allclose(
                distribution.exp().sum(-1)[valid],
                torch.ones_like(distribution[..., 0][valid]),
                atol=2e-6,
            ),
        )
    for length in (0, 7):
        values = torch.randn(2, length, 3, 3, device=device, requires_grad=True)
        packed = pack_words(values, torch.zeros(2, length, dtype=torch.bool, device=device))
        checks.equal(
            "empty chains restore shape", packed.restore(packed.emissions).shape, values.shape
        )
        packed.emissions.sum().backward()
        checks.check(
            "empty compaction differentiable", values.grad is not None and not values.grad.any()
        )

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        packed = pack_words(torch.randn(2, 8, 3, 3), torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]] * 2))
        packed.restore(packed.emissions)
    checks.check(
        "packing/restoration avoid dynamic nonzero",
        not any(event.key == "aten::nonzero" for event in profile.key_averages()),
    )


def build_fixture(base, tokenizer, corpus, architecture):
    binary = architecture == "multilabel"
    multiple = architecture in ("multihead", "multihead_crf", "multilabel")
    encoder = Encoder(
        tokenizer,
        None if multiple else "SYMPTOM",
        "es",
        max_length=32,
        include_word_mask=architecture in ("crf", "multihead_crf"),
        label_schema="multilabel" if binary else "bio",
    )
    rows = encoder.encode(corpus)
    kwargs = {"entity_labels": encoder.entity_labels} if multiple else {}
    model = build_model(
        str(base),
        encoder.label2id,
        encoder.id2label,
        architecture=architecture,
        **({"dropout": 0} if architecture != "linear" else {}),
        **kwargs,
    )
    collator = classification_collator(model, tokenizer, pad_to_multiple_of=None)
    dataset = to_dataset(rows)
    batch = collator([dataset[index] for index in range(2)])
    return model, encoder, rows, batch


def verify_models(checks, base, tokenizer, corpus, device):
    for architecture in ("linear", "crf", "multihead", "multihead_crf", "multilabel"):
        model, encoder, rows, batch = build_fixture(base, tokenizer, corpus, architecture)
        model.eval()
        with torch.no_grad():
            reference = model(**batch).logits
        model.to(device)
        batch = {name: value.to(device) for name, value in batch.items()}
        backbone = getattr(model, "backbone", getattr(model, "bert", None))
        calls = []
        hook = backbone.register_forward_hook(
            lambda module, inputs, outputs: calls.append(outputs.last_hidden_state.device)
        )
        result = model(**batch)
        checks.equal(architecture + " one backbone forward", len(calls), 1)
        checks.check(
            architecture + " output stays on device", result.logits.device == torch.device(device)
        )
        checks.check(
            architecture + " FP32 device equivalence",
            torch.allclose(reference, result.logits.detach().cpu(), atol=2e-5, rtol=2e-5),
        )
        result.loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        checks.check(
            architecture + " finite shared gradients",
            all(g is not None and torch.isfinite(g).all() for g in gradients),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if hasattr(model, "chain"):
            checks.check(
                "CRF parameters FP32",
                all(p.dtype == torch.float32 for p in model.chain.parameters()),
            )
            with patch.object(
                model.chain, "decode", side_effect=AssertionError("redundant Viterbi")
            ):
                probabilities = model.marginals_from_emissions(
                    result.logits.detach(), batch["attention_mask"], batch["crf_word_mask"]
                )
            checks.check("marginal-only API avoids Viterbi", probabilities is not None)
        hook.remove()
        if architecture == "multilabel":
            verify_binary_transport(checks, model, tokenizer, rows, device)
            verify_accumulation(checks, model, batch)
        if architecture != "linear":
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.train()
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            checks.check(
                architecture + " gradient checkpointing",
                torch.isfinite(model.classifier.weight.grad).all(),
            )


def verify_accumulation(checks, model, batch):
    model.eval()
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    reference = model.classifier.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    for index in range(2):
        (
            model(**{name: value[index : index + 1] for name, value in batch.items()}).loss / 2
        ).backward()
    checks.check(
        "equal-supervision microbatch accumulation",
        torch.allclose(model.classifier.weight.grad, reference, rtol=3e-5, atol=2e-6),
    )


def verify_binary_transport(checks, model, tokenizer, rows, device):
    rows = rows.iloc[[0, -1]].reset_index(drop=True)
    for side in ("right", "left"):
        tokenizer.padding_side = side
        raw = predict_logits(model, tokenizer, rows, device=device, batch_size=2)
        for decoder in ("threshold_bio", "constrained_viterbi"):
            model.decoding = BinaryDecoding(decoder=decoder)
            rows.attrs["multilabel_decoding"] = {"decoder": decoder}
            recorded_devices = []

            def decode(*args, **kwargs):
                recorded_devices.append(args[0].device.type)
                return decode_binary(*args, **kwargs)

            with patch("lab.ner.models.binary_decoding.decode_binary", side_effect=decode):
                transported = predict_logits(
                    model,
                    tokenizer,
                    rows,
                    device=device,
                    batch_size=2,
                    trainer_class=BinaryPredictionTrainer,
                )
            checks.equal(
                "binary decoder executes on inference device",
                recorded_devices,
                [torch.device(device).type],
            )
            checks.equal(
                "binary prediction transport shape", transported.shape, (*raw.shape[:-1], 6)
            )
            checks.check(
                "raw logits API unchanged", np.allclose(raw, transported[..., :2], atol=2e-5)
            )
            expected_tags, expected_probabilities = binary_predictions(
                rows, raw, model.entity_labels
            )
            with patch(
                "lab.ner.evaluation.multilabel.decode_binary",
                side_effect=AssertionError("CPU redecoding"),
            ):
                tags, probabilities = binary_predictions(rows, transported, model.entity_labels)
            checks.check("transport preserves tags", np.array_equal(tags, expected_tags))
            checks.check(
                "transport preserves real probabilities",
                np.allclose(probabilities, expected_probabilities, atol=2e-5),
            )
        streamed = predict_logits(
            model,
            tokenizer,
            rows,
            device=device,
            batch_size=1,
            trainer_class=BinaryPredictionTrainer,
        )
        tags, probabilities = binary_predictions(rows, streamed, model.entity_labels)
        expected_tags, expected_probabilities = binary_predictions(
            rows, streamed[..., :2], model.entity_labels
        )
        checks.equal("variable batch concatenation stays dense", streamed.shape[0], 2)
        for index, row in enumerate(rows.itertuples(index=False)):
            length = len(row.input_ids)
            checks.check(
                "streamed tags match batched tags",
                np.array_equal(tags[index, :length], expected_tags[index, :length]),
            )
            checks.check(
                "streamed probabilities match batched probabilities",
                np.allclose(
                    probabilities[index, :length], expected_probabilities[index, :length], atol=2e-5
                ),
            )
    tokenizer.padding_side = "right"


def verify_precision(checks, base, tokenizer, corpus, device):
    dtypes = (
        [torch.bfloat16] if torch.device(device).type == "cpu" else [torch.float16, torch.bfloat16]
    )
    for dtype in dtypes:
        if (
            dtype == torch.bfloat16
            and torch.device(device).type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            checks.skip("CUDA BF16", "device does not support BF16")
            continue
        for architecture in ("multihead_crf", "multilabel"):
            model, _, _, batch = build_fixture(base, tokenizer, corpus, architecture)
            model.to(device)
            batch = {name: value.to(device) for name, value in batch.items()}
            for loss in (
                ("bce", "weighted_bce", "focal", "asymmetric")
                if architecture == "multilabel"
                else ("crf",)
            ):
                if architecture == "multilabel":
                    from lab.ner.models.binary_loss import BinaryLoss

                    model.loss_options = BinaryLoss(loss=loss)
                model.zero_grad(set_to_none=True)
                with torch.autocast(torch.device(device).type, dtype=dtype):
                    result = model(**batch)
                checks.equal("sensitive loss remains FP32", result.loss.dtype, torch.float32)
                result.loss.backward()
                checks.check(
                    "autocast backward finite",
                    all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
                )


def verify_native_training(checks, base, tokenizer, corpus, root, device, precision=None):
    for architecture in ("multihead_crf", "multilabel"):
        model, encoder, rows, _ = build_fixture(base, tokenizer, corpus, architecture)
        overrides = (
            {}
            if precision is None
            else {"fp16": precision == torch.float16, "bf16": precision == torch.bfloat16}
        )
        arguments = training_arguments(
            root / (architecture + "-" + device.replace(":", "") + "-" + str(precision)),
            max_steps=2,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            use_cpu=torch.device(device).type == "cpu",
            disable_tqdm=True,
            dataloader_pin_memory=False,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            **overrides,
        )
        result = train(
            model,
            tokenizer,
            rows.iloc[:4],
            rows.iloc[:2],
            arguments,
            early_stopping_patience=None,
            save_model=True,
            model_encoding=model_encoding(encoder, str(base), architecture),
        )
        checks.equal("native accumulated training completes", result.trainer.state.global_step, 2)
        with torch.no_grad():
            before = predict_logits(result.model, tokenizer, rows.iloc[:2], device=device)
        loaded, _, _ = load_model(result.paths["best_model"], device=device)
        after = predict_logits(loaded, tokenizer, rows.iloc[:2], device=device)
        checks.check(
            "device train/save/load/predict round-trip", np.allclose(before, after, atol=2e-5)
        )


def verify_multiple_gpus(checks, base, tokenizer, corpus):
    if torch.cuda.device_count() < 2:
        checks.skip("DataParallel", "requires at least two CUDA devices")
        return
    for architecture in ("linear", "crf", "multihead", "multihead_crf", "multilabel"):
        model, _, _, batch = build_fixture(base, tokenizer, corpus, architecture)
        model.eval().to("cuda:0")
        batch = {name: value.to("cuda:0") for name, value in batch.items()}
        reference = model(**batch)
        calls = []
        backbone = getattr(model, "backbone", getattr(model, "bert", None))
        hook = backbone.register_forward_hook(
            lambda module, inputs, outputs: calls.append(outputs.last_hidden_state.device.index)
        )
        parallel = torch.nn.DataParallel(model, device_ids=[0, 1])
        result = parallel(**batch)
        result.loss.mean().backward()
        checks.equal("one forward per participating GPU replica", sorted(calls), [0, 1])
        checks.check(
            "parallel logits match serial",
            torch.allclose(result.logits, reference.logits, atol=2e-5, rtol=2e-5),
        )
        checks.check(
            "equal-supervision replica loss matches serial",
            torch.allclose(result.loss.mean(), reference.loss, atol=2e-5),
        )
        checks.check(
            "parallel shared gradient finite", torch.isfinite(model.classifier.weight.grad).all()
        )
        hook.remove()


def verify_device_policy(checks):
    arguments = SimpleNamespace(eval_strategy="epoch", n_gpu=1)
    copied = SimpleNamespace(device=torch.device("cpu"), _n_gpu=4)
    with patch("dataclasses.replace", return_value=copied):
        checks.equal("inference copy preserves replica pin", for_inference(arguments)._n_gpu, 1)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        checks.equal(
            "BF16 preferred when supported", default_precision(), {"fp16": False, "bf16": True}
        )
        with patch("torch.cuda.device_count", return_value=2):
            checks.raises(
                "indexed CUDA cannot silently change GPU",
                ValueError,
                predict_logits,
                None,
                None,
                None,
                device="cuda:1",
                match="CUDA_VISIBLE_DEVICES",
            )
    with patch("torch.cuda.is_available", return_value=False):
        checks.raises(
            "indexed CUDA unavailable is explicit",
            ValueError,
            resolve_device,
            "cuda:1",
            match="not available",
        )
    from warnings import catch_warnings, simplefilter

    fake_arguments = SimpleNamespace(
        use_cpu=False,
        device=torch.device("cuda:0"),
        _n_gpu=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        catch_warnings(),
    ):
        simplefilter("ignore", RuntimeWarning)
        checks.equal("single GPU policy pins replicas", apply_device_policy(fake_arguments, 1), 1)
        checks.equal("single GPU policy changes Trainer flag", fake_arguments._n_gpu, 1)
        checks.equal(
            "all policy restores visible replicas", apply_device_policy(fake_arguments, "all"), 4
        )
        checks.equal("all policy changes Trainer flag", fake_arguments._n_gpu, 4)
    with (
        patch("lab.ner.inference.prediction.build_training_arguments") as arguments,
        patch("lab.ner.inference.prediction._predict", return_value=np.zeros(1)),
    ):
        predict_logits(None, None, None, device="cpu")
        checks.check(
            "inference does not override resolved precision",
            "fp16" not in arguments.call_args.kwargs and "bf16" not in arguments.call_args.kwargs,
        )

    tracker = ResourceTracker.__new__(ResourceTracker)
    cuda = SimpleNamespace(
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=Mock(side_effect=lambda device=0: (device + 1) * 1024**3),
        max_memory_reserved=Mock(side_effect=lambda device=0: (device + 2) * 1024**3),
    )
    tracker._torch = SimpleNamespace(cuda=cuda)
    tracker._gpu_available = True
    tracker._device_ids = [0, 1]
    tracker._tracker = SimpleNamespace(start=Mock(), stop=Mock(), final_emissions_data=None)
    tracker.start()
    stats = tracker.stop()
    checks.equal(
        "VRAM peaks reset for each participating replica",
        [call.args[0] for call in cuda.reset_peak_memory_stats.call_args_list],
        [0, 1],
    )
    checks.equal("second GPU VRAM recorded", stats["gpu_peak_vram_allocated_gb__cuda_1"], 2.0)
    checks.equal(
        "legacy primary-GPU VRAM field preserved", stats["gpu_peak_vram_allocated_gb"], 1.0
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=1),
        patch("codecarbon.EmissionsTracker"),
    ):
        checks.equal("default tracker follows current device", ResourceTracker()._device_ids, [1])
        checks.equal(
            "tracker keeps explicit replica IDs",
            ResourceTracker(device_ids=[0, 1])._device_ids,
            [0, 1],
        )
        checks.check(
            "empty tuple disables GPU allocator tracking",
            not ResourceTracker(device_ids=())._gpu_available,
        )
        checks.check(
            "empty list disables GPU allocator tracking",
            not ResourceTracker(device_ids=[])._gpu_available,
        )


def verify_fold_policy(checks, base, corpus, root):
    from lab.core import prepare_dataset, write_corpus
    from lab.ner import train_model
    from lab.ner.training.devices import apply_device_policy

    source = root / "source.parquet"
    write_corpus(corpus, source)
    split = prepare_dataset(root / "split", source_parquet=source, validation_size=0.25)
    split_dir = split.split.paths["train"].parent
    with patch(
        "lab.ner.training.assessment.apply_device_policy", wraps=apply_device_policy
    ) as policy:
        train_model(
            split_dir,
            root / "fold-policy",
            str(base),
            architecture="multilabel",
            language="es",
            max_length=32,
            devices=1,
            track_resources=False,
            save_model=False,
            early_stopping_patience=None,
            training_arguments={
                "use_cpu": True,
                "max_steps": 1,
                "fp16": False,
                "disable_tqdm": True,
            },
        )
    checks.equal("device policy reapplied after fold argument copy", policy.call_count, 2)
    checks.check(
        "fold uses independent argument instance",
        policy.call_args_list[0].args[0] is not policy.call_args_list[1].args[0],
    )
    checks.equal("fold retains explicit device policy", policy.call_args_list[1].args[1], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-multi-gpu", action="store_true")
    arguments = parser.parse_args()
    if (arguments.require_cuda or arguments.require_multi_gpu) and not torch.cuda.is_available():
        parser.error("CUDA qualification requested but this environment has no CUDA device/build.")
    if arguments.require_multi_gpu and torch.cuda.device_count() < 2:
        parser.error("Multi-GPU qualification requires two or more CUDA devices.")
    torch.set_num_threads(2)
    checks = Checks("acceleration and parallel execution audit")
    verify_device_policy(checks)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer = local_tokenizer()
        corpus = overlap_corpus()
        base = tiny_base_model(root, tokenizer)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            verify_compaction(checks, device)
            verify_models(checks, base, tokenizer, corpus, device)
            verify_precision(checks, base, tokenizer, corpus, device)
            if device == "cpu":
                verify_native_training(checks, base, tokenizer, corpus, root, device)
            else:
                verify_native_training(checks, base, tokenizer, corpus, root, device, torch.float16)
                if torch.cuda.is_bf16_supported():
                    verify_native_training(
                        checks, base, tokenizer, corpus, root, device, torch.bfloat16
                    )
        verify_fold_policy(checks, base, corpus, root)
        if not torch.cuda.is_available():
            checks.skip("CUDA", "CPU-only Torch build; CPU results do not qualify CUDA")
        verify_multiple_gpus(checks, base, tokenizer, corpus)
    return checks.report()


if __name__ == "__main__":
    run(main)
