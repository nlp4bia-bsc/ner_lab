"""The cross-encoder reranker: scoring mention-candidate pairs, and training on them."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from lab.nel.cross_encoder import examples
from lab.nel.cross_encoder.training import (
    DEFAULT_TRAINING_LOGGING_STEPS,
    configure_quiet_transformers_logging,
    supported_kwargs,
    training_dataloader,
)
from lab.nel.device import resolve_torch_device


class CrossEncoderReranker(CrossEncoder):
    """
    A SentenceTransformers `CrossEncoder` that reranks linking candidates and trains on them.

    `rerank` takes packed `terms`/`codes` lists per mention, the form
    `EntityLinkingPipeline` hands it. Training goes through SentenceTransformers'
    own fit loop with CUDA-safe DataLoaders: binary pairs (`fit_bce_pairs`),
    triplets expanded to pairs (`fit_triplets_as_bce_pairs`,
    `fit_knowledge_graph_triplets_bce`, `train_hard_triplets`), or a pairwise
    margin loss over triplets (`fit_triplets_margin_ranking`).

    With `model_type="mask"` the model has one output label and inputs are cut at
    `max_length`, or `max_seq_length` when that is not given.
    """

    pairs_to_labeled_texts = staticmethod(examples.pairs_to_labeled_texts)
    triplets_to_labeled_texts = staticmethod(examples.triplets_to_labeled_texts)
    triplets_to_texts = staticmethod(examples.triplets_to_texts)
    triplets_to_knowledge_graph_labeled_texts = staticmethod(
        examples.triplets_to_knowledge_graph_labeled_texts
    )
    prepare_triplets = staticmethod(examples.prepare_triplets)
    transform_triplets_rankingeval = staticmethod(examples.transform_triplets_rankingeval)

    def __init__(
        self,
        model_name: str,
        model_type: str = "mask",
        max_seq_length: int = 256,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int | None = None,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> None:
        resolved_device = resolve_torch_device(device)
        init_kwargs = {**kwargs, "device": resolved_device}
        max_length = (max_length or max_seq_length) if model_type == "mask" else None

        if model_type == "mask":
            init_kwargs.setdefault("num_labels", 1)
            init_kwargs["max_length"] = max_length

        super().__init__(model_name, **init_kwargs)
        self.resolved_device = resolved_device

        if max_length is not None:
            self.max_length = max_length

        self.model_type = model_type
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    def cuda(self, device: int | None = None) -> CrossEncoderReranker:
        """Move the reranker to CUDA, or to the CUDA device numbered `device`."""
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def to(self, device: str | Any) -> CrossEncoderReranker:
        """Move the underlying model to `device`."""
        self.resolved_device = str(device)

        if hasattr(self, "model") and hasattr(self.model, "to"):
            self.model.to(device)

        return self

    def rerank(
        self, mentions: list[str], candidates: list[dict[str, list[str]]], k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Each mention's packed `terms`/`codes` reordered by cross-encoder score, best `k` kept.

        Returns per mention its text, the reordered `terms` and `codes`, and their
        scores as `similarity`. All pairs are scored in one batched call.
        """
        if len(mentions) != len(candidates):
            raise ValueError("mentions and candidates must have the same length")

        terms = [[str(term) for term in packed.get("terms", [])] for packed in candidates]
        pairs = [
            (str(mention), term)
            for mention, mention_terms in zip(mentions, terms)
            for term in mention_terms
        ]
        scores = np.asarray(self._pair_scores(pairs)).reshape(-1)
        reranked = []
        start = 0

        for mention, mention_terms, packed in zip(mentions, terms, candidates):
            end = start + len(mention_terms)
            codes = [str(code) for code in packed.get("codes", [])]
            local = scores[start:end] if end > start else np.array([])
            order = np.argsort(local)[::-1] if len(local) else np.array([], dtype=int)
            reranked.append(
                {
                    "mention": str(mention),
                    "terms": [mention_terms[index] for index in order[:k]],
                    "codes": [codes[index] for index in order[:k]] if codes else [],
                    "similarity": [float(local[index]) for index in order[:k]],
                }
            )
            start = end

        return reranked

    def rerank_candidates(
        self,
        df: pd.DataFrame,
        entity_col: str,
        candidates_col: str,
        codes_col: str,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Reorder each row's candidate and code lists by cross-encoder score, in place."""
        if any(column not in df.columns for column in (entity_col, candidates_col, codes_col)):
            raise ValueError("Specified columns not found in the DataFrame")

        for index in tqdm(df.index, desc="Reranking candidates"):
            entity = df.at[index, entity_col]
            candidates = list(df.at[index, candidates_col])
            codes = list(df.at[index, codes_col])
            scores = self.predict(
                [[entity, candidate] for candidate in candidates], batch_size=batch_size
            )
            order = sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)
            df.at[index, candidates_col] = [candidates[position] for position in order]
            df.at[index, codes_col] = [codes[position] for position in order]

        return df

    def fit_quiet(self, **fit_kwargs: Any) -> Any:
        """
        Train with a progress bar but without the loss table every few hundred steps.

        Prefers SentenceTransformers' classic `old_fit` loop, which shows tqdm but no
        Trainer logging table; `use_legacy_fit_loop=False`, or a version without
        `old_fit`, uses `fit` with logging pushed out of reach. Keyword arguments the
        chosen loop does not take are dropped.
        """
        configure_quiet_transformers_logging()
        fit_kwargs.setdefault("show_progress_bar", True)
        fit_kwargs.setdefault("logging_steps", DEFAULT_TRAINING_LOGGING_STEPS)
        fit_kwargs.setdefault("evaluation_steps", 0)
        fit_kwargs.setdefault("disable_tqdm", False)

        if fit_kwargs.pop("use_legacy_fit_loop", True) and hasattr(self, "old_fit"):
            return self.old_fit(**supported_kwargs(self.old_fit, fit_kwargs))

        return self.fit(**supported_kwargs(self.fit, fit_kwargs))

    def fit_bce_pairs(
        self,
        pairs: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train with binary cross-entropy over `mention`/`candidate`/`label` pair records."""
        labeled = self.pairs_to_labeled_texts(pairs)
        fit_kwargs["train_dataloader"] = training_dataloader(
            [
                InputExample(texts=[mention, candidate], label=float(label))
                for mention, candidate, label in labeled
            ],
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            device=self.resolved_device,
        )

        return self.fit_quiet(**fit_kwargs)

    def fit_triplets_as_bce_pairs(
        self,
        triplets: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train with binary cross-entropy, each triplet giving a positive and a negative pair."""
        return self.fit_bce_pairs(
            _pair_records(self.triplets_to_labeled_texts(triplets)),
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            **fit_kwargs,
        )

    def fit_knowledge_graph_triplets_bce(
        self,
        triplets: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train with binary cross-entropy on the distinct pairs of the KnowledgeGraph recipe."""
        return self.fit_bce_pairs(
            _pair_records(self.triplets_to_knowledge_graph_labeled_texts(triplets)),
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            **fit_kwargs,
        )

    def fit_triplets_margin_ranking(
        self,
        triplets: Any,
        batch_size: int = 16,
        epochs: int = 1,
        learning_rate: float = 2e-5,
        margin: float = 1.0,
        use_amp: bool = True,
        show_progress_bar: bool = True,
        **_: Any,
    ) -> None:
        """
        Train so that `score(anchor, positive)` beats `score(anchor, negative)` by `margin`.

        A `MarginRankingLoss` over complete triplets, with AdamW and, on CUDA with
        `use_amp`, mixed precision.
        """
        triplets = self.triplets_to_texts(triplets)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_function = torch.nn.MarginRankingLoss(margin=margin)
        on_cuda = str(self.resolved_device).startswith("cuda")
        scaler = torch.cuda.amp.GradScaler() if use_amp and on_cuda else None
        self.model.train()

        for epoch in range(int(epochs)):
            for start in tqdm(
                range(0, len(triplets), batch_size),
                desc=f"Triplet margin epoch {epoch + 1}",
                disable=not show_progress_bar,
            ):
                batch = triplets[start : start + batch_size]
                optimizer.zero_grad()

                with torch.cuda.amp.autocast() if scaler is not None else contextlib.nullcontext():
                    loss = self._margin_loss(batch, loss_function)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

    def train_hard_triplets(
        self,
        df_hard_triplets: pd.DataFrame,
        output_path: str,
        batch_size: int,
        epochs: int,
        evaluator_type: str | None = None,
        optimizer_parameters: dict[str, float] | None = None,
        weight_decay: float = 0.01,
        evaluation_steps: int = 10000,
        save_best_model: bool = True,
        test_size: float | None = None,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        use_amp: bool = True,
        show_progress_bar: bool = True,
        use_legacy_fit_loop: bool = True,
    ) -> None:
        """
        Train on hard triplets the KnowledgeGraph way: distinct binary pairs, 10% warmup.

        `test_size` holds out that share of triplets, stratified by anchor where
        possible, for an `evaluator_type` of `"BinaryClassificationEvaluator"` or
        `"CERankingEvaluator"`. The model is written to `output_path`.
        """
        train_samples, dev_samples = _split_triplets(df_hard_triplets, test_size)
        train_dataloader = training_dataloader(
            self.prepare_triplets(train_samples),
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            device=self.resolved_device,
        )
        evaluator = None

        if evaluator_type and dev_samples is not None:
            evaluator = self._evaluator(evaluator_type, self.prepare_triplets(dev_samples))

        self.fit_quiet(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            optimizer_params={"lr": 1e-5} if optimizer_parameters is None else optimizer_parameters,
            weight_decay=weight_decay,
            evaluation_steps=evaluation_steps,
            warmup_steps=int(len(train_dataloader) * int(epochs) * 0.1),
            output_path=output_path,
            save_best_model=save_best_model,
            use_amp=bool(use_amp and str(self.resolved_device).startswith("cuda")),
            show_progress_bar=show_progress_bar,
            use_legacy_fit_loop=use_legacy_fit_loop,
        )

    def save_final(self, output_path: str | Path) -> None:
        """Save the cross-encoder, plus its transformer and tokenizer in safetensors form."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save(str(output_path))

        if hasattr(self, "model") and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(output_path), safe_serialization=True)

        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(output_path))

    def _pair_scores(self, pairs: list[tuple[str, str]]) -> Any:
        if not pairs:
            return np.array([])

        return self.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

    def _margin_loss(
        self, batch: list[tuple[str, str, str]], loss_function: torch.nn.MarginRankingLoss
    ) -> torch.Tensor:
        positive_scores = self._scores_tensor([(anchor, positive) for anchor, positive, _ in batch])
        negative_scores = self._scores_tensor([(anchor, negative) for anchor, _, negative in batch])

        return loss_function(positive_scores, negative_scores, torch.ones_like(positive_scores))

    def _scores_tensor(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        features = self.tokenizer(
            [left for left, _ in pairs],
            [right for _, right in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=getattr(self, "max_length", None),
        )
        features = {key: value.to(self.resolved_device) for key, value in features.items()}
        output = self.model(**features)
        logits = output.logits if hasattr(output, "logits") else output[0]

        return logits.view(-1).float()

    def _evaluator(self, evaluator_type: str, dev_examples: list[InputExample]) -> Any:
        from sentence_transformers.cross_encoder import evaluation

        if evaluator_type == "BinaryClassificationEvaluator":
            return evaluation.CEBinaryClassificationEvaluator.from_input_examples(
                dev_examples, name="dev"
            )

        if evaluator_type in {"CERankingEvaluator", "CERerankingEvaluator"}:
            return evaluation.CERerankingEvaluator(
                self.transform_triplets_rankingeval(dev_examples), name="dev"
            )

        raise ValueError(
            "evaluator_type must be BinaryClassificationEvaluator or CERankingEvaluator"
        )


def _pair_records(labeled: list[tuple[str, str, float]]) -> list[dict[str, Any]]:
    return [
        {"mention": mention, "candidate": candidate, "label": label}
        for mention, candidate, label in labeled
    ]


def _split_triplets(
    triplets: pd.DataFrame, test_size: float | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not test_size:
        return triplets, None

    stratify = (
        triplets["anchor"] if hasattr(triplets, "__getitem__") and "anchor" in triplets else None
    )

    try:
        return tuple(train_test_split(triplets, test_size=test_size, stratify=stratify))
    except ValueError:
        return tuple(train_test_split(triplets, test_size=test_size))
