"""The causal LM that judges summaries: loaded once, it turns prompts into greedy answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.summarization.devices import resolve_device

Backend = Literal["transformers", "vllm"]

BACKENDS = ("transformers", "vllm")
DTYPES = ("float16", "bfloat16", "float32")
DEFAULT_DTYPE = "float16"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.70


@dataclass(frozen=True)
class JudgeModel:
    """A judge model loaded once, with its tokenizer and the token budget prompts must fit."""

    model: str
    backend: Backend
    tokenizer: Any
    generator: Any
    device: str
    description: dict[str, Any]

    @property
    def max_prompt_tokens(self) -> int:
        """How long a prompt may be and still leave room for the longest answer."""
        return self.description["max_model_len"] - self.description["max_new_tokens"]

    def prepare(self, prompt: str) -> tuple[list[int], bool]:
        """
        The token ids the model is given for `prompt`, and whether they were cut.

        The prompt is the user turn of the model's chat template. One longer than
        `max_prompt_tokens` loses its beginning, which is the source text, so the
        instruction and the JSON template at the end survive.
        """
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(formatted)

        if len(token_ids) <= self.max_prompt_tokens:
            return token_ids, False

        return token_ids[-self.max_prompt_tokens :], True

    def generate(self, prompt_token_ids: Sequence[list[int]]) -> list[str]:
        """The greedy answer to each prepared prompt: new text only, stripped."""
        if self.backend == "vllm":
            from vllm import SamplingParams

            sampling = SamplingParams(
                temperature=0.0, max_tokens=self.description["max_new_tokens"]
            )
            outputs = self.generator.generate(
                [{"prompt_token_ids": list(token_ids)} for token_ids in prompt_token_ids],
                sampling,
            )

            return [output.outputs[0].text.strip() for output in outputs]

        return [self._generate_one(token_ids) for token_ids in prompt_token_ids]

    def _generate_one(self, token_ids: list[int]) -> str:
        input_ids = torch.tensor([token_ids], device=self.device)
        pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = self.generator.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.description["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id if pad_token_id is None else pad_token_id,
            )

        new_ids = output_ids[0][len(token_ids) :]

        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def load_judge_model(
    model: str,
    backend: Backend = "transformers",
    dtype: str = DEFAULT_DTYPE,
    device: str = "auto",
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> JudgeModel:
    """
    Load an instruction-tuned causal LM to judge summaries with.

    `model` is a Hub identifier or a local snapshot, and must have a chat
    template. `backend="transformers"` generates one prompt at a time on
    `device`. `backend="vllm"` hands every prompt of a batch to one vLLM engine,
    several times faster on a GPU; vLLM pins its own torch, so it is not part of
    `lab[summarization]` and must be installed in the environment yourself.
    `tensor_parallel_size` and `gpu_memory_utilization` are vLLM's and apply to it
    only; 0.70 leaves room for BERTScore's encoder on the same GPU.

    Decoding is greedy, so a prompt always gets the same answer from the same
    model. `max_model_len` bounds prompt and answer together and
    `max_new_tokens` the answer, so a prompt keeps at most their difference.
    `dtype` defaults to float16, which MultiClinSum2's judges ran in.
    """
    _check_judge_arguments(
        backend,
        dtype,
        device,
        max_model_len,
        max_new_tokens,
        tensor_parallel_size,
        gpu_memory_utilization,
    )
    description = {
        "model": str(model),
        "backend": backend,
        "dtype": dtype,
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
    }

    if backend == "vllm":
        from vllm import LLM

        engine = LLM(
            model=str(model),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        tokenizer = engine.get_tokenizer()
        description |= {
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "vllm": metadata.version("vllm"),
        }
        resolved_device = "cuda"
    else:
        resolved_device = resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        engine = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype))
        engine.to(resolved_device)
        engine.eval()

    if tokenizer.chat_template is None:
        raise ValueError(
            f"{model} has no chat template. The judges send their prompts as a user turn, "
            "so the judge must be an instruction-tuned model."
        )

    description |= {"device": resolved_device, "transformers": metadata.version("transformers")}

    return JudgeModel(
        model=str(model),
        backend=backend,
        tokenizer=tokenizer,
        generator=engine,
        device=resolved_device,
        description=description,
    )


def _check_judge_arguments(
    backend: str,
    dtype: str,
    device: str,
    max_model_len: int,
    max_new_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}.")

    if max_new_tokens >= max_model_len:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room for a prompt in "
            f"max_model_len={max_model_len}. Lower max_new_tokens or raise max_model_len."
        )

    if backend == "vllm" and device != "auto":
        raise ValueError(
            f"device={device!r} only applies to backend='transformers'; vLLM places the "
            "model itself. Leave device='auto'."
        )

    vllm_only = {
        "tensor_parallel_size": (tensor_parallel_size, DEFAULT_TENSOR_PARALLEL_SIZE),
        "gpu_memory_utilization": (gpu_memory_utilization, DEFAULT_GPU_MEMORY_UTILIZATION),
    }
    changed = [name for name, (value, default) in vllm_only.items() if value != default]

    if backend != "vllm" and changed:
        raise ValueError(f"{changed} only apply to backend='vllm'; leave them at their defaults.")
