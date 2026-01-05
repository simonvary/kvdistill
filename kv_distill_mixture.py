"""\
KV-Cache Distillation with Dataset Mixtures

This script keeps the *training mechanics* from `kv_distill.py`:
- Teacher/student distillation via KL on logits
- Student KV-cache compression after each prefill chunk
- Explicit position_ids for the student

But it replaces the single GSM8K-only prompt stream with a *weighted mixture*
of datasets (instruction/chat + long-context/RAG-style + NIAH-style).

Notes
- Uses HuggingFace `datasets` for remote datasets.
- Also supports local JSONL datasets.
- Default dataset mixture is conservative (GSM8K only) to avoid accidental
  large downloads; pass `--mix ...` to enable additional sources.

Example
  python kv_distill_mixture.py \
    --mix gsm8k:0.2,ultrachat:0.6,openhermes:0.2 \
    --length_schedule curriculum \
    --steps 1000

"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    load_dataset = None
    _datasets_import_error = e

# Reuse the robust distillation + compression mechanics.
from kv_distill import (
    NOISE_CORPUS,
    DEVICE,
    DTYPE,
    TEACHER_TEMP,
    WINDOW_SIZE,
    safe_detach_cache,
    train_step_truncated,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


# ------------------------
# Dataset adapters
# ------------------------

ChatMessages = List[dict]


def _require_datasets() -> None:
    if load_dataset is None:
        raise RuntimeError(
            "The `datasets` package is required for remote datasets. "
            "Install it (pip install datasets) or use --local_jsonl. "
            f"Original import error: {_datasets_import_error}"
        )


def _safe_strip(x: Optional[str]) -> str:
    return (x or "").strip()


def _format_gsm8k_answer(ans: str) -> str:
    # GSM8K `answer` often contains reasoning + a final line like '#### 42'.
    # Keep the full answer to preserve the teacher's style/behavior.
    return _safe_strip(ans)


def iter_gsm8k(split: str, limit: int, seed: int) -> Iterator[ChatMessages]:
    _require_datasets()
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    for ex in ds:
        q = _safe_strip(ex.get("question"))
        a = _format_gsm8k_answer(ex.get("answer"))
        if not q or not a:
            continue
        yield [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]


def iter_ultrachat(split: str, limit: int, seed: int) -> Iterator[ChatMessages]:
    """UltraChat-like datasets usually expose a `messages` field."""
    _require_datasets()

    # Common dataset id. If you use a different UltraChat variant, adjust.
    # Note: this dataset uses split names like train_sft/test_sft/train_gen/test_gen.
    split_map = {
        "train": "train_sft",
        "training": "train_sft",
        "test": "test_sft",
        "validation": "test_sft",
        "val": "test_sft",
    }
    resolved_split = split_map.get(split, split)
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=resolved_split)
    except ValueError:
        # Fall back to a safe default rather than crash.
        resolved_split = "train_sft"
        print(f"[ultrachat] Unknown split '{split}', falling back to '{resolved_split}'")
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=resolved_split)
    ds = ds.shuffle(seed=seed)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    for ex in ds:
        msgs = ex.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue

        # Normalize roles/keys to {role, content}
        out: ChatMessages = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            role = _safe_strip(role).lower()
            content = _safe_strip(content)
            if role in {"human", "user"}:
                out.append({"role": "user", "content": content})
            elif role in {"assistant", "gpt", "bot"}:
                out.append({"role": "assistant", "content": content})

        if len(out) < 2:
            continue
        # Ensure last message is assistant to make a supervised sample.
        if out[-1]["role"] != "assistant":
            continue
        yield out


def iter_openhermes(split: str, limit: int, seed: int) -> Iterator[ChatMessages]:
    """OpenHermes-style instruction datasets often have instruction/output."""
    _require_datasets()

    # Common dataset id. If you use a different variant, adjust.
    ds = load_dataset("teknium/OpenHermes-2.5", split=split)
    ds = ds.shuffle(seed=seed)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    for ex in ds:
        # Many OpenHermes variants are ShareGPT-like: `conversations` with {from,value}
        conv = ex.get("conversations") or ex.get("conversation") or ex.get("messages")
        if isinstance(conv, list) and conv:
            out_msgs: ChatMessages = []
            for m in conv:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                role = _safe_strip(role).lower()
                content = _safe_strip(content)
                if not content:
                    continue
                if role in {"system"}:
                    out_msgs.append({"role": "system", "content": content})
                elif role in {"human", "user"}:
                    out_msgs.append({"role": "user", "content": content})
                elif role in {"assistant", "gpt", "bot"}:
                    out_msgs.append({"role": "assistant", "content": content})
            if len(out_msgs) >= 2 and out_msgs[-1]["role"] == "assistant":
                yield out_msgs
            continue

        # Fallback: instruction/output style
        instr = _safe_strip(ex.get("instruction") or ex.get("prompt") or ex.get("input"))
        out = _safe_strip(ex.get("output") or ex.get("response") or ex.get("chosen") or ex.get("text"))
        if not instr or not out:
            continue
        yield [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]


def infinite_repeat(
    make_iter: Callable[[int], Iterator[ChatMessages]],
    name: str,
) -> Iterator[ChatMessages]:
    """Repeat a finite iterator forever without caching all items.

    Also guards against the common pitfall where an adapter yields 0 samples:
    in that case we error early with a helpful message.
    """

    epoch = 0
    while True:
        any_yielded = False
        for item in make_iter(epoch):
            any_yielded = True
            yield item
        if not any_yielded:
            raise RuntimeError(
                f"Dataset adapter '{name}' produced 0 usable samples. "
                "Check split name and schema assumptions."
            )
        epoch += 1


def iter_hotpotqa(split: str, limit: int, seed: int, max_paragraphs: int = 10) -> Iterator[ChatMessages]:
    """HotpotQA: multi-hop QA. We build a long-context prompt with context."""
    _require_datasets()

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    ds = ds.shuffle(seed=seed)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    for ex in ds:
        q = _safe_strip(ex.get("question"))
        a = _safe_strip(ex.get("answer"))
        ctx = ex.get("context")

        if not q or not a or not isinstance(ctx, dict):
            continue

        titles = ctx.get("title")
        sents = ctx.get("sentences")
        if not isinstance(titles, list) or not isinstance(sents, list):
            continue

        blocks: List[str] = []
        for title, para_sents in zip(titles, sents):
            if len(blocks) >= max_paragraphs:
                break
            if not isinstance(para_sents, list):
                continue
            para = " ".join([_safe_strip(s) for s in para_sents if _safe_strip(s)])
            if not para:
                continue
            blocks.append(f"### {title}\n{para}")

        if not blocks:
            continue

        prompt = (
            "Use the following context to answer the question.\n\n"
            + "\n\n".join(blocks)
            + "\n\nQuestion: "
            + q
        )

        yield [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": a},
        ]


def iter_local_jsonl(path: str) -> Iterator[ChatMessages]:
    """Reads a local JSONL file.

    Supported formats per line:
      1) {"messages": [{"role": "user", "content": "..."}, ...]}
      2) {"prompt": "...", "response": "..."}
      3) {"instruction": "...", "output": "..."}

    A sample is used only if it ends with an assistant message.
    """

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(ex, dict) and isinstance(ex.get("messages"), list):
                msgs = ex["messages"]
                out: ChatMessages = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = _safe_strip(m.get("role")).lower()
                    content = _safe_strip(m.get("content"))
                    if role in {"user", "assistant", "system"} and content:
                        out.append({"role": role, "content": content})
                if len(out) >= 2 and out[-1]["role"] == "assistant":
                    yield out
                continue

            prompt = _safe_strip(ex.get("prompt") or ex.get("instruction"))
            response = _safe_strip(ex.get("response") or ex.get("output"))
            if prompt and response:
                yield [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]


DATASET_REGISTRY = {
    "gsm8k": iter_gsm8k,
    "ultrachat": iter_ultrachat,
    "openhermes": iter_openhermes,
    "hotpotqa": iter_hotpotqa,
}


# ------------------------
# Mixture sampling
# ------------------------


def parse_mix(spec: str) -> List[Tuple[str, float]]:
    """Parses a string like: 'gsm8k:0.2,ultrachat:0.6,openhermes:0.2'."""
    items: List[Tuple[str, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --mix entry '{part}'. Expected name:weight")
        name, w = part.split(":", 1)
        name = name.strip().lower()
        weight = float(w)
        if weight <= 0:
            continue
        items.append((name, weight))

    if not items:
        raise ValueError("--mix must contain at least one dataset with weight > 0")

    total = sum(w for _, w in items)
    return [(n, w / total) for n, w in items]


class WeightedSampler:
    def __init__(self, items: Sequence[Tuple[str, float]], seed: int):
        self.names = [n for n, _ in items]
        self.weights = [w for _, w in items]
        self.rng = random.Random(seed)

    def sample(self) -> str:
        return self.rng.choices(self.names, weights=self.weights, k=1)[0]


# ------------------------
# Length schedule
# ------------------------


@dataclass(frozen=True)
class LengthPhase:
    start_step: int
    end_step: int  # inclusive
    probs: Tuple[float, float, float]  # for (4k, 8k, 16k)


class LengthScheduler:
    def __init__(
        self,
        lengths: Tuple[int, int, int],
        phases: List[LengthPhase],
        seed: int,
    ):
        self.lengths = lengths
        self.phases = phases
        self.rng = random.Random(seed)

    def _phase_for_step(self, step: int) -> LengthPhase:
        for p in self.phases:
            if p.start_step <= step <= p.end_step:
                return p
        return self.phases[-1]

    def sample_length(self, step: int) -> int:
        p = self._phase_for_step(step)
        return self.rng.choices(self.lengths, weights=list(p.probs), k=1)[0]


def build_curriculum(num_steps: int) -> List[LengthPhase]:
    # A conservative curriculum:
    # - Warm: mostly 4k
    # - Ramp: 4k/8k
    # - Mix: introduce 16k and increase its share
    s1 = max(1, int(num_steps * 0.15))
    s2 = max(s1 + 1, int(num_steps * 0.55))

    return [
        LengthPhase(0, s1 - 1, (0.9, 0.1, 0.0)),
        LengthPhase(s1, s2 - 1, (0.5, 0.5, 0.0)),
        LengthPhase(s2, num_steps - 1, (0.25, 0.35, 0.40)),
    ]


# ------------------------
# Text building (chat template + optional haystack)
# ------------------------


def apply_chat(tokenizer, messages: ChatMessages) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Ensure an explicit EOS so the model learns to stop.
    return text + tokenizer.eos_token


def prepend_noise_to_reach_tokens(tokenizer, text: str, target_tokens: int, seed: int) -> str:
    """Prepends random noise sentences until token count >= target_tokens.

    This is a *mechanical* way to create long prefill sequences.
    For best behavior preservation, apply it primarily to long-context tasks.
    """

    if target_tokens <= 0:
        return text

    rng = random.Random(seed)

    # Fast path: check once.
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if ids.numel() >= target_tokens:
        return text

    # Build a noise prefix and re-check periodically.
    noise_parts: List[str] = []
    # Avoid excessive tokenizer calls; re-check every N sentences.
    check_every = 16
    iters = 0

    while True:
        noise_parts.append(rng.choice(NOISE_CORPUS))
        iters += 1
        if iters % check_every != 0:
            continue
        candidate = " ".join(noise_parts) + "\n\n" + text
        ids = tokenizer(candidate, return_tensors="pt", truncation=False)["input_ids"][0]
        if ids.numel() >= target_tokens:
            return candidate


# ------------------------
# Model setup
# ------------------------


def load_models_and_tokenizer(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    print(f"Loading models from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading Teacher (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_quant_type="nf4",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        device_map=DEVICE,
    )
    teacher.eval()

    print("Loading Student...")
    student_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        dtype=DTYPE,
        device_map=DEVICE,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    student = get_peft_model(student_base, lora_config)
    student.train()
    return teacher, student, tokenizer


# ------------------------
# Main training loop
# ------------------------


def build_iterators(
    mix: List[Tuple[str, float]],
    split: str,
    limit: int,
    seed: int,
    local_jsonl: Optional[str],
) -> dict[str, Iterator[ChatMessages]]:
    iters: dict[str, Iterator[ChatMessages]] = {}

    for name, _ in mix:
        if name == "local_jsonl":
            if not local_jsonl:
                raise ValueError("--local_jsonl must be provided when mix includes local_jsonl")
            iters[name] = infinite_repeat(
                make_iter=lambda epoch, p=local_jsonl: iter_local_jsonl(p),
                name=name,
            )
            continue

        fn = DATASET_REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown dataset '{name}'. Known: {sorted(DATASET_REGISTRY.keys()) + ['local_jsonl']}"
            )

        iters[name] = infinite_repeat(
            make_iter=lambda epoch, f=fn, n=name: f(split=split, limit=limit, seed=seed + epoch),  # type: ignore[misc]
            name=name,
        )

    return iters


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--learning_rate", type=float, default=1e-5)

    ap.add_argument(
        "--mix",
        type=str,
        default="gsm8k:1.0",
        help="Weighted dataset mixture, e.g. 'gsm8k:0.2,ultrachat:0.6,openhermes:0.2'",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF datasets split (train/validation/test depending on dataset)",
    )
    ap.add_argument(
        "--dataset_limit",
        type=int,
        default=5000,
        help="Max examples to load per dataset (0 = all).",
    )
    ap.add_argument(
        "--local_jsonl",
        type=str,
        default=None,
        help="Path to local JSONL file if you include 'local_jsonl' in --mix",
    )

    ap.add_argument(
        "--lengths",
        type=str,
        default="4096,8192,16384",
        help="Comma-separated token lengths for curriculum (3 values).",
    )
    ap.add_argument(
        "--length_schedule",
        choices=["curriculum", "uniform"],
        default="curriculum",
    )

    ap.add_argument("--block_size", type=int, default=1024)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument(
        "--pad_with_noise",
        action="store_true",
        help="If set, prepend noise to reach the sampled token length.",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mix = parse_mix(args.mix)
    sampler = WeightedSampler(mix, seed=args.seed)

    lengths_list = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if len(lengths_list) != 3:
        raise ValueError("--lengths must contain exactly 3 comma-separated integers")
    lengths = (lengths_list[0], lengths_list[1], lengths_list[2])

    if args.length_schedule == "curriculum":
        phases = build_curriculum(args.steps)
    else:
        phases = [LengthPhase(0, args.steps - 1, (1.0 / 3, 1.0 / 3, 1.0 / 3))]

    length_sched = LengthScheduler(lengths=lengths, phases=phases, seed=args.seed + 17)

    teacher, student, tokenizer = load_models_and_tokenizer(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    optimizer = AdamW(student.parameters(), lr=args.learning_rate)

    dataset_iters = build_iterators(
        mix=mix,
        split=args.split,
        limit=args.dataset_limit,
        seed=args.seed,
        local_jsonl=args.local_jsonl,
    )

    print("Starting mixture distillation...")
    print(f"Mixture: {mix}")
    print(f"Lengths: {lengths} | schedule={args.length_schedule} | pad_with_noise={args.pad_with_noise}")

    for step in range(args.steps):
        ds_name = sampler.sample()
        messages = next(dataset_iters[ds_name])

        target_len = length_sched.sample_length(step)
        text = apply_chat(tokenizer, messages)

        if args.pad_with_noise:
            # Use a deterministic but step-dependent seed so runs are reproducible.
            text = prepend_noise_to_reach_tokens(tokenizer, text, target_tokens=target_len, seed=args.seed + step)

        inputs = tokenizer([text], return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(DEVICE)

        # Skip pathological tiny sequences (should be rare).
        if input_ids.size(1) < 8:
            continue

        loss = train_step_truncated(teacher, student, optimizer, input_ids, args.block_size, tokenizer)

        if step % 10 == 0:
            print(
                f"Step {step:5d} | loss={loss:.4f} | ds={ds_name} | seq={input_ids.size(1)} | target={target_len}"
            )

        if step % 200 == 0 and step > 0:
            os.makedirs("./checkpoints", exist_ok=True)
            student.save_pretrained(f"./checkpoints/step_{step}")


if __name__ == "__main__":
    main()
