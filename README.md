# KV Distillation

`train_step_truncated(...)` in [kv_distill.py](kv_distill.py) trains a LoRA student to match a teacher **while the student runs with a compressed KV cache**.

## What it does

- Splits a long `input_ids` sequence into `block_size` chunks (prefill simulation).
- For each chunk:
  - Teacher forward pass with full cache (no compression).
  - Student forward pass using its (compressed) cache **plus explicit `position_ids`** (to fix RoPE).
  - KL distillation loss on logits (temperature `TEACHER_TEMP`) and one optimizer step.
  - Compress student `past_key_values` and carry the compressed cache into the next chunk.

Key implications:

- “Kept” tokens are **not permanent**: after the next chunk, previously kept tokens can be evicted (except forced-kept sink/window/EOS/EOT).

### How KV compression works (student only)

After each chunk, the student cache is detached and re-packed per layer using `KnormPress.score(...)` + top‑k selection.

Retention is **ratio + safety floor**:

- `target_budget = int(end_idx * COMPRESSION_RATIO)`
- `min_safe_budget = WINDOW_SIZE + MIN_MEMORY_BUFFER`
- `n_keep = clamp(max(target_budget, min_safe_budget), <= seq_len)`

This makes the kept KV length roughly `COMPRESSION_RATIO × end_idx` (per layer), unless you’re early in the sequence where the safety floor dominates.

Safety masking forces important positions to be kept:

- **Sink**: always keep token 0
- **Recency**: always keep last `WINDOW_SIZE` tokens
- **Boundaries**: always keep EOS and (if present) `<|eot_id|>`

### Parameters to tune

- `BLOCK_SIZE`: chunk size
- `COMPRESSION_RATIO`: target keep fraction
- `WINDOW_SIZE`: recency window always retained
- `MIN_MEMORY_BUFFER`: extra retained history

Generation-time compression is handled separately by `RobustEvaluationPress` in [kv_distill.py](kv_distill.py).

## Note on training strategies to get 4k/8k/16k lengths

For multiple target lengths (e.g., 4k/8k/16k), we could try a **curriculum → mixed sampling hybrid**:

- Phase 1 (warm start): mostly 4k until training is stable (e.g., first 10–20% of steps: 90% 4k / 10% 8k).
- Phase 2 (ramp): introduce 8k as a meaningful fraction (e.g., next 30–40%: 50% 4k / 50% 8k).
- Phase 3 (target mix): include 16k and gradually increase its share (e.g., start 40/40/20, end 25/35/40 for 4k/8k/16k).

Early 4k stabilizes general behavior and avoids cache-collapse; later mixing prevents catastrophic forgetting; slow 16k ramp reduces instability.

Practical tips: keep a small fixed eval set at each length and gate changes on all three; if compression is aggressive, ramp 16k more slowly.
