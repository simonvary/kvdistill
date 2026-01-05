# KV Distillation

`train_step_truncated(...)` in [kv_distill.py](kv_distill.py) trains a LoRA student to match a teacher **while the student runs with a compressed KV cache**.

## What it does

- Splits a long `input_ids` sequence into `block_size` chunks (prefill simulation).
- For each chunk:
  - Teacher forward pass with full cache (no compression).
  - Student forward pass using its (compressed) cache **plus explicit `position_ids`** (RoPE fix).
  - KL distillation loss on logits (temperature `TEACHER_TEMP`) and one optimizer step.
  - Compress student `past_key_values` and carry the compressed cache into the next chunk.

Key implications:

- The compression budget is recomputed **every chunk** based on the prefix length so far (`end_idx`).
- “Kept” tokens are **not permanent**: after the next chunk, previously kept tokens can be evicted (except forced-kept sink/window/EOS/EOT).

### Why explicit `position_ids`

Because the student cache is compressed, “cache length” != true token position. Passing `position_ids = i..i+chunk_len-1` keeps RoPE aligned to the real sequence positions.

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
