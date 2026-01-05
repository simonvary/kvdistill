from datasets import load_dataset

# Simple toy training length configs
MAX_GSM8K_SAMPLES = 64  # you can increase later

def build_prompts_from_gsm8k(
    split: str = "train",
    max_samples: int = MAX_GSM8K_SAMPLES,
) -> list[str]:
    """
    Load a small subset of GSM8K and turn each question into a prompt.

    GSM8K fields:
        - "question": the word problem (requires multi-step reasoning)
        - "answer": solution with reasoning and final answer
    We only need the question as the prompt; the teacher will generate.
    """
    # "main" configuration is the usual one for openai/gsm8k
    ds = load_dataset("openai/gsm8k", "main", split=split)

    # Shuffle and take first max_samples for a quick test
    ds = ds.shuffle(seed=42)
    max_samples = min(max_samples, len(ds))
    ds = ds.select(range(max_samples))

    prompts: list[str] = []
    for ex in ds:
        q = ex["question"].strip()
        # Encourage reasoning-style output from the teacher
        prompt = q + "\n\nLet's think step by step."
        prompts.append(prompt)

    return prompts