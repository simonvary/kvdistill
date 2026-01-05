"""
Ruler 4k Proxy Evaluation Script
Tests the model on a "Needle in a Haystack" task at 4k context length.
Compares:
1. Untrained Base Model (Full Cache)
2. Trained Student Model (Compressed Cache)
"""

import torch
import random
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from kvpress import KnormPress
from tqdm import tqdm

# ------------------------
# 1. Configuration
# ------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHECKPOINT_PATH = "./checkpoints/step_600" # Using the final checkpoint
CONTEXT_LENGTH = 4096
NUM_SAMPLES = 100  # Number of depths to test (0.0 to 1.0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Compression Settings for Trained Model
COMPRESSION_RATIO = 0.2
WINDOW_SIZE = 64

# ------------------------
# 2. Helpers
# ------------------------

NOISE_CORPUS = [
    "The weather in London is often rainy in the spring.", 
    "Quantization reduces the memory footprint of large models.",
    "Jupiter is the largest planet in our solar system.",
    "The history of Rome spans over two and a half thousand years.",
    "Photosynthesis is a process used by plants to convert light into energy.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence has seen rapid growth in the last decade.",
    "The mitochondria is the powerhouse of the cell.",
    "Shakespeare wrote Hamlet in the early 17th century.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "Coffee is one of the most popular beverages in the world.",
    "The internet was originally developed as ARPANET.",
    "Dolphins are known for their high intelligence and social behavior.",
    "The Great Wall of China is visible from space under certain conditions.",
    "Mathematics is often called the language of the universe.",
]

class RobustEvaluationPress(KnormPress):
    def __init__(self, compression_ratio=0.2, window_size=32):
        super().__init__(compression_ratio=compression_ratio)
        self.window_size = window_size

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        seq_len = keys.shape[2]
        
        # Safety Masking
        scores[:, :, 0] = float('inf') # Sink
        if seq_len > self.window_size:
            scores[:, :, -self.window_size:] = float('inf') # Window
            
        # Calculate Budget
        n_keep = int(seq_len * self.compression_ratio)
        min_keep = self.window_size + 1 
        if n_keep < min_keep: n_keep = min_keep
        if n_keep > seq_len: n_keep = seq_len

        topk_indices = torch.topk(scores, n_keep, dim=-1).indices
        topk_indices = topk_indices.sort(dim=-1).values
        
        def gather_tokens(tensor, indices):
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])
            return torch.gather(tensor, 2, expanded_indices)

        return gather_tokens(keys, topk_indices), gather_tokens(values, topk_indices)

def build_haystack(target_length_tokens):
    # Approx 4 chars per token
    target_chars = target_length_tokens * 4
    text = ""
    while len(text) < target_chars:
        text += random.choice(NOISE_CORPUS) + " "
    return text

def generate_niah_sample(tokenizer, context_length, depth):
    # Needle
    needle_code = str(random.randint(10000, 99999))
    needle = f"The special magic code is {needle_code}."
    question = "What is the special magic code?"
    answer = needle_code

    # Haystack
    # We want total length ~ context_length
    # Depth 0 = Start, Depth 1 = End
    
    # Reserve space for question and answer
    # Approx tokens
    q_len = 20
    n_len = 20
    
    haystack_len = context_length - q_len - n_len - 100 # buffer
    
    pre_needle_len = int(haystack_len * depth)
    post_needle_len = haystack_len - pre_needle_len
    
    pre_text = build_haystack(pre_needle_len)
    post_text = build_haystack(post_needle_len)
    
    # Construct full text
    full_context = pre_text + " " + needle + " " + post_text
    
    messages = [
        {"role": "user", "content": full_context + "\n\n" + question}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return input_text, answer

def evaluate_model(model, tokenizer, samples, use_press=False, description="Model"):
    print(f"\nEvaluating {description}...")
    correct = 0
    
    press = RobustEvaluationPress(compression_ratio=COMPRESSION_RATIO, window_size=WINDOW_SIZE)
    
    for i, (prompt, answer) in enumerate(tqdm(samples)):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Context Manager for Press
        # kvpress expects a *CausalLM* module (e.g. LlamaForCausalLM), not the inner LlamaModel.
        if use_press:
            if isinstance(model, PeftModel):
                # PeftModel wraps the CausalLM; kvpress doesn't support the LoRA wrapper itself.
                target_model = model.get_base_model()
            else:
                target_model = model

            ctx = press(target_model)
        else:
            # Null context manager
            from contextlib import nullcontext
            ctx = nullcontext()
            
        with torch.no_grad(), ctx:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if answer in generated_text:
            correct += 1
        else:
            # print(f"Fail at depth {i}: Expected {answer}, got '{generated_text}'")
            pass
            
    accuracy = correct / len(samples)
    print(f"{description} Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    return accuracy

# ------------------------
# 3. Main
# ------------------------

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Generating {NUM_SAMPLES} Samples (Context: {CONTEXT_LENGTH})...")
    samples = []
    depths = [i / (NUM_SAMPLES - 1) for i in range(NUM_SAMPLES)]
    for depth in depths:
        samples.append(generate_niah_sample(tokenizer, CONTEXT_LENGTH, depth))
        
    print("Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=DTYPE, 
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=bnb_config, 
        device_map=DEVICE,
        attn_implementation="flash_attention_2"
    )
    
    # 1. Evaluate Untrained Base Model (Full Cache)
    print("\n" + "="*50)
    print("BASELINE: Untrained Model (Full Cache)")
    print("="*50)
    evaluate_model(base_model, tokenizer, samples, use_press=False, description="Untrained Base Model (Full Cache)")

    # 1.5 Evaluate Untrained Base Model (Compressed Cache)
    print("\n" + "="*50)
    print(f"BASELINE: Untrained Model (Compressed Cache {COMPRESSION_RATIO*100}%)")
    print("="*50)
    evaluate_model(base_model, tokenizer, samples, use_press=True, description="Untrained Base Model (Compressed Cache)")
    
    # 2. Load Adapter
    print(f"\nLoading Adapter from {CHECKPOINT_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 3. Evaluate Trained Model (Compressed Cache)
    print("\n" + "="*50)
    print(f"EXPERIMENT: Trained Model (Compressed Cache {COMPRESSION_RATIO*100}%)")
    print("="*50)
    evaluate_model(model, tokenizer, samples, use_press=True, description="Trained Model (Compressed Cache)")

if __name__ == "__main__":
    main()
