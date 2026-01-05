"""
KV-Cache Inference & Evaluation Script (Fixed Top-K Logic)
"""

print(">>> SCRIPT INITIALIZING... If you see this, Python is running the file.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from kvpress import KnormPress

# ------------------------
# 1. Configuration
# ------------------------

CHECKPOINT_PATH = "./checkpoints/step_400" 
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

COMPRESSION_RATIO = 0.2
WINDOW_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ------------------------
# 2. The Robust Press
# ------------------------

class RobustEvaluationPress(KnormPress):
    def __init__(self, compression_ratio, window_size=32):
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

        # If we want n_keep tokens but keep have seq_len at most
        if n_keep > seq_len:
            n_keep = seq_len

        topk_indices = torch.topk(scores, n_keep, dim=-1).indices
        topk_indices = topk_indices.sort(dim=-1).values
        
        def gather_tokens(tensor, indices):
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])
            return torch.gather(tensor, 2, expanded_indices)

        return gather_tokens(keys, topk_indices), gather_tokens(values, topk_indices)

# ------------------------
# 3. Model Loading
# ------------------------

def load_eval_model():
    print(f"Loading Base Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
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
    
    print(f"Loading LoRA Adapter from: {CHECKPOINT_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        #model = base_model
    except Exception as e:
        print(f"\n[ERROR] Could not load checkpoint from {CHECKPOINT_PATH}")
        print(f"Make sure the folder exists. Error details: {e}")
        exit(1)
        
    model.eval()
    return model, tokenizer

# ------------------------
# 4. Reasoning Tasks
# ------------------------

tasks = [
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
    "Frankie has a dog. Frankie has a cat. Frankie has a dog. Frankie has a cat. " * 50 + "Q: How many distinct types of pets does Frankie have mentioned above?\nA:",
    "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. He then loses 2 tennis balls. How many tennis balls does he have now?\nA:",
    "Please explain the concept of 'Cache Collapse' in one sentence using the word 'budget'.\nAnswer:"
]

# ------------------------
# 5. Execution
# ------------------------

def run_tests():
    print(">>> RUN_TESTS STARTED")
    model, tokenizer = load_eval_model()
    
    press = RobustEvaluationPress(
        compression_ratio=COMPRESSION_RATIO, 
        window_size=WINDOW_SIZE
    )
    
    print(f"\nStarting Evaluation with Ratio={COMPRESSION_RATIO} | Window={WINDOW_SIZE}")
    print("="*60)

    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}:")
        print("-" * 20)
        
        inputs = tokenizer(task, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad(), press(model.base_model.model):
        #with torch.no_grad(), press(model):
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                # --- STOP REPETITION LOOPS ---
                do_sample=True,          # Allow some creativity (breaks loops)
                temperature=0.7,         # Standard temperature
                repetition_penalty=1.2,  # Penalize repeating words
                no_repeat_ngram_size=3,  # Hard ban on 3-word loops
                # ---------------------------------- 
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        answer = generated_text[input_len:].strip()
        
        display_prompt = task #task[:100] + "..." if len(task) > 100 else task
        print(f"Prompt: {display_prompt}")
        print(f"Output: {answer}")
        print("="*60)

if __name__ == "__main__":
    run_tests()