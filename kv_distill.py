"""
KV-Cache Distillation Script (Final Robust Version)
Fixes: 
1. Cache Collapse (via Manual Budget)
2. Context Loss (via Sink/Window Protection)
3. RoPE/Position Errors (via Explicit Position IDs)
"""
import random
import itertools
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from kvpress import KnormPress
from dataset_helpers import build_prompts_from_gsm8k, MAX_GSM8K_SAMPLES

# ------------------------
# 1. Configuration
# ------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BLOCK_SIZE = 1024
WINDOW_SIZE = 64  # For Safety Window in Cache
COMPRESSION_RATIO = 0.2  # Keep COMPRESSION_RATIO of TOTAL sequence
TEACHER_TEMP = 2.0
LEARNING_RATE = 1e-5
NUM_STEPS = 601
HAYSTACK_SIZE = 2000  # Number of tokens in the 'haystack' context
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

WANDB_PROJECT = "kv-distill-final"
WANDB_RUN_NAME = f"llama3-robust-ratio{COMPRESSION_RATIO}"

# ------------------------
# 2. Setup
# ------------------------

def load_models_and_tokenizer():
    print(f"Loading models from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    print("Loading Teacher (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_type="nf4")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="flash_attention_2", quantization_config=bnb_config, device_map=DEVICE
    )
    teacher.eval()

    print("Loading Student...")
    # NOTE: We use standard Flash Attention 2, but we MUST handle positions manually.
    student_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="flash_attention_2", torch_dtype=DTYPE, device_map=DEVICE
    )
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, lora_config)
    student.train()

    return teacher, student, tokenizer

def safe_detach_cache(cache):
    if cache is None: return None
    new_cache = DynamicCache()
    for layer_idx in range(len(cache)):
        layer_k, layer_v = cache[layer_idx]
        new_cache.update(layer_k.detach(), layer_v.detach(), layer_idx)
    return new_cache


# A corpus of "Noise" text (Distinct sentences to prevent repetition loops)
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
    # ... (The model won't memorize these, it just needs to process them as 'context')
]

def build_needle_in_haystack(needle_prompt, target_length=2000):
    """
    Hides the 'needle' (math question) inside a 'haystack' (random text).
    This fills the cache WITHOUT teaching the model to repeat itself.
    """
    current_text = ""
    
    # 1. Fill 90% of the context with random noise
    while len(current_text) < target_length * 3: # *3 approx char->token ratio
        sentence = random.choice(NOISE_CORPUS)
        current_text += sentence + " "
        
    # 2. Insert the Needle (The actual prompt) at the VERY END
    # This ensures the model has to attend to the end (Recency) 
    # while holding the noise in memory.
    final_text = current_text + "\n\n" + needle_prompt
    return final_text

def build_needle_in_haystack_chat(tokenizer, question, answer, target_length=2000):
    """
    1. Creates a 'Haystack' of noise.
    2. Formats the 'Needle' (Q&A) using the Official Llama-3 Chat Template.
    3. Appends the EOS token explicitly to teach the model to STOP.
    """
    
    # 1. Format the Needle as a Chat
    # We treat the 'Noise' as a system context or just distinct text preceding the chat.
    # Ideally, we put the noise in the 'system' prompt or just prepend it raw.
    # Prepending raw is harder for the model (good for robustness testing).
    
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    # This generates: "<|begin_of_text|><|start_header_id|>user... Q ...<|eot_id|>... A ...<|eot_id|>"
    chat_formatted_needle = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # 2. Build the Haystack (Noise)
    current_text = ""
    while len(current_text) < target_length * 3: 
        sentence = random.choice(NOISE_CORPUS)
        current_text += sentence + " "
    
    # 3. Combine: Noise + Chat + EXPLICIT EOS
    # We append tokenizer.eos_token to ensure the loss function sees the "STOP" signal.
    final_text = current_text + "\n\n" + chat_formatted_needle + tokenizer.eos_token
    
    return final_text

# ------------------------
# 3. Training Loop
# ------------------------

def train_step_truncated(teacher, student, optimizer, input_ids, block_size, tokenizer):
    total_seq_len = input_ids.size(1)
    teacher_cache = DynamicCache()
    student_cache = DynamicCache()
    
    total_loss = 0
    num_blocks = 0
    
    # Use your harsh ratio here (e.g., 0.1)
    scorer = KnormPress(compression_ratio=COMPRESSION_RATIO)
    
    # --- CONFIGURATION FOR SAFETY ---
    # Guarantees the "48" and "April" survive in short prompts
    MIN_MEMORY_BUFFER = 32  
    SAFE_WINDOW = WINDOW_SIZE 

    # print(f"\n{'='*20} START TRAINING STEP {'='*20}")

    for i in range(0, total_seq_len, block_size):
        end_idx = min(i + block_size, total_seq_len)
        if end_idx - i < 16: break
        
        chunk = input_ids[:, i:end_idx]
        chunk_len = chunk.size(1)
        
        # 1. Explicit Position IDs (RoPE Fix)
        current_pos_ids = torch.arange(i, i + chunk_len, device=DEVICE).unsqueeze(0)

        # 2. Forward Passes
        with torch.no_grad():
            t_out = teacher(chunk, past_key_values=teacher_cache, use_cache=True)
            teacher_cache = t_out.past_key_values
            teacher_logits = t_out.logits

        s_out = student(
            chunk, 
            past_key_values=student_cache, 
            use_cache=True, 
            position_ids=current_pos_ids 
        )
        student_logits = s_out.logits
        
        # 3. Loss Calculation
        s_logits_flat = student_logits.view(-1, student_logits.size(-1))
        t_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        loss = F.kl_div(
            F.log_softmax(s_logits_flat / TEACHER_TEMP, dim=-1),
            F.softmax(t_logits_flat / TEACHER_TEMP, dim=-1),
            reduction='batchmean', 
            log_target=False
        ) * (TEACHER_TEMP ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_blocks += 1

        # -----------------------------------------------------------
        # FIX: Dynamic Floor Compression (Safe for Training)
        # -----------------------------------------------------------
        raw_cache = s_out.past_key_values
        detached_cache = safe_detach_cache(raw_cache)

        # Identify Special Tokens (EOS, EOT) in the current sequence
        # We want to force-keep these so the model knows when to stop.
        current_seq = input_ids[:, :end_idx] # (batch, seq_len)
        
        eos_id = tokenizer.eos_token_id
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        # Create mask (1 for special tokens, 0 otherwise)
        special_mask = (current_seq == eos_id)
        # Check if eot_id is valid (usually it is for Llama 3)
        if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
            special_mask = special_mask | (current_seq == eot_id)
        
        # A. Calculate Harsh Target (e.g. 10% of 100 = 10 tokens)
        target_budget = int(end_idx * COMPRESSION_RATIO)
        
        # B. Apply Safety Floor (Window + 32 tokens history)
        # If we have 100 tokens, 10% is 10. But we NEED 64+32=96.
        # So we force budget to 96.
        min_safe_budget = SAFE_WINDOW + MIN_MEMORY_BUFFER
        
        real_budget = max(target_budget, min_safe_budget)
        
        # Clamp if sequence is shorter than the floor
        if real_budget > end_idx:
            real_budget = end_idx
            
        compressed_cache = DynamicCache()
        
        for layer_idx in range(len(detached_cache)):
            k, v = detached_cache[layer_idx]
            seq_len = k.shape[2]
            
            # Score
            scores = scorer.score(
                module=None, hidden_states=None, 
                keys=k, values=v, attentions=None, 
                kwargs={"attention_mask": None}
            )
            
            # Masking (Sink + Window)
            scores[:, :, 0] = float('inf')
            if seq_len > SAFE_WINDOW:
                scores[:, :, -SAFE_WINDOW:] = float('inf')

            # Masking (Special Tokens: EOS/EOT)
            # Expand mask to (batch, num_heads, seq_len)
            # scores is (batch, heads, seq_len)
            if special_mask.size(1) == seq_len:
                mask_expanded = special_mask.unsqueeze(1).expand(-1, scores.size(1), -1)
                scores[mask_expanded] = float('inf')

            # Select Top-K using the REAL (Safe) Budget
            n_keep = min(real_budget, seq_len)
            
            topk_indices = torch.topk(scores, n_keep, dim=-1).indices
            topk_indices = topk_indices.sort(dim=-1).values
            
            def gather_tokens(tensor, indices):
                expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])
                return torch.gather(tensor, 2, expanded_indices)
                
            k_new = gather_tokens(k, topk_indices)
            v_new = gather_tokens(v, topk_indices)
            
            compressed_cache.update(k_new, v_new, layer_idx)

        student_cache = compressed_cache
        
        # Optional: Print status to verify it's working
        k_sample, _ = student_cache[0]
        print(f"   >>> KVPRESS: Hist: {end_idx} -> Kept: {k_sample.size(2)} (Target Ratio: {COMPRESSION_RATIO} | Floor: {min_safe_budget})")

    return total_loss / max(1, num_blocks)



class RobustEvaluationPress(KnormPress):
    """
    Replicates the 'training' logic for model.generate()
    Ensures Sink + Window + Knorm Budget are applied during inference.
    """
    def __init__(self, compression_ratio, window_size=32):
        super().__init__(compression_ratio=compression_ratio)
        self.window_size = window_size

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # 1. Score (L2 Norm)
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        seq_len = keys.shape[2]
        
        # 2. Safety Masking (Sink + Window) - Same as training
        # Always keep Token 0
        scores[:, :, 0] = float('inf')
        
        # Always keep last N
        if seq_len > self.window_size:
            scores[:, :, -self.window_size:] = float('inf')
            
        # 3. Determine Top-K based on Ratio
        n_keep = int(seq_len * self.compression_ratio)
        if n_keep < self.window_size: n_keep = self.window_size

        # 4. Select and Gather
        topk_indices = torch.topk(scores, n_keep, dim=-1).indices
        topk_indices = topk_indices.sort(dim=-1).values
        
        def gather_tokens(tensor, indices):
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])
            return torch.gather(tensor, 2, expanded_indices)

        return gather_tokens(keys, topk_indices), gather_tokens(values, topk_indices)

def evaluate_reasoning(teacher, student, tokenizer, step):
    print(f"\n{'-'*20} EVALUATING REASONING (Step {step}) {'-'*20}")
    
    # A specific math reasoning question
    question = "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:"
    
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)
    
    # 1. Teacher (Baseline - Full Cache)
    # We want to confirm the teacher knows the answer.
    with torch.no_grad():
        t_out = teacher.generate(
            **inputs, 
            max_new_tokens=60, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    t_text = tokenizer.decode(t_out[0], skip_special_tokens=True)
    
    # 2. Student (Compressed - Robust Logic)
    # We apply the Press hook to the base model
    press = RobustEvaluationPress(compression_ratio=COMPRESSION_RATIO, window_size=32)
    
    # Target the inner model for hooks
    student_base = student.base_model.model 
    
    with torch.no_grad(), press(student_base):
        s_out = student.generate(
            **inputs, 
            max_new_tokens=60, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    s_text = tokenizer.decode(s_out[0], skip_special_tokens=True)
    
    print(f"[Teacher]: {t_text[len(question):].strip()}")
    print(f"[Student]: {s_text[len(question):].strip()}")
    print("-" * 60 + "\n")



# ------------------------
# 4. Main
# ------------------------

def main_old():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    teacher, student, tokenizer = load_models_and_tokenizer()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)
    
    prompts = build_prompts_from_gsm8k(max_samples=100) 
    prompt_iter = itertools.cycle(prompts)
    
    print("Starting Distillation Loop (Needle-in-Haystack Mode)...")
    
    for step in range(NUM_STEPS):
        base_prompt = next(prompt_iter)
        
        # --- FIX: USE NOISE, NOT REPETITION ---
        # Old: long_text_batch = [base_prompt * 50]
        # New:
        long_text = build_needle_in_haystack(base_prompt, target_length=HAYSTACK_SIZE)
        long_text_batch = [long_text]
        # --------------------------------------

        inputs = tokenizer(long_text_batch, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(DEVICE)
        
        # Ensure we have enough data to trigger compression
        if input_ids.size(1) < BLOCK_SIZE: continue

        loss = train_step_truncated(teacher, student, optimizer, input_ids, BLOCK_SIZE, tokenizer)
        
        wandb.log({"train/loss": loss, "step": step})
        print(f"Step {step} | Loss: {loss:.4f} | Total SeqLen: {input_ids.size(1)}")
        
        # Evaluate less frequently to save time
        if step % 50 == 0 and step > 0:
            evaluate_reasoning(teacher, student, tokenizer, step)
            student.save_pretrained(f"./checkpoints/step_{step}")
            
    wandb.finish()

def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    teacher, student, tokenizer = load_models_and_tokenizer()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE) # Use 5e-5 as discussed
    
    # Load dataset raw (so we can split Q and A)
    # Note: You might need to adjust 'build_prompts_from_gsm8k' to return tuples (q, a)
    # For now, assuming prompts are "Q: ... \nA: ..." strings:
    prompts = build_prompts_from_gsm8k(max_samples=100) 
    prompt_iter = itertools.cycle(prompts)
    
    print("Starting Distillation Loop (Chat Mode + EOS Training)...")
    
    for step in range(NUM_STEPS):
        raw_prompt = next(prompt_iter)
        
        # Quick parse to separate Question and Answer for the template
        # (Assuming standard "Q: ... A: ..." format from your helper)
        try:
            q_part, a_part = raw_prompt.split("A:", 1)
            q_part = q_part.replace("Q:", "").strip()
            a_part = a_part.strip()
        except ValueError:
            # Fallback if split fails
            q_part = raw_prompt
            a_part = "Solution..."

        # --- FIX: USE CHAT TEMPLATE ---
        long_text = build_needle_in_haystack_chat(tokenizer, q_part, a_part, target_length=2000)
        long_text_batch = [long_text]
        # ------------------------------

        inputs = tokenizer(long_text_batch, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(DEVICE)
        
        if input_ids.size(1) < BLOCK_SIZE: continue

        # (Use the Robust/Dynamic Floor train_step we fixed previously)
        loss = train_step_truncated(teacher, student, optimizer, input_ids, BLOCK_SIZE, tokenizer)
        
        wandb.log({"train/loss": loss, "step": step})
        print(f"Step {step} | Loss: {loss:.4f} | Total SeqLen: {input_ids.size(1)}")
        
        if step % 50 == 0 and step > 0:
            evaluate_reasoning(teacher, student, tokenizer, step)
            student.save_pretrained(f"./checkpoints/step_{step}")
            
    wandb.finish()

if __name__ == "__main__":
    main()