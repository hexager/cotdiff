import torch
import random
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the same template for consistency
PROMPT_TEMPLATE = "User: Compute {} * {}. Show your working step by step, then give \"Final Answer: \" followed by the number on the last line.\nAssistant:"

def generate_natural_pairs(model, tokenizer, num_samples=50):
    dataset = []
    
    # Generators for Easy (1x1) and Hard (2x2)
    # We generate MORE than needed because we will discard mismatches
    candidates_easy = []
    for _ in range(num_samples * 5):
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        b_c = random.randint(2, 9)
        while b_c == b: b_c = random.randint(2, 9)
        candidates_easy.append((a, b, b_c, "easy"))

    candidates_hard = []
    for _ in range(num_samples * 5):
        a = random.randint(11, 99)
        b = random.randint(11, 99)
        b_c = random.randint(11, 99)
        while b_c == b: b_c = random.randint(11, 99)
        candidates_hard.append((a, b, b_c, "hard"))
        
    all_candidates = candidates_easy + candidates_hard
    
    print(f"Filtering {len(all_candidates)} candidates for Natural CoT Length Match...")
    
    count_easy = 0
    count_hard = 0
    
    for a, b, b_c, difficulty in tqdm(all_candidates):
        if difficulty == "easy" and count_easy >= num_samples: continue
        if difficulty == "hard" and count_hard >= num_samples: continue
        
        # 1. Generate Clean
        prompt_clean = PROMPT_TEMPLATE.format(a, b)
        inputs_clean = tokenizer(prompt_clean, return_tensors="pt").to(model.device)
        # We generate slightly more tokens to ensure we catch the answer
        out_clean = model.generate(**inputs_clean, max_new_tokens=200, do_sample=False, temperature=0.0)
        text_clean = tokenizer.decode(out_clean[0], skip_special_tokens=True)
        
        # 2. Generate Corrupt (Natural)
        prompt_corrupt = PROMPT_TEMPLATE.format(a, b_c)
        inputs_corrupt = tokenizer(prompt_corrupt, return_tensors="pt").to(model.device)
        out_corrupt = model.generate(**inputs_corrupt, max_new_tokens=200, do_sample=False, temperature=0.0)
        text_corrupt = tokenizer.decode(out_corrupt[0], skip_special_tokens=True)
        
        # 3. Extract Answers (Simple parsing)
        if "Final Answer:" not in text_clean or "Final Answer:" not in text_corrupt:
            continue
            
        ans_clean = text_clean.split("Final Answer:")[-1].strip()
        ans_corrupt = text_corrupt.split("Final Answer:")[-1].strip()
        
        # 4. CRITICAL: Length Match Check
        # We check the length of the GENERATED SEQUENCE (Tokens)
        len_clean = len(out_clean[0])
        len_corrupt = len(out_corrupt[0])
        
        if len_clean == len_corrupt:
            # Success!
            dataset.append({
                "clean_text": text_clean,
                "corrupt_text": text_corrupt,
                "clean_ans": ans_clean,
                "corrupt_ans": ans_corrupt,
                "type": difficulty,
                "length": len_clean
            })
            
            if difficulty == "easy": count_easy += 1
            else: count_hard += 1
            
    print(f"Collected {count_easy} Easy and {count_hard} Hard matched pairs.")
    return dataset

if __name__ == "__main__":
    # Test run (requires model loaded)
    pass 
f