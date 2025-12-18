import torch
import random
import json
from tqdm import tqdm


def generate_raw_pairs(num_samples=50):
    data = {"easy": [], "hard": []}
    
    # Easy: 1x1 digits
    for _ in range(num_samples):
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        b_c = random.randint(2, 9)
        while b_c == b: b_c = random.randint(2, 9)
        data["easy"].append((a, b, b_c, a*b, a*b_c))

    # Hard: 2x2 digits
    for _ in range(num_samples):
        a = random.randint(11, 99)
        b = random.randint(11, 99)
        b_c = random.randint(11, 99)
        while b_c == b: b_c = random.randint(11, 99)
        data["hard"].append((a, b, b_c, a*b, a*b_c))
        
    return data

def precompute_cot(model, raw_data, prompt_template):
    final_dataset = []
    
    print(f"Generating CoT traces for {len(raw_data)} samples...")
    
    for a, b, b_c, ans_clean, ans_corr in tqdm(raw_data):
        user_prompt = prompt_template.format(a, b)
        try:
            output = model.generate(
                user_prompt, 
                max_new_tokens=200, 
                stop_at_eos=False, # We want to stop manually or specific token
                temperature=0 # Deterministic
            )

            full_text = output
            if "Final Answer:" in full_text:
                pre_answer_text = full_text.split("Final Answer:")[0] + "Final Answer:"
            else:
                continue
            
            clean_full = user_prompt + pre_answer_text[len(user_prompt):]
            prompt_corrupt = prompt_template.format(a, b_c)
            corrupt_full = prompt_corrupt + pre_answer_text[len(user_prompt):]
            
            final_dataset.append({
                "clean_text": clean_full,
                "corrupt_text": corrupt_full,
                "clean_ans": str(ans_clean),
                "corrupt_ans": str(ans_corr),
                "type": "hard" if a > 10 else "easy"
            })
            
        except Exception as e:
            print(f"Error on {a}*{b}: {e}")
            continue

    return final_dataset

PROMPT_TEMPLATE = "User: Compute {} * {}. Show your working step by step, then give \"Final Answer: \" followed by the number on the last line.\nAssistant:"
