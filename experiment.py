import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


MODEL_NAME = "Qwen/Qwen2.5-1.5B" 
WEIGHTS_PATH = "Qwen/Qwen2.5-Math-1.5B"
print(f"Loading HERE {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(
    model_name=MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
    hf_model_name=WEIGHTS_PATH
)

with open("dataset_final.json", "r") as f:
    data = json.load(f)

easy_items = [d for d in data if d["type"] == "easy"]
hard_items = [d for d in data if d["type"] == "hard"]

print(f"Loaded {len(easy_items)} Easy and {len(hard_items)} Hard samples.")


def get_answer_token_ids(ans_str, model):
    """
    Converts answer string "1541" to the single token ID.
    WARNING: This assumes the answer tokenizes to a SINGLE token.
    For multi-token answers, we usually take the FIRST token.
    """
    # Note: Qwen might tokenize "1541" as multiple tokens.
    # We focus on the FIRST token of the answer for the Logit Diff.
    tokens = model.to_tokens(ans_str, prepend_bos=False).squeeze()
    if len(tokens.shape) == 0: return tokens.item()
    return tokens[0].item()

def run_layer_sweep(items, model, batch_size=5):
    results = torch.zeros((n_layers, len(items)))
    n_layers = model.cfg.n_layers

    for i in tqdm(range(0, len(items), batch_size), desc="Batching"):
        batch = items[i : i+batch_size]
        
        clean_prompts = [b["clean_text"] for b in batch]
        corrupt_prompts = [b["corrupt_text"] for b in batch]
        
        clean_tokens = model.to_tokens(clean_prompts)
        corrupt_tokens = model.to_tokens(corrupt_prompts)
        
        clean_ans_ids = torch.tensor([get_answer_token_ids(b["clean_ans"], model) for b in batch], device=model.cfg.device)
        corrupt_ans_ids = torch.tensor([get_answer_token_ids(b["corrupt_ans"], model) for b in batch], device=model.cfg.device)

        clean_cache = {}
        
        def cache_head_hook(resid_pre, hook):
            clean_cache[hook.name] = resid_pre[:, -1, :].detach().clone()
        filter_names = [get_act_name("resid_post", l) for l in range(n_layers)]
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(name, cache_head_hook) for name in filter_names]
        )
        
        for layer in range(n_layers):
            hook_name = get_act_name("resid_post", layer)
            
            def patch_hook(resid_post, hook):
                resid_post[:, -1, :] = clean_cache[hook.name]
                return resid_post
                
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_hook)]
            )
            last_token_logits = logits[:, -1, :]
            
            clean_logits = last_token_logits.gather(1, clean_ans_ids.unsqueeze(1)).squeeze()
            corrupt_logits = last_token_logits.gather(1, corrupt_ans_ids.unsqueeze(1)).squeeze()
            
            logit_diff = clean_logits - corrupt_logits
            
            # Store
            results[layer, i : i+len(batch)] = logit_diff.cpu()
            
    return results

print("Running Sweep on EASY items...")
easy_results = run_layer_sweep(easy_items, model)
torch.save(easy_results, "results_easy.pt")

print("Running Sweep on HARD items...")
hard_results = run_layer_sweep(hard_items, model)
torch.save(hard_results, "results_hard.pt")

print("Done! Results saved.")
