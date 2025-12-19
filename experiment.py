import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from tqdm import tqdm
import json

def get_answer_token_id(tokenizer, ans_str):
    # Get the single token ID for the answer number
    # WARNING: This assumes the answer is a single token. 
    # For multi-token answers (e.g. 1541 might be [15, 41]), this is tricky.
    # Hack for speed: Just take the first token of the answer.
    tokens = tokenizer.encode(ans_str, add_special_tokens=False)
    return tokens[0]
@torch.no_grad()
def run_layer_sweep(model, dataset, batch_size=4):
    n_layers = model.cfg.n_layers
    n_samples = len(dataset)
    
    # Store results: [Layers, Samples]
    logit_diffs = torch.zeros((n_layers, n_samples))
    
    for i in tqdm(range(0, n_samples, batch_size)):
        batch = dataset[i : i+batch_size]
        
        # Prepare inputs
        clean_prompts = [b["clean_text"] for b in batch]
        corrupt_prompts = [b["corrupt_text"] for b in batch]
        
        # Tokenize (TransformerLens handles padding automatically if needed, 
        # but since we filtered for exact length, they should stack perfectly!)
        clean_tokens = model.to_tokens(clean_prompts)
        corrupt_tokens = model.to_tokens(corrupt_prompts)
        
        # Get target token IDs
        # We want: Logit(Clean_Ans) - Logit(Corrupt_Ans)
        clean_ans_ids = [get_answer_token_id(model.tokenizer, b["clean_ans"]) for b in batch]
        corrupt_ans_ids = [get_answer_token_id(model.tokenizer, b["corrupt_ans"]) for b in batch]
        
        clean_ans_t = torch.tensor(clean_ans_ids, device=model.cfg.device)
        corrupt_ans_t = torch.tensor(corrupt_ans_ids, device=model.cfg.device)

        # 1. Cache Clean Run (Last Token Only)
        clean_cache = {}
        def cache_last_token(resid_pre, hook):
            # Save [Batch, d_model] at pos -1
            clean_cache[hook.name] = resid_pre[:, -1, :].detach().clone()
            
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(get_act_name("resid_post", l), cache_last_token) for l in range(n_layers)]
        )
        
        # 2. Patching Sweep (Layer by Layer)
        for layer in range(n_layers):
            hook_name = get_act_name("resid_post", layer)
            
            def patch_last_token(resid_post, hook):
                # Overwrite last token with clean state
                resid_post[:, -1, :] = clean_cache[hook.name]
                return resid_post
                
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_last_token)]
            )
            
            # Calculate Metric on Last Token
            final_logits = logits[:, -1, :]
            
            clean_score = final_logits.gather(1, clean_ans_t.unsqueeze(1)).squeeze()
            corrupt_score = final_logits.gather(1, corrupt_ans_t.unsqueeze(1)).squeeze()
            
            diff = clean_score - corrupt_score
            logit_diffs[layer, i : i+len(batch)] = diff.cpu()
            
    return logit_diffs
