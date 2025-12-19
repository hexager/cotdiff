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
        # 1. Prepare Inputs: STRIP THE ANSWER
        # We want the input to end at "Final Answer: "
        clean_prompts_stripped = []
        corrupt_prompts_stripped = []
        
        for b in batch:
            # Assumes b['clean_text'] ends with the answer number
            # We cut off the answer string
            clean_cut = b["clean_text"].replace(b["clean_ans"], "").strip()
            corrupt_cut = b["corrupt_text"].replace(b["corrupt_ans"], "").strip()
            clean_prompts_stripped.append(clean_cut)
            corrupt_prompts_stripped.append(corrupt_cut)
            
        clean_tokens = model.to_tokens(clean_prompts_stripped)
        corrupt_tokens = model.to_tokens(corrupt_prompts_stripped)
        
        # 2. Get Targets (The Answer Tokens)
        # Note: We need the Token ID that represents the answer (e.g. "15")
        # Be careful with whitespace! " 15" vs "15".
        # Best way: Tokenize the FULL text and take the token *after* the cut.
        
        clean_ans_ids = []
        corrupt_ans_ids = []
        
        for b in batch:
            full_toks = model.to_tokens(b["clean_text"])[0]
            # The answer token is the one that was stripped. 
            # Since we stripped from end, it's likely the last token(s).
            # Let's assume the answer is the LAST token of the full text (if single token).
            clean_ans_ids.append(full_toks[-1].item())
            
            full_toks_corr = model.to_tokens(b["corrupt_text"])[0]
            corrupt_ans_ids.append(full_toks_corr[-1].item())

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
