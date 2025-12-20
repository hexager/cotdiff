import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from tqdm import tqdm
import json

def get_next_token_id(model, full_text, stripped_text):
    """
    Identifies the CORRECT next token the model should predict.
    Logic: Tokenize(Full) vs Tokenize(Stripped). The first mismatch is the target.
    """
    # Note: add_special_tokens=False is usually safer for mid-sentence arithmetic 
    # but depends on your specific template. We use the model's standard encoding.
    full_tokens = model.to_tokens(full_text)[0]
    stripped_tokens = model.to_tokens(stripped_text)[0]
    
    # Validation: Stripped should be a prefix of Full
    n_context = len(stripped_tokens)
    
    # If the lengths are the same, we can't predict anything (Answer was empty?)
    if n_context >= len(full_tokens):
        # Fallback: Just return the last token (unsafe, but handles edge cases)
        return full_tokens[-1]
        
    # The target is the first token *after* the stripped sequence
    return full_tokens[n_context]

@torch.no_grad()
def run_layer_sweep(model, dataset, batch_size=4):
    n_layers = model.cfg.n_layers
    n_samples = len(dataset)
    
    # Store results: [Layers, Samples]
    logit_diffs = torch.zeros((n_layers, n_samples))
    
    print(f"Starting sweep on {n_samples} samples with Full-Sequence Patching...")
    
    for i in tqdm(range(0, n_samples, batch_size)):
        batch = dataset[i : i+batch_size]      
        
        # --- 1. PREPARE INPUTS ---
        clean_prompts_stripped = []
        corrupt_prompts_stripped = []
        clean_target_ids = []
        corrupt_target_ids = []
        
        for b in batch:
            # Strip answer from text
            clean_cut = b["clean_text"].replace(b["clean_ans"], "").strip()
            corrupt_cut = b["corrupt_text"].replace(b["corrupt_ans"], "").strip()
            
            clean_prompts_stripped.append(clean_cut)
            corrupt_prompts_stripped.append(corrupt_cut)
            
            # Identify the correct next token (The first token of the answer)
            c_target = get_next_token_id(model, b["clean_text"], clean_cut)
            corr_target = get_next_token_id(model, b["corrupt_text"], corrupt_cut)
            
            clean_target_ids.append(c_target.item())
            corrupt_target_ids.append(corr_target.item())
            
        clean_tokens = model.to_tokens(clean_prompts_stripped)
        corrupt_tokens = model.to_tokens(corrupt_prompts_stripped)
        
        # --- SHAPE GUARD ---
        # Full Sequence Patching requires identical shapes [Batch, Seq_Len]
        if clean_tokens.shape != corrupt_tokens.shape:
            print(f"Skipping batch {i}: Shape mismatch {clean_tokens.shape} vs {corrupt_tokens.shape}")
            # Fill with NaNs or 0s to avoid crashing
            continue

        clean_ans_t = torch.tensor(clean_target_ids, device=model.cfg.device)
        corrupt_ans_t = torch.tensor(corrupt_target_ids, device=model.cfg.device)

        # --- 2. CACHE CLEAN STATE (FULL SEQUENCE) ---
        clean_cache = {}
        def cache_full_sequence(resid_pre, hook):
            # Save [Batch, Seq_Len, d_model]
            clean_cache[hook.name] = resid_pre.detach().clone()
            
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(get_act_name("resid_post", l), cache_full_sequence) for l in range(n_layers)]
        )
        
        # --- 3. PATCHING SWEEP (FULL SEQUENCE) ---
        for layer in range(n_layers):
            hook_name = get_act_name("resid_post", layer)
            
            def patch_full_sequence(resid_post, hook):
                # Overwrite EVERYTHING (Prompt + CoT) at this layer
                # This tests: "If we switch to the clean timeline at Layer X, do we recover?"
                resid_post[:, :, :] = clean_cache[hook.name]
                return resid_post
                
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_full_sequence)]
            )
            
            # Calculate Metric on Last Token (Pos -1)
            final_logits = logits[:, -1, :]
            
            clean_score = final_logits.gather(1, clean_ans_t.unsqueeze(1)).squeeze()
            corrupt_score = final_logits.gather(1, corrupt_ans_t.unsqueeze(1)).squeeze()
            
            diff = clean_score - corrupt_score
            logit_diffs[layer, i : i+len(batch)] = diff.cpu()
            
            # Clear cache for this layer to save memory (optional, Python GC usually handles it)
            # del clean_cache[hook_name] 

    return logit_diffs
