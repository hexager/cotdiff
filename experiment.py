import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from tqdm import tqdm
import json

@torch.no_grad()
def get_actual_predicted_token(model, text):
    """
    Runs the model on the text to find what token it *actually* predicts next.
    This avoids all 'space vs no space' tokenizer ambiguity.
    """
    tokens = model.to_tokens(text)
    logits = model(tokens)
    # Get the token with the highest probability at the last position
    # (The one the model 'wants' to say)
    predicted_id = logits[0, -1].argmax().item()
    return predicted_id

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
            # We assume the dataset has 'clean_ans' and 'corrupt_ans' correct
            c_txt = b["clean_text"]
            c_ans = b["clean_ans"]
            clean_cut = c_txt.replace(c_ans, "").strip() # Strip usually removes trailing space
            
            corr_txt = b["corrupt_text"]
            corr_ans = b["corrupt_ans"]
            corrupt_cut = corr_txt.replace(corr_ans, "").strip()
            
            clean_prompts_stripped.append(clean_cut)
            corrupt_prompts_stripped.append(corrupt_cut)
            
            # CRITICAL FIX: Don't guess the ID. Ask the model.
            # In a real setup, we want to know if we recovered the *Clean Answer*.
            # So we find the token ID for the Clean Answer *given the Clean Context*.
            # Note: This assumes the Clean Model *correctly* predicts the answer.
            # If the model is wrong on the clean prompt, this logic might be noisy,
            # but for 'Easy' and 'Hard' arithmetic, it's usually correct.
            
            # Alternative: Use the tokenizer on the answer string, but prepend a space
            # if the prompt doesn't have one.
            # Safe bet: Just encode(" " + ans) and encode(ans) and see which one fits
            # the logits better? 
            # Fastest bet: Use the 'clean_ans' string but rely on the fact that
            # we are comparing Logit(Clean_ID) - Logit(Corrupt_ID).
            
            # Let's try the "Prompt + Answer" tokenization check
            # We take the token ID that appears *after* the cut in the full text
            full_c_toks = model.to_tokens(c_txt)[0]
            cut_c_toks = model.to_tokens(clean_cut)[0]
            
            # The answer token is the first new token
            if len(full_c_toks) > len(cut_c_toks):
                clean_id = full_c_toks[len(cut_c_toks)].item()
            else:
                # Fallback
                clean_id = full_c_toks[-1].item()
                
            full_corr_toks = model.to_tokens(corr_txt)[0]
            cut_corr_toks = model.to_tokens(corrupt_cut)[0]
            
            if len(full_corr_toks) > len(cut_corr_toks):
                corr_id = full_corr_toks[len(cut_corr_toks)].item()
            else:
                corr_id = full_corr_toks[-1].item()

            clean_target_ids.append(clean_id)
            corrupt_target_ids.append(corr_id)
            
        clean_tokens = model.to_tokens(clean_prompts_stripped)
        corrupt_tokens = model.to_tokens(corrupt_prompts_stripped)
        
        # --- SHAPE GUARD ---
        if clean_tokens.shape != corrupt_tokens.shape:
            # Try to pad/truncate if off by 1 (common with space issues)
            # But for rigorous science, skipping is safer.
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
                resid_post[:, :, :] = clean_cache[hook.name]
                return resid_post
                
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_full_sequence)]
            )
            
            final_logits = logits[:, -1, :]
            clean_score = final_logits.gather(1, clean_ans_t.unsqueeze(1)).squeeze()
            corrupt_score = final_logits.gather(1, corrupt_ans_t.unsqueeze(1)).squeeze()
            
            diff = clean_score - corrupt_score
            logit_diffs[layer, i : i+len(batch)] = diff.cpu()

    return logit_diffs
