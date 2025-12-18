import os
import json
import torch
from generate_data import generate_raw_pairs, precompute_cot, PROMPT_TEMPLATE
from transformer_lens import HookedTransformer

def main():
    DATA_PATH = "dataset_final.json"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    WEIGHTS_PATH = "Qwen/Qwen2.5-Math-1.5B"
    
    if not os.path.exists(DATA_PATH):
        print("--- Step 1: Generating Dataset ---")
        temp_model = HookedTransformer.from_pretrained(
            model_name=MODEL_NAME, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            center_unembed=True
            center_writing_weights=True,
            fold_ln=True,
            hf_model_name=WEIGHTS_PATH
        )
        
        raw_data = generate_raw_pairs(num_samples=50)
        final_dataset = precompute_cot(temp_model, raw_data["easy"] + raw_data["hard"], PROMPT_TEMPLATE)
        
        with open(DATA_PATH, "w") as f:
            json.dump(final_dataset, f)
        
        del temp_model
        torch.cuda.empty_cache()
        print(f"Dataset saved to {DATA_PATH}\n")
    else:
        print(f"--- Step 1: {DATA_PATH} already exists. Skipping generation. ---\n")

    print("--- Step 2: Running Activation Patching Experiment ---")
    import experiment 
    
    print("\n--- Step 3: Generating Visualizations ---")
    import plot_results
    
    print("\n[SUCCESS] Experiment Pipeline Complete.")
    print("Check 'laziness_switch_plot.png' for the final results.")

if __name__ == "__main__":
    main()