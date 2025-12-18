import os
import json
import torch
from generate_data import generate_raw_pairs, precompute_cot, PROMPT_TEMPLATE
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import experiment
import plot_results 

# SETUP
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
DATA_PATH = "dataset_final.json"

def load_model():
    print(f"Loading {MODEL_ID} via HF...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda:0", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print("Wrapping in TransformerLens...")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", # This alias string matters less when hf_model is passed
        hf_model=hf_model,
        tokenizer=tokenizer,
        device="cuda:0",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False
        fold_value_biases=False
    )
    return model

def main():
    # 1. Load Model ONCE
    model = load_model()

    # 2. Generate Data (if missing)
    if not os.path.exists(DATA_PATH):
        print("--- Step 1: Generating Dataset ---")
        raw_data = generate_raw_pairs(num_samples=50)
        
        # Flatten
        combined_raw = []
        for item in raw_data["easy"]: combined_raw.append(item)
        for item in raw_data["hard"]: combined_raw.append(item)
            
        final_dataset = precompute_cot(model, combined_raw, PROMPT_TEMPLATE)
        
        with open(DATA_PATH, "w") as f:
            json.dump(final_dataset, f, indent=2)
        print(f"Dataset saved to {DATA_PATH}\n")
    else:
        print(f"--- Step 1: {DATA_PATH} exists. Skipping generation. ---\n")

    # 3. Run Experiment
    print("--- Step 2: Running Activation Patching Experiment ---")
    experiment.run_pipeline(model)
    
    # 4. Plot
    print("\n--- Step 3: Generating Visualizations ---")
    plot_results.generate_plot()
    print("\n[SUCCESS] Experiment Pipeline Complete.")
    print("Check 'laziness_switch_plot.png' for the final results.")

if __name__ == "__main__":
    main()
