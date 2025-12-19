import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import generate_data
import experiment
import plot_results

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
DATA_FILE = "dataset_natural.json"

def main():
    # 1. Load Model
    print("Loading Model...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", hf_model=hf_model, tokenizer=tokenizer)
    
    # 2. Generate Data
    if not os.path.exists(DATA_FILE):
        print("Generating Natural Pairs...")
        dataset = generate_data.generate_natural_pairs(hf_model, tokenizer, num_samples=50)
        with open(DATA_FILE, "w") as f:
            json.dump(dataset, f, indent=2)
    else:
        print("Loading existing dataset...")
        with open(DATA_FILE, "r") as f:
            dataset = json.load(f)

    # 3. Run Experiment
    print("Running Experiment...")
    easy_data = [d for d in dataset if d["type"] == "easy"]
    hard_data = [d for d in dataset if d["type"] == "hard"]
    
    res_easy = experiment.run_layer_sweep(model, easy_data)
    torch.save(res_easy, "results_easy.pt")
    
    res_hard = experiment.run_layer_sweep(model, hard_data)
    torch.save(res_hard, "results_hard.pt")
    
    # 4. Plot
    print("Plotting...")
    plot_results.generate_plot("results_easy.pt", "results_hard.pt")
    print("Done! Check laziness_plot.png")

if __name__ == "__main__":
    main()
