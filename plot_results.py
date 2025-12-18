import torch
import matplotlib.pyplot as plt

def generate_plot():
    print("Generating plot...")
    
    # Check if files exist
    import os
    if not os.path.exists("results_easy.pt"):
        print("Error: results_easy.pt not found.")
        return

    # Load results
    easy_scores = torch.load("results_easy.pt", map_location="cpu") # Shape: [Layers, Samples]
    hard_scores = torch.load("results_hard.pt", map_location="cpu")

    # Calculate Means and Standard Errors
    easy_mean = easy_scores.mean(dim=1)
    hard_mean = hard_scores.mean(dim=1)

    # Standard Error = Std / Sqrt(N)
    easy_sem = easy_scores.std(dim=1) / (easy_scores.shape[1] ** 0.5)
    hard_sem = hard_scores.std(dim=1) / (hard_scores.shape[1] ** 0.5)

    layers = range(len(easy_mean))

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot Easy Line (Blue)
    plt.plot(layers, easy_mean, label="Easy (Memorized)", color="blue", marker="o")
    plt.fill_between(layers, easy_mean - easy_sem, easy_mean + easy_sem, color="blue", alpha=0.2)

    # Plot Hard Line (Red)
    plt.plot(layers, hard_mean, label="Hard (Reasoning)", color="red", marker="o")
    plt.fill_between(layers, hard_mean - hard_sem, hard_mean + hard_sem, color="red", alpha=0.2)

    plt.xlabel("Layer Patched (Residual Stream)")
    plt.ylabel("Logit Difference (Clean - Corrupted)")
    plt.title("Faithfulness 'Laziness Switch':\nWhen does the model decide the answer?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig("laziness_switch_plot.png")
    print("Plot saved to laziness_switch_plot.png")

if __name__ == "__main__":
    generate_plot()
