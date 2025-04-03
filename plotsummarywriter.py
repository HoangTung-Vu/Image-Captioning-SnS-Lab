import argparse
import math
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os 

def plot_summary(summary_path: str, title : str ):
    output_path: str = f"logs/{title}/plot_train_progress.png"
    # Load the event file
    print(f"Loading summary from: {summary_path}")
    ea = event_accumulator.EventAccumulator(summary_path)
    ea.Reload()

    # Get available scalar tags
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print("No scalar tags found in the summary file.")
        return

    print("Found scalar tags:", tags)

    num_tags = len(tags)
    # Determine grid size (square-ish layout)
    cols = math.ceil(math.sqrt(num_tags))
    rows = math.ceil(num_tags / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    # axes could be a 2D array or a 1D array if rows or cols equals 1, so flatten for easier handling.
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, tag in enumerate(tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        axes[idx].plot(steps, values, label=tag)
        axes[idx].set_title(tag)
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel("Value")
        axes[idx].grid(True)
        axes[idx].legend()

    # Hide any extra subplots if they exist
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.suptitle(f"{title}")
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard summary event file metrics with subplots for each loss.")
    parser.add_argument("summary_path", type=str, help="Path to the TensorBoard summary event file")
    parser.add_argument("title", type=str, help="Model name")
    args = parser.parse_args()

    plot_summary(args.summary_path, args.title)

if __name__ == "__main__":
    main()
