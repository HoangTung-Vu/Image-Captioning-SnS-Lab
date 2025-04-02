# eval.py

import os
import sys
import torch
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Type, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the updated Evaluator class (which no longer uses pycocoevalcap)
from utils.evaluate import Evaluator
from utils.config import Config
from model.base_model import BaseModel
from model.cnntornn import CNNtoRNN
from model.transformer import VisionEncoderDecoder

# Keep get_model_config consistent with train.py defaults or load from checkpoint config if possible
def get_model_config(model_name: str) -> Tuple[Type[BaseModel], Dict[str, Any]]:
    """ Gets model class and default parameters (used as fallback). """
    model_name_lower = model_name.lower()
    if model_name_lower == "transformer":
        return VisionEncoderDecoder, {
            'image_size': 224, 'channels_in': 3, 'patch_size': 16,
            'hidden_size': 256, 'num_layers': (1, 1), 'num_heads': 4,
        }
    elif model_name_lower == "cnntornn":
        return CNNtoRNN, {
            'embed_size': 256, 'hidden_size': 512, 'num_layers': 1,
            'dropout_rate': 0.5, 'trainCNN': False,
        }
    else:
        raise ValueError(f"Unknown model name: {model_name_lower}. Choose 'transformer' or 'cnntornn'.")

def compare_results(results_dict: Dict[str, Dict[str, float]], comparison_dir: str = "comparisons") -> None:
    """ Compare evaluation results from different models using a table and chart. """
    if not results_dict or len(results_dict) < 1:
        print("No results to compare.")
        return

    print("\n" + "="*60)
    print(" Model Evaluation Results Comparison ")
    print("="*60)

    model_names = list(results_dict.keys())
    # Get all metrics reported by the first model (assume others are similar)
    # Filter out CIDEr if it's present as 0.0 placeholder
    metrics = [m for m in next(iter(results_dict.values())).keys() if m != 'CIDEr']

    # --- Create Comparison Table ---
    try:
        data = {metric: [results_dict[model].get(metric, 'N/A') for model in model_names]
                for metric in metrics}
        df = pd.DataFrame(data, index=model_names).T # Metrics as rows

        # Format floats nicely
        df_display = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) else x)

        print("\nComparison Table:")
        try: print(df_display.to_markdown())
        except ImportError: print(df_display.to_string())

    except Exception as e:
        print(f"Error creating comparison table: {e}")

    # --- Create Comparison Bar Chart (Selected Metrics) ---
    # Keep CIDEr out as it was omitted
    metrics_to_plot = ['BLEU-4', 'METEOR', 'ROUGE-L', 'BERTScore-F1']
    # Filter metrics_to_plot based on what's actually available in the results
    available_metrics_to_plot = [m for m in metrics_to_plot if m in metrics]

    if not available_metrics_to_plot or len(model_names) == 0:
         print("\nNo plottable metric data found.")
         return

    plot_data = {m: [] for m in available_metrics_to_plot}
    valid_metrics_found = False

    for model in model_names:
        for metric in available_metrics_to_plot:
            score = results_dict[model].get(metric, 0.0) # Default to 0 if metric missing
            if score != 0.0: valid_metrics_found = True
            plot_data[metric].append(score)

    if not valid_metrics_found:
        print("\nNo valid scores found for plotting.")
        return

    try:
        num_models = len(model_names)
        num_metrics_plot = len(available_metrics_to_plot)
        bar_width = 0.8 / num_models
        index = np.arange(num_metrics_plot)

        plt.figure(figsize=(max(8, num_metrics_plot * num_models * 0.8), 6))

        for i, model_name in enumerate(model_names):
            scores = [plot_data[metric][i] for metric in available_metrics_to_plot]
            plt.bar(index + i * bar_width, scores, bar_width, label=model_name)

        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Model Evaluation Metric Comparison')
        plt.xticks(index + bar_width * (num_models - 1) / 2, available_metrics_to_plot)
        plt.legend(loc='best')
        plt.ylim(bottom=0) # Start y-axis at 0
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        os.makedirs(comparison_dir, exist_ok=True)
        chart_path = os.path.join(comparison_dir, "metrics_comparison.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\nComparison chart saved to: {chart_path}")

    except Exception as e:
        print(f"Error creating comparison chart: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Image Captioning Models (No pycocoevalcap)")
    parser.add_argument(
        "--models", type=str, nargs='+', default=["cnntornn", "transformer"],
        choices=["transformer", "cnntornn"], help="Specify which trained model(s) to evaluate (best checkpoint)."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON config file for global settings."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=Config.train.checkpoint_dir,
        help="Base directory containing model checkpoint folders (e.g., 'checkpoints/')."
    )
    parser.add_argument(
        "--captions_file", type=str, default=Config.data.captions_file,
        help="Path to the captions file for the evaluation set."
    )
    parser.add_argument(
        "--data_root", type=str, default=Config.data.data_root,
        help="Path to the directory containing evaluation images."
    )
    # Args to override specific evaluation behaviors from Config
    parser.add_argument("--beam", action="store_true", help="Force use of beam search.")
    parser.add_argument("--no-beam", action="store_true", help="Force use of greedy search.")
    parser.add_argument("--beam_size", type=int, help="Override beam size.")
    parser.add_argument("--visualize", action="store_true", help="Force visualization generation.")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization generation.")
    parser.add_argument("--num_examples", type=int, help="Override number of examples to visualize.")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Generate comparison table/chart if multiple models are evaluated.")
    parser.add_argument("--no-compare", action="store_false", dest="compare",
                        help="Disable comparison generation.")

    args = parser.parse_args()

    # --- Configuration Loading & Overrides ---
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        Config.load_from_json(args.config)
    else:
        print("Using default configuration (or previously loaded).")

    Config.train.checkpoint_dir = args.checkpoint_dir
    Config.data.captions_file = args.captions_file
    Config.data.data_root = args.data_root
    if args.beam: Config.evaluate.beam_search = True
    if args.no_beam: Config.evaluate.beam_search = False
    if args.beam_size is not None: Config.evaluate.beam_size = args.beam_size
    if args.visualize: Config.evaluate.visualize = True
    if args.no_visualize: Config.evaluate.visualize = False
    if args.num_examples is not None: Config.evaluate.num_examples = args.num_examples

    # --- Evaluation Loop ---
    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating Best Model: {model_name.upper()}")
        print(f"{'='*50}\n")

        model_ckpt_dir = os.path.join(Config.train.checkpoint_dir, model_name)
        best_checkpoint_path = os.path.join(model_ckpt_dir, "best_model.pth.tar")

        if not os.path.exists(best_checkpoint_path):
            print(f"Error: 'best_model.pth.tar' not found in {model_ckpt_dir}.")
            print(f"Skipping evaluation for {model_name}.")
            continue

        print(f"Found best checkpoint: {best_checkpoint_path}")

        try:
            model_class, model_kwargs = get_model_config(model_name)
        except ValueError as e:
            print(f"Error getting model config: {e}")
            continue

        evaluator = Evaluator(
            model_name=model_name,
            checkpoint_path=best_checkpoint_path,
            data_root=Config.data.data_root,
            captions_file=Config.data.captions_file,
            # Using beam/visualize settings from Config potentially overridden by args
            beam_search=Config.evaluate.beam_search,
            beam_size=Config.evaluate.beam_size,
            visualization_dir=Config.evaluate.visualize_dir
        )

        metrics = evaluator.run_full_evaluation(
            model_class=model_class,
            visualize=Config.evaluate.visualize,
            num_examples=Config.evaluate.num_examples,
            **model_kwargs
        )

        if metrics:
             all_results[model_name] = metrics
        else:
             print(f"\nEvaluation did not produce results for {model_name.upper()}.")


    # --- Compare Results ---
    if args.compare and len(all_results) > 0:
        compare_results(all_results)
    elif args.compare:
         print("\nComparison requested, but no successful evaluation results to compare.")

    print("\n" + "="*50)
    print("Evaluation Script Finished.")
    print("="*50)

if __name__ == "__main__":
    main()