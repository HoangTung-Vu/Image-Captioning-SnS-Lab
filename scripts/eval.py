import os
import sys
import torch
import argparse
import json
from typing import Dict, Any, Optional, Type, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Add parent directory to path to allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluate import Evaluator
from utils.config import Config
from model.base_model import BaseModel
from model.cnntornn import CNNtoRNN
from model.transformer import VisionEncoderDecoder

def main():
    """
    Main function to evaluate image captioning models
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate image captioning models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--models", type=str, nargs='+', default=["transformer", "cnntornn"], 
                        help="Models to evaluate: transformer, cnntornn, or both")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search for caption generation")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search")
    parser.add_argument("--visualize", action="store_true", help="Visualize examples")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to visualize")
    parser.add_argument("--test_image", type=str, default=None, help="Path to a single test image")
    parser.add_argument("--compare", action="store_true", help="Compare results from different models")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Store results for comparison
    all_results = {}
    
    # Evaluate each specified model
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name.upper()} model")
        print(f"{'='*50}\n")
        
        # Load model-specific config if available
        model_config_path = args.config
        if model_config_path is None:
            # Try to find model-specific config
            default_config_path = f"configs/{model_name.lower()}_config.json"
            if os.path.exists(default_config_path):
                model_config_path = default_config_path
                print(f"Using model-specific config: {model_config_path}")
        
        # Load configuration
        if model_config_path and os.path.exists(model_config_path):
            Config.load_from_json(model_config_path)
            print(f"Loaded configuration from {model_config_path}")
        else:
            print("Using default configuration")
        
        # Get model class and kwargs
        model_class, model_kwargs = get_model_config(model_name)
        
        # Set checkpoint path
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            # Try to use the best model checkpoint
            best_model_path = os.path.join(Config.train.checkpoint_dir, model_name, "best_model.pth.tar")
            if os.path.exists(best_model_path):
                checkpoint_path = best_model_path
                print(f"Using best model checkpoint: {checkpoint_path}")
            else:
                # Fall back to latest checkpoint
                latest_path = os.path.join(Config.train.checkpoint_dir, model_name, "latest_checkpoint.pth.tar")
                if os.path.exists(latest_path):
                    checkpoint_path = latest_path
                    print(f"Using latest checkpoint: {checkpoint_path}")
                else:
                    print(f"No checkpoint found for {model_name} model")
                    continue
        
        # Create visualization directory
        visualization_dir = os.path.join(Config.evaluate.visualize_dir, model_name)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Initialize evaluator
        evaluator = Evaluator(
            data_root=Config.data.data_root,
            captions_file=Config.data.captions_file,
            checkpoint_path=checkpoint_path,
            beam_search=args.beam_search or Config.evaluate.beam_search,
            device=device,
            batch_size=1,  # Use batch size 1 for evaluation
            visualization_dir=visualization_dir
        )
        
        # If a single test image is provided, evaluate it
        if args.test_image:
            result = evaluate_single_image(
                args.test_image, 
                evaluator, 
                model_class, 
                model_kwargs, 
                args.beam_search, 
                args.beam_size,
                model_name
            )
            all_results[model_name] = {"single_image": result}
            continue
        
        # Run full evaluation
        print(f"Evaluating {model_name} model...")
        bleu_scores = evaluator.run_evaluation(
            model_class=model_class,
            visualize=args.visualize or Config.evaluate.visualize,
            num_examples=args.num_examples or Config.evaluate.num_examples,
            **model_kwargs
        )
        
        # Print results
        print(f"\nEvaluation Results for {model_name} model:")
        for metric, score in bleu_scores.items():
            print(f"{metric}: {score:.2f}")
        
        # Save results to file
        results_path = os.path.join(visualization_dir, "bleu_scores.json")
        with open(results_path, 'w') as f:
            json.dump(bleu_scores, f, indent=4)
        
        print(f"Results saved to {results_path}")
        
        # Store results for comparison
        all_results[model_name] = bleu_scores
    
    # Compare results if requested and multiple models were evaluated
    if args.compare and len(all_results) > 1:
        compare_results(all_results)
    
    print("\nEvaluation complete!")

def get_model_config(model_name: str) -> Tuple[Type[BaseModel], Dict[str, Any]]:
    """
    Get the model class and configuration based on the model name
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (model_class, model_kwargs)
    """
    # Model-specific parameters
    if model_name.lower() == "transformer":
        return VisionEncoderDecoder, {
            'image_size': 224,
            'channels_in': 3,
            'patch_size': 16,
            'hidden_size': 512,
            'num_layers': (6, 6),  # (encoder_layers, decoder_layers)
            'num_heads': 8,
        }
    elif model_name.lower() == "cnntornn":
        return CNNtoRNN, {
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 2,
            'dropout_rate': 0.5,
            'trainCNN': True,  # For evaluation, we want the full model
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_single_image(
    image_path: str, 
    evaluator: Evaluator, 
    model_class: Type[BaseModel], 
    model_kwargs: Dict[str, Any],
    use_beam_search: bool = True,
    beam_size: int = 3,
    model_name: str = "model"
) -> Dict[str, str]:
    """
    Evaluate a single image
    
    Args:
        image_path: Path to the image
        evaluator: Evaluator instance
        model_class: Model class
        model_kwargs: Model keyword arguments
        use_beam_search: Whether to use beam search
        beam_size: Beam size for beam search
        model_name: Name of the model
        
    Returns:
        Dictionary with caption results
    """
    # Load model
    model, vocab = evaluator.load_model(model_class, **model_kwargs)
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = evaluator.transform(image).unsqueeze(0).to(evaluator.device)
        
        results = {}
        
        # Generate caption with beam search if available
        if use_beam_search and hasattr(model, 'caption_image_beam_search'):
            tokens = model.caption_image_beam_search(image_tensor, vocab, beam_size=beam_size)
            beam_caption = " ".join([token for token in tokens if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            results["beam_search"] = beam_caption
            print(f"\n{model_name} - Beam Search (beam_size={beam_size}):")
            print(beam_caption)
        
        # Generate caption with greedy search if available
        if hasattr(model, 'caption_image_greedy'):
            tokens = model.caption_image_greedy(image_tensor, vocab)
            greedy_caption = " ".join([token for token in tokens if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            results["greedy"] = greedy_caption
            print(f"\n{model_name} - Greedy Search:")
            print(greedy_caption)
        
        # Display results
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        
        title = f"{model_name.upper()} Generated Captions:\n"
        if "beam_search" in results:
            title += f"Beam Search: {results['beam_search']}\n"
        if "greedy" in results:
            title += f"Greedy: {results['greedy']}"
            
        plt.title(title, fontsize=12)
        plt.axis('off')
        
        # Save figure
        output_path = os.path.join(evaluator.visualization_dir, f"single_image_result.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
        plt.close()
        
        print(f"\nVisualization saved to {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating image: {e}")
        return {"error": str(e)}

def compare_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Compare results from different models
    
    Args:
        results: Dictionary mapping model names to their results
    """
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    # Create comparison table
    metrics = list(next(iter(results.values())).keys())
    
    # Print header
    header = "| Metric | " + " | ".join(results.keys()) + " |"
    separator = "|" + "-"*(len(header) - 2) + "|"
    
    print("\n" + header)
    print(separator)
    
    # Print rows
    for metric in metrics:
        row = f"| {metric} | "
        for model_name in results.keys():
            if metric in results[model_name]:
                value = results[model_name][metric]
                if isinstance(value, float):
                    row += f"{value:.2f} | "
                else:
                    row += f"{value} | "
            else:
                row += "N/A | "
        print(row)
    
    # Create comparison chart for BLEU scores
    if "BLEU-1" in metrics:
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        model_names = list(results.keys())
        bleu_metrics = [m for m in metrics if m.startswith("BLEU")]
        
        # Set width of bars
        bar_width = 0.2
        index = range(len(bleu_metrics))
        
        # Plot bars for each model
        for i, model_name in enumerate(model_names):
            values = [results[model_name][metric] for metric in bleu_metrics]
            plt.bar([x + i * bar_width for x in index], values, bar_width, label=model_name)
        
        # Add labels and title
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('BLEU Score Comparison')
        plt.xticks([x + bar_width * (len(model_names) - 1) / 2 for x in index], bleu_metrics)
        plt.legend()
        
        # Save figure
        os.makedirs("comparisons", exist_ok=True)
        plt.savefig("comparisons/bleu_comparison.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        print("\nComparison chart saved to comparisons/bleu_comparison.png")

if __name__ == "__main__":
    main()

