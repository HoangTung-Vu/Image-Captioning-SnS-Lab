import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnntornn import CNNtoRNN
from utils.evaluate import Evaluator
from utils.config import Config

def main():
    """
    Main function to evaluate the image captioning model
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate an image captioning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        Config.load_from_json(args.config)
    
    # Override config with command line arguments
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join("checkpoints", "best_model.pth.tar")
    
    beam_search = Config.evaluate.beam_search
    beam_size = Config.evaluate.beam_size
    visualize = Config.evaluate.visualize
    
    # Create visualization directory if it doesn't exist
    visualize_dir = os.path.join("logs", "visualizations")
    if visualize:
        os.makedirs(visualize_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator(
        data_root=Config.data.data_root,
        captions_file=Config.data.captions_file,
        checkpoint_path=checkpoint_path,
        beam_search=beam_search,
        device=torch.device(Config.device),
        batch_size=1,  # Use batch size 1 for evaluation
        visualization_dir=visualize_dir
    )
    
    # Run evaluation
    print(f"Evaluating model from checkpoint: {checkpoint_path}")
    print(f"Using beam search: {beam_search}, beam size: {beam_size}")
    
    bleu_scores = evaluator.run_evaluation(
        model_class=CNNtoRNN,
        embed_size=Config.model.embed_size,
        hidden_size=Config.model.hidden_size,
        num_layers=Config.model.num_layers,
        vocab_size=None,  # Will be determined from checkpoint
        dropout_rate=Config.model.dropout,
        visualize=visualize,
        num_examples=Config.evaluate.num_examples,
    )
    
    # Plot BLEU scores
    if visualize:
        plt.figure(figsize=(10, 6))
        metrics = list(bleu_scores.keys())
        scores = list(bleu_scores.values())
        plt.bar(metrics, scores, color='skyblue')
        plt.title('BLEU Scores')
        plt.ylabel('Score')
        plt.ylim(0, 100)
        for i, score in enumerate(scores):
            plt.text(i, score + 1, f"{score:.2f}", ha='center')
        plt.savefig(os.path.join(visualize_dir, 'bleu_scores.png'))
        plt.close()
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()

