# utils/evaluate.py

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm
import nltk
# NLTK BLEU Score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# NLTK METEOR Score (ensure NLTK data like 'wordnet' is downloaded)
from nltk.translate.meteor_score import meteor_score
# ROUGE Score (using rouge-score library)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge-score library not found. ROUGE-L will not be calculated.")
    print("Install it using: pip install rouge-score")
    ROUGE_AVAILABLE = False
# BERTScore (handle potential import error)
try:
    from bert_score import score as bert_score_compute
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score library not found. BERTScore will not be calculated.")
    print("Install it using: pip install bert-score")
    BERT_SCORE_AVAILABLE = False

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Callable, Type

from model.base_model import BaseModel
from utils.config import Config
from utils.dataloader import Vocabulary

class Evaluator:
    def __init__(
        self,
        model_name: str,
        data_root: str = Config.data.data_root,
        captions_file: str = Config.data.captions_file, # Evaluation set captions
        checkpoint_path: str = 'checkpoints/model/best_model.pth.tar', # Path to the specific best model
        beam_search: bool = Config.evaluate.beam_search,
        beam_size: int = Config.evaluate.beam_size,
        device: Optional[torch.device] = None,
        batch_size: int = Config.evaluate.batch_size, # For potential future batch eval
        visualization_dir: str = Config.evaluate.visualize_dir
    ):
        self.model_name = model_name
        self.data_root = data_root
        self.captions_file = captions_file
        self.checkpoint_path = checkpoint_path
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.visualization_dir = os.path.join(visualization_dir, self.model_name)
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.device = device or torch.device(Config.device)
        print(f"Evaluator using device: {self.device}")

        self._ensure_nltk_punkt_wordnet() # Ensure necessary NLTK data

        self.model: Optional[BaseModel] = None
        self.vocab: Optional[Vocabulary] = None

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _ensure_nltk_punkt_wordnet(self) -> None:
        """Download NLTK 'punkt' and 'wordnet' if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK 'wordnet'...")
            nltk.download('wordnet', quiet=True)

    def load_model(
        self,
        model_class: Type[BaseModel],
        **model_kwargs
    ) -> Tuple[BaseModel, Vocabulary]:
        """ Loads model and vocabulary from the specified checkpoint path. """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {self.checkpoint_path}")

        print(f"Loading checkpoint for evaluation: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if 'vocab' in checkpoint and isinstance(checkpoint['vocab'], Vocabulary):
            self.vocab = checkpoint['vocab']
            print(f"Vocabulary loaded from checkpoint (Size: {len(self.vocab)})")
        else:
            raise ValueError("Vocabulary object not found or invalid in the checkpoint.")

        if 'vocab_size' not in model_kwargs:
            model_kwargs['vocab_size'] = len(self.vocab)

        self.model = model_class(**model_kwargs).to(self.device)
        try:
            state_dict = checkpoint['model_state_dict']
            if all(key.startswith('module.') for key in state_dict.keys()):
                print("Removing 'module.' prefix from state_dict keys.")
                state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
             print(f"Error loading state_dict: {e}")
             raise e

        self.model.eval()
        print(f"Model '{self.model.__class__.__name__}' loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
        return self.model, self.vocab

    def _prepare_evaluation_data(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Loads captions and returns references grouped by image ID and unique IDs.

        Returns:
            Tuple: (imgToRefs, unique_image_ids)
                   - imgToRefs: {image_id_str: [ref1_str, ref2_str, ...]}
                   - unique_image_ids: List of unique image ID strings.
        """
        try:
            df = pd.read_csv(self.captions_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Evaluation captions file not found: {self.captions_file}")
        except Exception as e:
            raise IOError(f"Error reading captions file {self.captions_file}: {e}")

        if 'image' not in df.columns or 'caption' not in df.columns:
             raise ValueError("Captions file must contain 'image' and 'caption' columns.")

        imgToRefs = defaultdict(list)
        print(f"Processing reference captions from {self.captions_file}...")
        df['image'] = df['image'].astype(str)
        for _, row in df.iterrows():
            img_id_str = row['image']
            caption_str = str(row['caption'])
            imgToRefs[img_id_str].append(caption_str)

        unique_image_ids = df['image'].unique().tolist()
        print(f"Found {len(unique_image_ids)} unique images with {len(df)} total reference captions.")
        return imgToRefs, unique_image_ids

    @torch.no_grad()
    def _generate_hypotheses(self, unique_image_ids: List[str]) -> Dict[str, str]:
        """ Generates one hypothesis string for each image ID. """
        if self.model is None or self.vocab is None:
            raise ValueError("Model and vocabulary must be loaded first.")

        self.model.eval()
        imgToHyp = {} # {image_id_str: hyp_str}

        print(f"Generating hypotheses for {len(unique_image_ids)} unique images...")
        for image_id in tqdm(unique_image_ids, desc="Generating Hyps"):
            image_path = os.path.join(self.data_root, image_id)
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                pred_tokens = self._generate_caption_tokens(image_tensor)
                pred_caption = self._tokens_to_caption_string(pred_tokens)
                imgToHyp[image_id] = pred_caption # Store single string

            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}. Skipping image {image_id}.")
                imgToHyp[image_id] = "[Skipped Image]"
                continue
            except Exception as e:
                print(f"Warning: Error processing image {image_id}: {e}. Skipping.")
                imgToHyp[image_id] = "[Error Processing]"
                continue
        return imgToHyp

    def _generate_caption_tokens(self, image_tensor: torch.Tensor) -> List[str]:
        """ Generates caption tokens using the configured method (beam/greedy). """
        if self.beam_search:
            if hasattr(self.model, 'caption_image_beam_search'):
                # Ensure beam search returns list of strings (tokens)
                caption_output = self.model.caption_image_beam_search(image_tensor, self.vocab, beam_size=self.beam_size)
                return caption_output if isinstance(caption_output, list) else [] # Handle potential non-list return
            else:
                print("Warning: Beam search requested, but model lacks 'caption_image_beam_search'. Falling back to greedy.")
        if hasattr(self.model, 'caption_image_greedy'):
             # Ensure greedy returns list of strings (tokens)
             caption_output = self.model.caption_image_greedy(image_tensor, self.vocab)
             return caption_output if isinstance(caption_output, list) else [] # Handle potential non-list return
        else:
            raise NotImplementedError("Model must implement 'caption_image_greedy' or 'caption_image_beam_search'.")


    def _tokens_to_caption_string(self, tokens: List[str]) -> str:
        """ Converts a list of tokens into a clean caption string. """
        return ' '.join([token for token in tokens if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])

    def calculate_metrics(self, imgToRefs: Dict[str, List[str]], imgToHyp: Dict[str, str]) -> Dict[str, float]:
        """
        Calculates BLEU, METEOR, ROUGE-L, and BERTScore using NLTK and other libraries.
        Omits CIDEr.

        Args:
            imgToRefs: Dictionary mapping image IDs to lists of reference strings.
            imgToHyp: Dictionary mapping image IDs to a hypothesis string.

        Returns:
            Dictionary containing all calculated metric scores.
        """
        print("\nCalculating evaluation metrics (using NLTK, rouge-score, bert-score)...")
        metrics_scores = {}

        # Filter out images where hypothesis generation failed
        valid_ids = [img_id for img_id, hyp_str in imgToHyp.items()
                     if not hyp_str.startswith("[Skipped") and not hyp_str.startswith("[Error")]
        if len(valid_ids) != len(imgToHyp):
            print(f"Warning: Evaluating on {len(valid_ids)} images where hypothesis generation succeeded.")

        if not valid_ids:
             print("Error: No valid hypotheses found for metric calculation.")
             return {}

        # Prepare lists for corpus-level calculation or iteration
        references_corpus = [] # List of lists of tokenized references for each image (for BLEU)
        references_strings_per_image = [] # List of lists of reference strings (for METEOR, ROUGE)
        hypotheses_corpus = [] # List of tokenized hypotheses (for BLEU)
        hypotheses_strings = [] # List of hypothesis strings (for ROUGE, BERTScore)

        print("Tokenizing references and hypotheses...")
        for img_id in tqdm(valid_ids, desc="Tokenizing"):
            if img_id not in imgToRefs: continue # Skip if refs missing for a valid hyp

            refs_str_list = imgToRefs[img_id]
            hyp_str = imgToHyp[img_id]

            # Tokenize for BLEU and METEOR
            refs_tokenized = [nltk.word_tokenize(ref.lower()) for ref in refs_str_list]
            hyp_tokenized = nltk.word_tokenize(hyp_str.lower())

            references_corpus.append(refs_tokenized)
            hypotheses_corpus.append(hyp_tokenized)

            # Keep strings for ROUGE and BERTScore
            references_strings_per_image.append(refs_str_list)
            hypotheses_strings.append(hyp_str)

        if not references_corpus or not hypotheses_corpus:
            print("Error: No tokenized data available after filtering.")
            return {}

        # --- BLEU (using NLTK) ---
        print("Calculating BLEU...")
        try:
            bleu_weights = {
                "BLEU-1": (1.0, 0.0, 0.0, 0.0), "BLEU-2": (0.5, 0.5, 0.0, 0.0),
                "BLEU-3": (0.333, 0.333, 0.333, 0.0), "BLEU-4": (0.25, 0.25, 0.25, 0.25)
            }
            smoother = SmoothingFunction().method1
            for name, weights in bleu_weights.items():
                score = corpus_bleu(references_corpus, hypotheses_corpus, weights=weights, smoothing_function=smoother)
                metrics_scores[name] = score * 100
        except Exception as e:
            print(f"Warning: Could not calculate BLEU scores: {e}")
            for name in bleu_weights: metrics_scores[name] = 0.0

        # --- METEOR (using NLTK) ---
        print("Calculating METEOR...")
        try:
            meteor_scores = [meteor_score(refs_tok, hyp_tok)
                             for refs_tok, hyp_tok in zip(references_corpus, hypotheses_corpus)]
            metrics_scores['METEOR'] = np.mean(meteor_scores) * 100
        except Exception as e:
            print(f"Warning: Could not calculate METEOR score: {e}")
            metrics_scores['METEOR'] = 0.0

        # --- ROUGE-L (using rouge-score) ---
        if ROUGE_AVAILABLE:
            print("Calculating ROUGE-L...")
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_l_f_scores = []
                # Calculate max ROUGE-L F1 against all references for each hypothesis
                for refs_str_list, hyp_str in zip(references_strings_per_image, hypotheses_strings):
                    max_rouge_f = 0.0
                    if not hyp_str: # Handle empty hypothesis
                        rouge_l_f_scores.append(0.0)
                        continue
                    for ref_str in refs_str_list:
                         if not ref_str: continue # Handle empty reference
                         score = scorer.score(ref_str, hyp_str)
                         max_rouge_f = max(max_rouge_f, score['rougeL'].fmeasure)
                    rouge_l_f_scores.append(max_rouge_f)
                metrics_scores['ROUGE-L'] = np.mean(rouge_l_f_scores) * 100
            except Exception as e:
                print(f"Warning: Could not calculate ROUGE-L score: {e}")
                metrics_scores['ROUGE-L'] = 0.0
        else:
             metrics_scores['ROUGE-L'] = 0.0 # Not available


        # --- BERTScore ---
        if BERT_SCORE_AVAILABLE:
            print("Calculating BERTScore (this may take a while)...")
            try:
                bert_device = 0 if self.device.type == 'cuda' else -1
                # BERTScore needs references as list of lists of strings
                bert_references = [refs for refs in references_strings_per_image]
                bert_candidates = hypotheses_strings
                P, R, F1 = bert_score_compute(bert_candidates, bert_references, lang="en", verbose=False, device=bert_device, batch_size=max(16, self.batch_size)) # Use reasonable batch size
                metrics_scores['BERTScore-F1'] = F1.mean().item() * 100
            except Exception as e:
                 print(f"Warning: Could not calculate BERTScore: {e}")
                 metrics_scores['BERTScore-F1'] = 0.0
        else:
            metrics_scores['BERTScore-F1'] = 0.0 # Not available

        # --- CIDEr (Omitted) ---
        print("Note: CIDEr calculation is omitted as it typically relies on pycocoevalcap corpus statistics.")
        metrics_scores['CIDEr'] = 0.0 # Placeholder


        print("Metric calculation finished.")
        return metrics_scores

    @torch.no_grad()
    def visualize_examples(self, num_examples: int = 10) -> None:
        """ Visualize random examples with predictions and references. """
        if self.model is None or self.vocab is None:
            raise ValueError("Model and vocabulary must be loaded.")
        self.model.eval()

        imgToRefs, unique_image_ids = self._prepare_evaluation_data()
        num_to_show = min(num_examples, len(unique_image_ids))
        if num_to_show == 0: return
        selected_image_ids = np.random.choice(unique_image_ids, num_to_show, replace=False)

        print(f"\nGenerating visualizations for {num_to_show} examples...")
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()

        for i, image_id in enumerate(tqdm(selected_image_ids, desc="Visualizing")):
            image_path = os.path.join(self.data_root, image_id)
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                pred_tokens = self._generate_caption_tokens(image_tensor)
                pred_caption = self._tokens_to_caption_string(pred_tokens)
                references = imgToRefs.get(image_id, ["N/A"])

                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                title = f"Model: {self.model_name.upper()} | Image: {image_id}"
                ref_text = "References:\n" + "\n".join([f"- {ref}" for ref in references])
                pred_method = f"Beam(k={self.beam_size})" if self.beam_search else "Greedy"
                pred_text = f"\nPrediction ({pred_method}):\n- {pred_caption}"
                full_caption_text = f"{ref_text}{pred_text}"
                plt.figtext(0.5, 0.01, full_caption_text, wrap=True, ha='center', fontsize=9)
                plt.title(title, fontsize=11)
                plt.subplots_adjust(bottom=0.25)

                output_path = os.path.join(self.visualization_dir, f"example_{i+1}_{image_id}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
            except Exception as e:
                print(f"Warning: Error visualizing image {image_id}: {e}")
                plt.close()
                continue
        print(f"\nVisualizations saved to: {self.visualization_dir}")


    def run_full_evaluation(
        self,
        model_class: Type[BaseModel],
        visualize: bool = Config.evaluate.visualize,
        num_examples: int = Config.evaluate.num_examples,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Run the full evaluation: load model, generate hypotheses, calculate all metrics, visualize.
        """
        start_time = time.time()
        all_metrics = {}
        try:
            self.load_model(model_class, **model_kwargs)
            imgToRefs, unique_image_ids = self._prepare_evaluation_data()
            imgToHyp = self._generate_hypotheses(unique_image_ids)
            all_metrics = self.calculate_metrics(imgToRefs, imgToHyp)

            print("\n--- Evaluation Metrics ---")
            for metric, score in all_metrics.items():
                 print(f"  {metric:<12}: {score:.2f}") # Use consistent formatting for now
            print("--------------------------")

            results_path = os.path.join(self.visualization_dir, "metrics.json")
            try:
                with open(results_path, 'w') as f:
                    json.dump(all_metrics, f, indent=4)
                print(f"Metrics saved to: {results_path}")
            except Exception as e:
                print(f"Error saving metrics to {results_path}: {e}")

            if visualize:
                self.visualize_examples(num_examples)

            end_time = time.time()
            print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds.")

        except FileNotFoundError as e:
            print(f"Error during evaluation setup: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")
            import traceback
            traceback.print_exc()

        return all_metrics