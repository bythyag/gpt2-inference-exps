import torch
import datasets
import time
from tqdm.notebook import tqdm # Use notebook version if running in Colab/Jupyter
# from tqdm import tqdm # Use standard tqdm if running as plain script
import logging
import math
from typing import Dict

logger = logging.getLogger(__name__)

def calculate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    stride: int,
    device: torch.device
) -> float:
    """Calculates perplexity for a given model on a dataset."""
    logger.info(f"Calculating perplexity for model on '{dataset_name}/{dataset_config}' [{split}] split...")
    try:
        logger.debug("Loading dataset...")
        data = datasets.load_dataset(dataset_name, dataset_config, split=split)
        logger.debug("Dataset loaded.")

        logger.debug("Tokenizing dataset...")
        # Consider handling potential memory issues for very large datasets
        all_text = "\n\n".join(filter(None, data['text'])) # Filter out potential None/empty strings
        encodings = tokenizer(all_text, return_tensors='pt', truncation=False) # Don't truncate here
        logger.debug("Tokenization complete.")

        max_length = model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        logger.info(f"Processing sequence length {seq_len} with stride {stride} and max_length {max_length}...")

        # Use standard tqdm here if not in notebook
        # for begin_loc in tqdm(range(0, seq_len, stride), desc="Perplexity Calculation"):
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Perplexity Calculation", leave=False):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 # Ignore context tokens for loss calculation

            if input_ids.size(1) <= 1: # Need at least one token to predict
                 logger.debug(f"Skipping short sequence at begin_loc {begin_loc}")
                 prev_end_loc = end_loc # Still advance prev_end_loc
                 continue

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            if not torch.isnan(neg_log_likelihood) and not torch.isinf(neg_log_likelihood):
                 # Multiply by number of labels considered to get total NLL for the window
                 num_labels = (target_ids != -100).sum().item()
                 if num_labels > 0:
                     nlls.append(neg_log_likelihood.item() * num_labels) # Store total NLL
                 else:
                      logger.warning(f"No labels found for window starting at {begin_loc}")

            else:
                 logger.warning(f"NaN or Inf NLL detected at begin_loc {begin_loc}. Skipping.")


            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if not nlls:
            logger.error("No valid negative log likelihoods were calculated.")
            return float('inf')

        # Calculate perplexity correctly from total NLL over all tokens
        total_nll = sum(nlls)
        total_tokens_evaluated = encodings.input_ids.size(1) # Total tokens in the test set
        # A more precise count would be sum of num_labels across loops,
        # but using total tokens is common practice for stride-based PPL.
        # Let's use total tokens for simplicity, consistent with HF examples.
        mean_nll = total_nll / total_tokens_evaluated
        perplexity = math.exp(mean_nll)


        logger.info(f"Perplexity calculation complete. PPL: {perplexity:.4f}")
        return perplexity

    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}", exc_info=True)
        return float('inf')


def measure_inference_speed(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    generation_config: Dict,
    n_runs: int,
    device: torch.device
) -> float:
    """Measures the average inference speed for text generation."""
    logger.info(f"Measuring inference speed (average over {n_runs} runs)...")
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    total_time = 0.0

    # Ensure pad_token_id is set in the generation config
    if "pad_token_id" not in generation_config or generation_config["pad_token_id"] is None:
         generation_config["pad_token_id"] = tokenizer.eos_token_id
         logger.debug(f"Set pad_token_id in generation_config to {tokenizer.eos_token_id}")


    logger.debug("Performing warm-up run...")
    try:
        with torch.no_grad():
             _ = model.generate(
                 inputs.input_ids,
                 attention_mask=inputs.attention_mask,
                 **generation_config
             )
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
    except Exception as e:
        logger.error(f"Warm-up run failed: {e}", exc_info=True)
        return float('inf') # Cannot proceed if warm-up fails


    logger.debug(f"Starting {n_runs} timed runs...")
    start_all_runs = time.time()
    # Use standard tqdm here if not in notebook
    # for _ in tqdm(range(n_runs), desc="Inference Speed Test"):
    for _ in tqdm(range(n_runs), desc="Inference Speed Test", leave=False):
        try:
            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
            if device == torch.device("cuda"):
                torch.cuda.synchronize() # Ensure GPU work is done
            end_time = time.time()
            total_time += (end_time - start_time)
        except Exception as e:
            logger.error(f"Timed run failed: {e}", exc_info=True)
            # Decide how to handle: skip run, return inf? Let's return inf for now
            return float('inf')

    avg_time_ms = (total_time / n_runs) * 1000
    end_all_runs = time.time()
    logger.info(f"Speed test complete. Average time: {avg_time_ms:.2f} ms per generation.")
    logger.debug(f"Total time for {n_runs} runs: {end_all_runs - start_all_runs:.2f}s")
    return avg_time_ms

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)