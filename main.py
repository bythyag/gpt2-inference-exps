import torch
import gc
import time
import logging
import argparse
import os
from itertools import combinations

# Import configurations and functions from our library
from pruning_lib import config
from pruning_lib import model_loader
from pruning_lib import pruning
from pruning_lib import evaluation
from pruning_lib import analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger(__name__)

def run_baseline_evaluation(model, tokenizer):
    """Evaluates the original, unpruned model."""
    logger.info("--- Evaluating Original Model (Baseline) ---")
    start_time = time.time()

    ppl_original = evaluation.calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name=config.PERPLEXITY_DATASET,
        dataset_config=config.PERPLEXITY_CONFIG,
        split=config.PERPLEXITY_SPLIT,
        stride=config.PPL_STRIDE,
        device=config.DEVICE
    )

    speed_original_ms = evaluation.measure_inference_speed(
        model=model,
        tokenizer=tokenizer,
        prompt=config.INFERENCE_PROMPT,
        generation_config=config.GENERATION_CONFIG.copy(), # Pass copy
        n_runs=config.SPEED_TEST_RUNS,
        device=config.DEVICE
    )
    params_original_M = evaluation.count_parameters(model) / 1e6
    layers_original = model.config.n_layer

    baseline_result = {
        "Layers Removed": "None (Original)",
        "Num Removed": 0,
        "Layers": layers_original,
        "Perplexity (PPL)": ppl_original,
        "Inference Speed (ms)": speed_original_ms,
        "Parameters (M)": params_original_M,
        "PPL Change (%)": 0.0,
        "Speed Change (%)": 0.0
    }
    end_time = time.time()
    logger.info(f"Baseline evaluation finished in {end_time - start_time:.2f} seconds.")
    return baseline_result

def run_sequential_evaluation(original_model, tokenizer, baseline_result):
    """Runs evaluation by removing one layer at a time."""
    logger.info("--- Starting Sequential Pruning Evaluation (Removing 1 Layer at a Time) ---")
    all_results = [baseline_result]
    original_num_layers = original_model.config.n_layer
    ppl_original = baseline_result["Perplexity (PPL)"]
    speed_original_ms = baseline_result["Inference Speed (ms)"]

    for i in range(original_num_layers):
        logger.info(f">>> Processing: Removing Layer {i} <<<")
        start_time_layer = time.time()

        pruned_model = pruning.prune_single_transformer_layer(original_model, i)
        if pruned_model is None:
            logger.warning(f"Skipping evaluation for layer {i} due to pruning error.")
            continue

        ppl_pruned = evaluation.calculate_perplexity(
            model=pruned_model, tokenizer=tokenizer, dataset_name=config.PERPLEXITY_DATASET,
            dataset_config=config.PERPLEXITY_CONFIG, split=config.PERPLEXITY_SPLIT,
            stride=config.PPL_STRIDE, device=config.DEVICE
        )
        speed_pruned_ms = evaluation.measure_inference_speed(
            model=pruned_model, tokenizer=tokenizer, prompt=config.INFERENCE_PROMPT,
            generation_config=config.GENERATION_CONFIG.copy(), n_runs=config.SPEED_TEST_RUNS, device=config.DEVICE
        )
        params_pruned_M = evaluation.count_parameters(pruned_model) / 1e6

        # Calculate changes
        ppl_change = ((ppl_pruned - ppl_original) / ppl_original * 100) if ppl_original and ppl_original != 0 and ppl_pruned != float('inf') else float('inf')
        speed_change = ((speed_pruned_ms - speed_original_ms) / speed_original_ms * 100) if speed_original_ms and speed_original_ms != 0 and speed_pruned_ms != float('inf') else float('inf')

        current_result = {
            "Layers Removed": str(i), # Keep as string for consistency
            "Num Removed": 1,
            "Layers": pruned_model.config.n_layer,
            "Perplexity (PPL)": ppl_pruned,
            "Inference Speed (ms)": speed_pruned_ms,
            "Parameters (M)": params_pruned_M,
            "PPL Change (%)": ppl_change,
            "Speed Change (%)": speed_change
        }
        all_results.append(current_result)

        # Clean up memory
        del pruned_model
        gc.collect()
        if config.DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()
        end_time_layer = time.time()
        logger.info(f"Finished processing layer {i} in {end_time_layer - start_time_layer:.2f} seconds.")

    return all_results


def run_combinatorial_evaluation(original_model, tokenizer, baseline_result):
    """Runs evaluation by removing k layers at a time (k=2, 3, ...)."""
    logger.info("--- Starting Combinatorial Pruning Evaluation ---")
    all_results = [baseline_result]
    original_num_layers = original_model.config.n_layer
    ppl_original = baseline_result["Perplexity (PPL)"]
    speed_original_ms = baseline_result["Inference Speed (ms)"]
    layer_indices = list(range(original_num_layers))

    max_k = config.MAX_COMBINATORIAL_K or (original_num_layers - 1)
    if max_k >= original_num_layers:
        max_k = original_num_layers - 1
        logger.warning(f"Adjusted max layers to remove to {max_k} (cannot remove all layers).")

    total_combinations_estimated = sum(1 for k in range(2, max_k + 1) for _ in combinations(layer_indices, k))
    logger.info(f"Will evaluate combinations for k=2 to {max_k} (estimated {total_combinations_estimated} combinations). This may take a long time.")

    combination_counter = 0
    for k_layers_to_remove in range(2, max_k + 1):
        logger.info(f"===== Evaluating Combinations with {k_layers_to_remove} Layers Removed =====")
        num_combs_for_k = len(list(combinations(layer_indices, k_layers_to_remove)))
        logger.info(f"Total combinations for k={k_layers_to_remove}: {num_combs_for_k}")

        for indices_to_remove in combinations(layer_indices, k_layers_to_remove):
            combination_counter += 1
            indices_tuple = tuple(sorted(indices_to_remove))
            logger.info(f">>> Processing Combination {combination_counter}/{total_combinations_estimated} (Remove Layers: {indices_tuple}) <<<")
            start_time_comb = time.time()

            pruned_model = pruning.prune_multiple_transformer_layers(original_model, indices_to_remove)
            if pruned_model is None:
                 logger.warning(f"Skipping evaluation for combination {indices_tuple} due to pruning error.")
                 continue

            try:
                ppl_pruned = evaluation.calculate_perplexity(
                    model=pruned_model, tokenizer=tokenizer, dataset_name=config.PERPLEXITY_DATASET,
                    dataset_config=config.PERPLEXITY_CONFIG, split=config.PERPLEXITY_SPLIT,
                    stride=config.PPL_STRIDE, device=config.DEVICE
                )
                speed_pruned_ms = evaluation.measure_inference_speed(
                    model=pruned_model, tokenizer=tokenizer, prompt=config.INFERENCE_PROMPT,
                    generation_config=config.GENERATION_CONFIG.copy(), n_runs=config.SPEED_TEST_RUNS, device=config.DEVICE
                )
                params_pruned_M = evaluation.count_parameters(pruned_model) / 1e6
            except Exception as e:
                 logger.error(f"Evaluation failed for combination {indices_tuple}: {e}", exc_info=True)
                 ppl_pruned = float('inf')
                 speed_pruned_ms = float('inf')
                 # Still try to get params if model exists
                 params_pruned_M = evaluation.count_parameters(pruned_model) / 1e6 if pruned_model else float('nan')


            # Calculate changes
            ppl_change = ((ppl_pruned - ppl_original) / ppl_original * 100) if ppl_original and ppl_original != 0 and ppl_pruned != float('inf') else float('inf')
            speed_change = ((speed_pruned_ms - speed_original_ms) / speed_original_ms * 100) if speed_original_ms and speed_original_ms != 0 and speed_pruned_ms != float('inf') else float('inf')

            current_result = {
                "Layers Removed": str(indices_tuple),
                "Num Removed": k_layers_to_remove,
                "Layers": pruned_model.config.n_layer,
                "Perplexity (PPL)": ppl_pruned,
                "Inference Speed (ms)": speed_pruned_ms,
                "Parameters (M)": params_pruned_M,
                "PPL Change (%)": ppl_change,
                "Speed Change (%)": speed_change
            }
            all_results.append(current_result)

            # Clean up memory
            del pruned_model
            gc.collect()
            if config.DEVICE == torch.device("cuda"):
                torch.cuda.empty_cache()

            end_time_comb = time.time()
            logger.info(f"Finished processing combination {indices_tuple} in {end_time_comb - start_time_comb:.2f} seconds.")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run layer pruning evaluations on a transformer model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "combinatorial", "baseline"],
        required=True,
        help="Type of evaluation to run: 'baseline' (only original model), 'sequential' (remove 1 layer at a time), 'combinatorial' (remove 2+ layers)."
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save the results DataFrame to a CSV file in the results directory."
    )
    args = parser.parse_args()

    logger.info(f"Starting pruning evaluation with mode: {args.mode}")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME}")

    # Load model and tokenizer
    original_model, tokenizer, _ = model_loader.load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    if original_model is None or tokenizer is None:
        logger.error("Exiting due to model loading failure.")
        return

    # Ensure pad token ID is set in global config for speed tests
    if config.GENERATION_CONFIG.get("pad_token_id") is None:
        config.GENERATION_CONFIG["pad_token_id"] = tokenizer.eos_token_id

    # Always run baseline evaluation first
    baseline_result = run_baseline_evaluation(original_model, tokenizer)
    all_results = [baseline_result] # Start results list with baseline

    # Run selected evaluation mode
    start_time_total = time.time()
    if args.mode == "sequential":
        all_results = run_sequential_evaluation(original_model, tokenizer, baseline_result)
    elif args.mode == "combinatorial":
        all_results = run_combinatorial_evaluation(original_model, tokenizer, baseline_result)
    # If mode is 'baseline', we've already done the work.

    end_time_total = time.time()
    logger.info(f"Evaluation mode '{args.mode}' completed in {(end_time_total - start_time_total)/60:.2f} minutes.")

    # Display results
    results_df = analysis.display_results(all_results, sort_by=["Num Removed", "Perplexity (PPL)"])

    # Perform analysis
    if results_df is not None:
        analysis.analyze_impact(results_df)

    # Save results if requested
    if args.save_results and results_df is not None:
        if not os.path.exists(config.RESULTS_DIR):
            os.makedirs(config.RESULTS_DIR)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(config.RESULTS_DIR, f"{config.MODEL_NAME}_pruning_{args.mode}_results_{timestamp}.csv")
        try:
            results_df.to_csv(filename)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")

    logger.info("Evaluation script finished.")


if __name__ == "__main__":
    main()