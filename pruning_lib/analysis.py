import pandas as pd
import logging

logger = logging.getLogger(__name__)

def display_results(results_list: list, sort_by=None):
    """Displays evaluation results stored in a list of dicts using pandas."""
    if not results_list:
        logger.warning("No results to display.")
        return None

    results_df = pd.DataFrame(results_list)

    # Try setting index, handle potential errors if column missing
    if "Layers Removed" in results_df.columns:
         results_df.set_index("Layers Removed", inplace=True)
    elif "Identifier" in results_df.columns: # Use a generic identifier if needed
         results_df.set_index("Identifier", inplace=True)

    # Sort if requested
    if sort_by and all(col in results_df.columns for col in sort_by):
        try:
             # Determine default ascending based on common metrics
             ascending_flags = []
             for col in sort_by:
                 if "PPL" in col or "Speed" in col or "Params" in col:
                     ascending_flags.append(True) # Lower is better
                 else:
                     ascending_flags.append(True) # Default
             results_df.sort_values(by=sort_by, ascending=ascending_flags, inplace=True)
        except Exception as e:
             logger.warning(f"Could not sort DataFrame by {sort_by}: {e}")


    # Format floats for better readability
    pd.options.display.float_format = '{:,.2f}'.format
    pd.set_option('display.max_rows', 100) # Show more rows if needed
    pd.set_option('display.width', 1000) # Wider display

    print("\n--- Evaluation Summary ---")
    print(results_df)
    print("-" * 50)
    return results_df # Return the DataFrame for further use


def analyze_impact(results_df: pd.DataFrame):
    """Performs basic analysis on the results DataFrame."""
    if results_df is None or results_df.empty:
        logger.warning("Cannot analyze empty results DataFrame.")
        return

    print("\n--- Analysis ---")
    try:
        # Exclude the original model row if present
        if "None (Original)" in results_df.index:
             pruned_rows = results_df.drop("None (Original)")
        else:
             pruned_rows = results_df

        if pruned_rows.empty:
            print("No results from pruned models available for analysis.")
            return

        # Find worst PPL overall
        if "Perplexity (PPL)" in pruned_rows.columns:
            # Handle potential infinity values before finding idxmax
            valid_ppl_rows = pruned_rows[pruned_rows["Perplexity (PPL)"] != float('inf')]
            if not valid_ppl_rows.empty:
                worst_ppl_idx = valid_ppl_rows["Perplexity (PPL)"].idxmax()
                worst_ppl = valid_ppl_rows.loc[worst_ppl_idx, "Perplexity (PPL)"]
                worst_ppl_change = valid_ppl_rows.loc[worst_ppl_idx, "PPL Change (%)"] if "PPL Change (%)" in valid_ppl_rows.columns else 'N/A'
                worst_num_removed = valid_ppl_rows.loc[worst_ppl_idx, "Num Removed"] if "Num Removed" in valid_ppl_rows.columns else 'N/A'
                print(f"* Worst overall PPL ({worst_ppl:,.2f}, change: {worst_ppl_change:+.2f}%) occurred for combination: {worst_ppl_idx} (Removed: {worst_num_removed})")
            else:
                print("* Could not determine worst PPL (all pruned results might be Inf or missing).")


        # Analyze based on number of layers removed (if Num Removed column exists)
        if "Num Removed" in pruned_rows.columns:
             for k in sorted(pruned_rows["Num Removed"].unique()):
                  if k == 0: continue # Skip original if somehow included
                  rows_for_k = pruned_rows[pruned_rows["Num Removed"] == k]
                  valid_rows_for_k = rows_for_k[rows_for_k["Perplexity (PPL)"] != float('inf')]

                  if not valid_rows_for_k.empty:
                      best_ppl_for_k_idx = valid_rows_for_k["Perplexity (PPL)"].idxmin()
                      best_ppl_for_k = valid_rows_for_k.loc[best_ppl_for_k_idx, "Perplexity (PPL)"]
                      print(f"  - Best PPL when removing {k} layer(s): {best_ppl_for_k:,.2f} (Layers removed: {best_ppl_for_k_idx})")
                  else:
                      print(f"  - No valid PPL results found for removing {k} layer(s).")
        else:
             # Fallback for single layer removal analysis if 'Num Removed' is not present
             if "Perplexity (PPL)" in pruned_rows.columns and "None (Original)" in results_df.index :
                 valid_ppl_rows = pruned_rows[pruned_rows["Perplexity (PPL)"] != float('inf')]
                 if not valid_ppl_rows.empty:
                      most_impactful_idx = valid_ppl_rows["Perplexity (PPL)"].idxmax()
                      most_impactful_ppl = valid_ppl_rows.loc[most_impactful_idx, "Perplexity (PPL)"]
                      print(f"* Most impactful single layer removal (highest PPL): Layer {most_impactful_idx} (PPL: {most_impactful_ppl:,.2f})")


    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)

    print("-" * 50)