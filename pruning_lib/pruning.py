import torch
from torch import nn
import copy
import logging
from typing import List, Union, Set, Tuple

logger = logging.getLogger(__name__)

def _get_transformer_layers(model):
    """Helper to find the transformer layer list in common architectures."""
    # Add more checks here if supporting different model types
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, 'h', model.transformer
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'): # E.g., LLaMA
         return model.model.layers, 'layers', model.model
    # Add other potential paths (e.g., model.encoder.layer for BERT-like)
    else:
        raise AttributeError("Cannot automatically find the transformer layer list in the model. Check architecture.")


def prune_single_transformer_layer(original_model: torch.nn.Module, layer_index_to_remove: int):
    """
    Prunes a single specified transformer layer.

    Args:
        original_model (torch.nn.Module): The baseline model (will NOT be modified).
        layer_index_to_remove (int): The 0-based index of the layer to remove.

    Returns:
        torch.nn.Module or None: A deep copy of the model with the layer removed, or None on failure.
    """
    pruned_model = copy.deepcopy(original_model)
    try:
        layers, layer_attr, parent_module = _get_transformer_layers(pruned_model)
    except AttributeError as e:
        logger.error(f"Pruning failed: {e}")
        return None

    original_num_layers = len(layers)

    if not 0 <= layer_index_to_remove < original_num_layers:
        logger.error(f"Invalid layer index {layer_index_to_remove}. Must be between 0 and {original_num_layers - 1}.")
        return None

    num_layers_to_keep = original_num_layers - 1
    logger.info(f"Pruning layer {layer_index_to_remove}. Keeping {num_layers_to_keep} layers.")

    pruned_layers_list = [layers[i] for i in range(original_num_layers) if i != layer_index_to_remove]
    new_module_list = nn.ModuleList(pruned_layers_list)
    setattr(parent_module, layer_attr, new_module_list)

    # Update config (important!)
    if hasattr(pruned_model.config, 'n_layer'):
        pruned_model.config.n_layer = num_layers_to_keep
        logger.debug(f"Updated model config 'n_layer' to: {pruned_model.config.n_layer}")
    else:
        logger.warning("Model config does not have 'n_layer'. Ensure model forward pass doesn't depend on it.")

    pruned_model.to(original_model.device)
    pruned_model.eval()
    logger.debug(f"Pruned model layers: {len(getattr(parent_module, layer_attr))}")
    return pruned_model


def prune_multiple_transformer_layers(original_model: torch.nn.Module, layer_indices_to_remove: Union[List[int], Set[int], Tuple[int]]):
    """
    Prunes specific transformer layers identified by their indices.

    Args:
        original_model (torch.nn.Module): The baseline model (will NOT be modified).
        layer_indices_to_remove (Union[List[int], Set[int], Tuple[int]]): Indices of layers to remove (0-based).

    Returns:
        torch.nn.Module or None: A deep copy of the model with specified layers removed, or None on failure.
    """
    pruned_model = copy.deepcopy(original_model)
    try:
        layers, layer_attr, parent_module = _get_transformer_layers(pruned_model)
    except AttributeError as e:
        logger.error(f"Pruning failed: {e}")
        return None

    original_num_layers = len(layers)
    indices_to_remove_set = set(layer_indices_to_remove)

    if not all(0 <= idx < original_num_layers for idx in indices_to_remove_set):
        invalid_indices = [idx for idx in indices_to_remove_set if not (0 <= idx < original_num_layers)]
        logger.error(f"Invalid layer indices detected: {invalid_indices}. Max index is {original_num_layers - 1}.")
        return None

    num_layers_to_remove = len(indices_to_remove_set)
    num_layers_to_keep = original_num_layers - num_layers_to_remove

    if num_layers_to_keep <= 0:
        logger.error(f"Pruning indices {layer_indices_to_remove} would result in 0 layers.")
        return None

    logger.info(f"Pruning layers {sorted(list(indices_to_remove_set))}. Keeping {num_layers_to_keep} layers.")

    pruned_layers_list = []
    kept_indices = []
    for i in range(original_num_layers):
        if i not in indices_to_remove_set:
            pruned_layers_list.append(layers[i])
            kept_indices.append(i)
    logger.debug(f"Indices of layers KEPT: {kept_indices}")

    new_module_list = nn.ModuleList(pruned_layers_list)
    setattr(parent_module, layer_attr, new_module_list)

    # Update config
    if hasattr(pruned_model.config, 'n_layer'):
        pruned_model.config.n_layer = num_layers_to_keep
        logger.debug(f"Updated model config 'n_layer' to: {pruned_model.config.n_layer}")

    pruned_model.to(original_model.device)
    pruned_model.eval()
    logger.debug(f"Pruned model layers: {len(getattr(parent_module, layer_attr))}")
    return pruned_model