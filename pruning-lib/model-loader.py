import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """
    Loads the specified Hugging Face model and tokenizer.

    Args:
        model_name (str): The name of the model to load (e.g., "distilgpt2").
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: (model, tokenizer, original_config) or (None, None, None) on failure.
    """
    try:
        logger.info(f"Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Handle padding token for GPT-2 style models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")

        logger.info(f"Loading model '{model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Set to evaluation mode by default
        original_config = model.config # Store original config

        logger.info(f"Model '{model_name}' loaded successfully on {device}.")
        logger.info(f"Original layers: {original_config.n_layer}, Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        return model, tokenizer, original_config

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer '{model_name}': {e}", exc_info=True)
        return None, None, None