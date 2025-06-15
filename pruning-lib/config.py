import torch

# --- Model & Tokenizer ---
MODEL_NAME = "distilgpt2"

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation Settings ---
PERPLEXITY_DATASET = "wikitext"
PERPLEXITY_CONFIG = "wikitext-2-raw-v1"
PERPLEXITY_SPLIT = "test"
PPL_STRIDE = 512 # Stride for perplexity calculation

# --- Inference Speed Test Settings ---
INFERENCE_PROMPT = "The future of artificial intelligence is"
SPEED_TEST_RUNS = 10 # Number of runs to average for speed test

# --- Text Generation Settings (used for speed test and qualitative) ---
GENERATION_CONFIG = {
    "max_length": 75,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.7,
    # pad_token_id will be set dynamically after tokenizer is loaded
}

# --- Pruning Execution Settings ---
# Maximum number of layers to remove in combinatorial search (e.g., up to N-1)
# Set to None to go up to total_layers - 1
MAX_COMBINATORIAL_K = None

# --- Output ---
RESULTS_DIR = "results" # Optional: Directory to save CSV results