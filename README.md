# GPT2-Pruning-Kit

## Overview

**`gpt2-pruning-kit`** is a flexible toolkit for pruning layers in a GPT-2 model. It allows you to:

- Prune individual or multiple layers from GPT-2 models (in our experiment we are using `distilgpt2`)
- Evaluate the impact on **Perplexity (PPL)**, **Inference Speed**, and **Model Parameters**
- Analyze performance trends across different pruning strategies and ratios

---

## Results

### Random Layer Removal

| Metric               | Original Model | Pruned Model | Change (%)      |
|----------------------|----------------|--------------|-----------------|
| Perplexity (↓)       | 39.29          | 1511.92      | +3748.17%       |
| Inference Speed (ms) | 459.66         | 281.38       | -38.78%         |
| Parameters (M)       | 81.91          | 60.65        | -25.96%         |
| Layers               | 6              | 3            | -50.00%         |

### Sequential Layer Pruning

| Layer Removed | Layers | Perplexity (↓) | Speed (ms) ↓ | Parameters (M) | PPL Change (%) | Speed Change (%) |
|---------------|--------|----------------|--------------|----------------|----------------|-------------------|
| None          | 6      | 39.29          | 403.38       | 81.91          | 0.00           | 0.00              |
| 0             | 5      | 12,095.66      | 460.28       | 74.82          | +30686.07      | +14.11            |
| 1             | 5      | 81.45          | 229.94       | 74.82          | +107.30        | -43.00            |
| 2             | 5      | 83.24          | 248.60       | 74.82          | +111.86        | -38.37            |
| 3             | 5      | 69.57          | 369.78       | 74.82          | +77.06         | -8.33             |
| 4             | 5      | 101.32         | 297.44       | 74.82          | +157.89        | -26.26            |
| 5             | 5      | 410.44         | 344.71       | 74.82          | +944.65        | -14.55            |

### Key Observations

- Removing **Layer 0** significantly degrades performance (PPL +30,686%)
- Removing **Layer 1** results in the **fastest inference speed** (229.94 ms, -43.00%)

---

## Installation & Usage

### Installation

Navigate to the project directory and install dependencies:

```bash
cd layer_pruning_project
pip install -r requirements.txt
```

### Running Evaluations

Execute the main script with the appropriate mode:

**1. Baseline Only**

```bash
python main.py --mode baseline
```

**2. Sequential Pruning (remove 1 layer at a time)**

```bash
python main.py --mode sequential
```

**3. Combinatorial Pruning (remove 2+ layers — slower)**

```bash
python main.py --mode combinatorial
```

**4. Save Results to CSV**

Add the `--save_results` flag to any mode:

```bash
python main.py --mode sequential --save_results
```
---
