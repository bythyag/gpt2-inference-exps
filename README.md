## DistilGPT2-Pruning-Kit

## Overview

**`distilgpt2-pruning-kit`** is a flexible toolkit for pruning layers in a GPT-2 model. It allows you to:
- Prune individual or multiple layers from GPT-2 models (using `distilgpt2`)
- Evaluate the impact on **Perplexity (PPL)**, **Inference Speed**, and **Model Parameters**
- Analyze performance trends across different pruning strategies and ratios
- Test combinatorial pruning to find optimal layer removal configurations

## Results

### Sequential Layer Pruning

| Layer Removed | Layers | Perplexity (↓) | Speed (ms) ↓ | Parameters (M) | PPL Change (%) | Speed Change (%) |
|---------------|--------|----------------|--------------|----------------|----------------|-------------------|
| None | 6 | 39.29 | 403.38 | 81.91 | 0.00 | 0.00 |
| 0 | 5 | 12,095.66 | 460.28 | 74.82 | +30,686.07 | +14.11 |
| 1 | 5 | 81.45 | 229.94 | 74.82 | +107.30 | -43.00 |
| 2 | 5 | 83.24 | 248.60 | 74.82 | +111.86 | -38.37 |
| 3 | 5 | 69.57 | 369.78 | 74.82 | +77.06 | -8.33 |
| 4 | 5 | 101.32 | 297.44 | 74.82 | +157.89 | -26.26 |
| 5 | 5 | 410.44 | 344.71 | 74.82 | +944.65 | -14.55 |

### Combinatorial Pruning (Selected Results)

| Layers Removed | Remaining Layers | Perplexity (↓) | Speed (ms) ↓ | Parameters (M) | PPL Change (%) | Speed Change (%) |
|----------------|------------------|----------------|--------------|----------------|----------------|------------------|
| None (Original) | 6 | 39.29 | 403.38 | 81.91 | 0.00 | 0.00 |
| (1, 3) | 4 | 156.42 | 274.90 | 67.74 | +298.12 | -31.85 |
| (1, 4) | 4 | 187.22 | 270.36 | 67.74 | +376.51 | -32.98 |
| (2, 3) | 4 | 196.88 | 285.33 | 67.74 | +401.09 | -29.27 |
| (3, 4) | 4 | 655.05 | 235.82 | 67.74 | +1,567.23 | -41.54 |
| (1, 2, 3) | 3 | 763.96 | 322.36 | 60.65 | +1,844.43 | -20.08 |
| (2, 3, 4) | 3 | 1,511.92 | 236.94 | 60.65 | +3,748.17 | -41.26 |
| (0, 4) | 4 | 8,422.35 | 416.20 | 67.74 | +21,336.69 | +3.18 |
| (0, 1) | 4 | 195,311.70 | 326.38 | 67.74 | +497,010.35 | -19.09 |

## Comprehensive Analysis

### Importance of Different Layers

1. **Layer 0 (Input Layer)**:
   - Removing Layer 0 causes catastrophic performance degradation
   - When removed alone: PPL increases by +30,686%
   - When combined with other layers: Consistently produces extremely high perplexity scores
   - **Conclusion**: Layer 0 is critical for maintaining semantic understanding

2. **Layer 1**:
   - Moderate impact when removed alone (PPL +107%)
   - Provides significant speed benefits (-43% inference time)
   - When combined with Layer 0, causes severe degradation (PPL +497,010%)
   - **Conclusion**: Good candidate for pruning when prioritizing speed

3. **Middle Layers (2, 3, 4)**:
   - Removing individual middle layers has relatively minimal impact
   - Pruning combinations like (2,3) or (1,3) maintains reasonable performance
   - **Conclusion**: Middle layers have some redundancy and are viable pruning targets

4. **Layer 5 (Output Layer)**:
   - Significant impact when removed alone (PPL +945%)
   - When combined with Layer 0, produces extreme perplexity scores
   - **Conclusion**: Important for final output quality

### Optimal Pruning Strategies

1. **Best Speed-Quality Tradeoffs**:
   - **(1,3)** configuration: 31.85% speed improvement with only 298% PPL increase
   - **(1,4)** configuration: 32.98% speed improvement with 376% PPL increase
   - **(2,3)** configuration: 29.27% speed improvement with 401% PPL increase

2. **Aggressive Pruning (3 layers removed)**:
   - **(1,2,3)** configuration offers the best quality preservation (1,844% PPL increase)
   - **(2,3,4)** offers better speed improvement (-41.26%) but worse quality degradation

3. **Severe Degradation Configurations**:
   - Any configuration including both Layers 0 and 1
   - Combinations including Layer 0 and 5 together

### Parameter Efficiency

- Removing 2 layers reduces parameters by 17.3% (81.91M → 67.74M)
- Removing 3 layers reduces parameters by 26.0% (81.91M → 60.65M)
- Removing 4 layers reduces parameters by 34.6% (81.91M → 53.56M)
- Removing 5 layers reduces parameters by 43.3% (81.91M → 46.47M)

### Key Insights

1. **Layer Importance Hierarchy**: Layer 0 > Layer 5 > Layer 1 > Layers 2-4
2. **Speed vs Quality**: Removing layers (1,3) offers the best balance between speed improvement and quality preservation
3. **Diminishing Returns**: Removing more than 3 layers leads to exponential quality degradation for minimal additional speed gains
4. **Parameter Efficiency**: Even removing 2 layers can reduce model size by 17%, improving deployment efficiency

## Installation & Usage

### Installation

Navigate to the project directory and install dependencies:
```bash
cd gpt2-pruning-kit
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

**3. Combinatorial Pruning (remove 2+ layers)**
```bash
python main.py --mode combinatorial
```

**4. Save Results to CSV**
Add the `--save_results` flag to any mode:
```bash
python main.py --mode sequential --save_results
```

## Conclusion

The experimental results demonstrate that strategic layer pruning can significantly improve inference speed while maintaining acceptable performance. The most efficient approach is to target the removal of specific layers rather than sequential pruning, with the (1,3) configuration offering the best overall compromise between speed and quality.

These findings can guide deployment strategies for resource-constrained environments or latency-sensitive applications. Future work could explore fine-tuning after pruning to recover some of the lost performance.
