## gpt2-pruning-Kit

the project aims to explore the effects of layer pruning on perplexity, parameters, and inference speed. 

## results

### sequential layer pruning

| layer removed | layers | perplexity (↓) | speed (ms) ↓ | parameters (M) | ppl Change (%) | speed change (%) |
|---------------|--------|----------------|--------------|----------------|----------------|-------------------|
| None | 6 | 39.29 | 403.38 | 81.91 | 0.00 | 0.00 |
| 0 | 5 | 12,095.66 | 460.28 | 74.82 | +30,686.07 | +14.11 |
| 1 | 5 | 81.45 | 229.94 | 74.82 | +107.30 | -43.00 |
| 2 | 5 | 83.24 | 248.60 | 74.82 | +111.86 | -38.37 |
| 3 | 5 | 69.57 | 369.78 | 74.82 | +77.06 | -8.33 |
| 4 | 5 | 101.32 | 297.44 | 74.82 | +157.89 | -26.26 |
| 5 | 5 | 410.44 | 344.71 | 74.82 | +944.65 | -14.55 |

### combinatorial pruning (selected results)

| layers removed | remaining layers | perplexity (↓) | speed (ms) ↓ | parameters (M) | ppl Change (%) | speed change (%) |
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

## notes

1. **layer 0 (input layer)**
   removing layer 0 alone leads to catastrophic performance drops. perplexity scores shoot up dramatically, especially when it's removed along with other layers.

2. **layer 1**
   dropping layer 1 alone has a moderate effect — perplexity goes up by about 107%, but it speeds up inference by 43%. however, when you remove it along with layer 0, performance completely collapses (perplexity increases over 497,000%, check results).

3. **middle layers (2, 3, 4)**
   pruning individual middle layers has minimal effect. even combinations like (2,3) or (1,3) keep the model reasonably intact. 

4. **layer 5 (output layer)**
   this layer has a strong influence on final predictions. removing it alone raises perplexity by 945%. combining its removal with layer 0 leads to extremely poor results. it likely handles the last-stage mapping from internal representation to output tokens — without it, the model can't "speak" fluently.

## llm usage

this is an exporation/educational project whose codes were generated using ai tools. the main goal of the project was to study the effects of layer pruining on different inference parameters so i used advanced llm models to write codes for the tasks. 
