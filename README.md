## gpt2-pruning-kit

### Introduction

The codebase is a pruning kit which you can use to prune single/multiple layer in your gpt2 model, evaluate its Perplexity (PPL) score, Inference Speed (ms) and how model parameters changes across layers and with change in pruning ratio. 

### Results

--- Evaluation Summary For Removing a Random Layer ---
Model Name: distilgpt2
Pruning Ratio: 0.50 (Random Layer Removal)
Metric	Original Model	Pruned Model	Change (%)
Perplexity (PPL) ↓	39.29	1511.92	+3748.17%
Inference Speed (ms) ↓	459.66	281.38	-38.78%
Parameters (M) ↓	81.91	60.65	-25.96%
Layers	6	3	-50.00%

--- Sequential Pruning Evaluation Summary ---
                 Layers  Perplexity (PPL)  Inference Speed (ms)  \
Layer Removed                                                     
None (Original)       6             39.29                403.38   
0                     5         12,095.66                460.28   
1                     5             81.45                229.94   
2                     5             83.24                248.60   
3                     5             69.57                369.78   
4                     5            101.32                297.44   
5                     5            410.44                344.71   

                 Parameters (M)  PPL Change (%)  Speed Change (%)  
Layer Removed                                                      
None (Original)           81.91            0.00              0.00  
0                         74.82       30,686.07             14.11  
1                         74.82          107.30            -43.00  
2                         74.82          111.86            -38.37  
3                         74.82           77.06             -8.33  
4                         74.82          157.89            -26.26  
5                         74.82          944.65            -14.55  

--- Analysis ---
Removing Layer 0 had the most significant negative impact on performance.
  - Resulting PPL: 12,095.66 (+30686.07% change from original)
Removing Layer 1 resulted in the fastest inference speed.
  - Resulting Speed: 229.94 ms (-43.00% change from original)


### Notes on Installation

Install Requirements: Open your terminal, navigate into the layer_pruning_project directory, and run:
```
pip install -r requirements.txt
```

Run Evaluation: From the terminal inside the layer_pruning_project directory, execute the main script with the desired mode:
    1. Baseline Only:
        ```
        python main.py --mode baseline
        ```
    2. Sequential (Remove 1 layer at a time):
        ```
        python main.py --mode sequential
        ```
    3. Combinatorial (Remove 2+ layers - WILL BE SLOW):
        ```
        python main.py --mode combinatorial
        ```
    4. Save Results to CSV: Add the --save_results flag to any mode:
        ```
        python main.py --mode sequential --save_results
        ```