# Blockchain_Anomaly_Detection

## Introduction
This repository contains the implementation of my Final Year Project (FYP) at **The Hong Kong Polytechnic University**, which focused on detecting illicit transactions in blockchain networks. The project achieved a final grade of **B+**.

The framework leverages Graph Neural Networks (GNNs), advanced feature engineering, and ensemble learning techniques to identify anomalies in transaction graphs. We specifically focus on the Elliptic (Bitcoin) and Ethereum datasets.

## Installation & Prerequisites
The proposed methodology is implemented in a Python environment using several key libraries:
- **PyTorch Geometric**: For implementing and training GNN models, with CUDA 12.6 acceleration.
- **CatBoost**: Used as the meta-classifier for the stacking ensemble.
- **Optuna**: For automated hyperparameter tuning via Tree-structured Parzen Estimator (TPE).
- **PyGAD**: For the genetic algorithm used in model selection for the ensemble.
- **Other Dependencies**: `kornia`, `scikit-learn`, `pandas`, `numpy`, `networkx`, `community`, `matplotlib`, `seaborn`, `torch_scatter`.

### Experimental Environment
All experiments were conducted on high-performance hardware:
- **GPU**: NVIDIA A800-SXM4-80GB (utilizing a 40GB MIG partition).
- **CPU**: 8 allocated cores.
- **RAM**: 1.6TB available system RAM.

## Data Acquisition
### Elliptic Dataset (Bitcoin)
The Elliptic dataset is collected from Kaggle and represents a network of Bitcoin transactions.
- **Citation**: M. Weber et al., “Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics,” pp. 1-7, 2019. [Online]. doi: 10.48550/arxiv.1908.02591 [Accessed Sep. 20, 2025].

### Ethereum Dataset
The Ethereum dataset was collected from X-Block, focusing on phishing scam detection.
- **Citation**: L. Chen, J. Peng, Y. Liu, J. Li, F. Xie, and Z. Zheng, “Phishing scams detection in Ethereum transaction network,” ACM Trans. Internet Technol., vol. 21, no. 1, pp. 1–16, 2021. [Online]. doi: 10.1145/3398071. [Accessed Feb. 1, 2026].

## Methodology
The framework follows a multi-stage pipeline:
1. **Feature Engineering**: Beyond the raw features, we incorporate graph-theoretical metrics including **PageRank**, **In/Out Degree statistics**, **Louvain Community Detection**, and **Clustering Coefficients** to capture the structural properties of the transaction network.
2. **GNN Model Zoo**: We explore a variety of GNN architectures:
   - Graph Convolutional Network (GCN)
   - Graph Attention Network (GAT)
   - GraphSAGE
   - Graph Isomorphism Network (GIN)
   - APPNP
   - ChebNet
   - MixHop
3. **Hyperparameter Optimization**: Each GNN model is individually fine-tuned using **Optuna** to find the optimal configuration for each dataset.
4. **Genetic Algorithm Model Selection**: A Genetic Algorithm (**PyGAD**) is employed to select the most effective subset of GNN models to participate in the final ensemble, optimizing for a balance between F1-score and AUC.
5. **Stacking Ensemble**: The final prediction is made using a **CatBoost** meta-classifier that takes the prediction probabilities from the GA-selected GNN models combined with the original node features.

## Usage
### 1. Data Preprocessing
Run the following commands to process the raw datasets and generate the engineered features:
```bash
python dataset_zoo.py e1  # Process Elliptic dataset
python dataset_zoo.py e2  # Process Ethereum dataset
```

### 2. Hyperparameter Tuning
To fine-tune the GNN models and save the best parameters:
```bash
python gnn_optimal.py e1  # Tune for Elliptic
python gnn_optimal.py e2  # Tune for Ethereum
```

### 3. Run Main Experiment
To run the full pipeline (GNN training, GA selection, and Stacking Ensemble):
```bash
python main.py e1  # Run for Elliptic
python main.py e2  # Run for Ethereum
```

## Results
We evaluated several Graph Neural Network (GNN) models and an ensemble learning approach (CatBoost) on two benchmark datasets: Elliptic (Bitcoin) and Ethereum. Below are the key insights and performance highlights from our experiments.
<img width="766" height="496" alt="image" src="https://github.com/user-attachments/assets/02cff7bf-99a9-49cf-8616-e9954542e150" />

1. Base GNNs vs. Advanced GNNs
Hyperparameter Tuning Impact: After rigorous hyperparameter tuning (specifically adjusting the number of layers), the base GNN models (GCN, GAT, GIN, and GraphSAGE) achieved performance comparable to more complex models like APPNP, ChebNet, and MixHop on the test set.
Layer Depth vs. Hop Count: Adjusting the layer depth in base GNNs effectively boosts anomaly detection performance. Conversely, advanced models (APPNP, ChebNet, MixHop) did not show significant performance gains as the number of hops increased.

2. CatBoost Performance (The Winning Approach)
The ensemble learning approach (CatBoost) consistently outperformed all single GNN models across most evaluation metrics on both datasets, demonstrating the strong capability of gradient boosting in tabular/graph feature combinations.

## Ablation Study
In this ablation study, community features are not included in the dataset.
<img width="764" height="473" alt="image" src="https://github.com/user-attachments/assets/dfccf69c-197a-4e2a-96c2-04408a64db60" />

1. Elliptic Dataset (Bitcoin)
Recall vs. Precision: Integrating community detection features led to a slight increase in macro recall but caused an overall decrease in macro precision.
Model Variances:
GraphSAGE, APPNP, MixHop, and CatBoost experienced a substantial drop in macro precision, with reductions ranging from 10% to 30%.
ChebNet and GAT stood out as exceptions, demonstrating a slight improvement in macro precision.

2. Ethereum Dataset
Fluctuating Trends: Similar to the Elliptic dataset, models showed slight fluctuations in macro recall alongside a general decline in macro precision.
Model Variances:
Most models suffered a significant drop in macro precision of approximately 10%.
GraphSAGE was the sole exception, showing a minor increase of +0.00564 in macro precision.
