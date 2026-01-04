# GCN for Elliptic Dataset

This project implements a Graph Convolutional Network (GCN) to detect illicit transactions in the Elliptic dataset.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare the Data:**
    This script will download the Elliptic dataset and preprocess it into `nodes.csv` and `edges.csv`.
    ```bash
    python prepare_data.py
    ```

## Training

To train the GCN model, run the following command:
```bash
python train.py
```

You can also specify hyperparameters as command-line arguments:
```bash
python train.py --learning_rate 0.005 --hidden_channels 256 --epochs 150
```
