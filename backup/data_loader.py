import pandas as pd
import torch
from torch_geometric.data import Data

def load_graph_data(nodes_path='nodes.csv', edges_path='edges.csv'):
    """
    Loads the graph data from CSV files and creates a PyG Data object.

    Args:
        nodes_path (str): The path to the nodes.csv file.
        edges_path (str): The path to the edges.csv file.

    Returns:
        torch_geometric.data.Data: A PyG Data object representing the graph.
    """
    # Load nodes and edges
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # --- Node Features ---
    # The first column is the transaction ID, and the last is the class.
    # The second column is the timestep, which we'll use for splitting.
    # The rest are the node features.
    features = nodes_df.iloc[:, 2:-1].values
    x = torch.tensor(features, dtype=torch.float)

    # --- Node Labels (y) ---
    # Convert classes to numerical labels. 'unknown' -> -1, '2' (illicit) -> 1, '1' (licit) -> 0
    # In the Elliptic dataset, '2' is illicit and '1' is licit. 'unknown' should be ignored in training/evaluation.
    labels = nodes_df['class'].apply(lambda c: 1 if c == '2' else (0 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)

    # --- Edge Index ---
    # PyG requires the edge index in a specific [2, num_edges] format.
    # We also need to map the transaction IDs to zero-based indices.
    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    source_indices = [tx_id_map[src] for src in edges_df['source'] if src in tx_id_map]
    target_indices = [tx_id_map[target] for target in edges_df['target'] if target in tx_id_map]
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

    # --- Timestep for Splitting ---
    timesteps = torch.tensor(nodes_df.iloc[:, 1].values, dtype=torch.long)

    # --- Create the Data Object ---
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps

    # --- Create Masks for Splitting ---
    # We will split the data based on the timestep.
    # The first 34 timesteps are for training, 35-41 for validation, and 42-49 for testing.
    # We only care about nodes with known labels for training and evaluation.
    known_mask = y != -1

    train_mask = (timesteps < 35) & known_mask
    val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    test_mask = (timesteps >= 42) & known_mask

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

if __name__ == '__main__':
    # A simple test to see if the data loader works.
    graph_data = load_graph_data()
    print("Data loaded successfully!")
    print(graph_data)
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Number of training nodes: {graph_data.train_mask.sum()}")
    print(f"Number of validation nodes: {graph_data.val_mask.sum()}")
    print(f"Number of test nodes: {graph_data.test_mask.sum()}")
