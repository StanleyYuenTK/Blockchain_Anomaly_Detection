import pandas as pd
import os
import kagglehub

def preprocess_data(data_dir):
    """
    Preprocesses the Elliptic dataset.

    Args:
        data_dir (str): The directory where the raw data is located.
    """
    output_dir = './'

    # Construct full paths for the raw data files
    classes_path = os.path.join(data_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(data_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(data_dir, 'elliptic_txs_features.csv')

    # Load the raw data
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)

    # The first column of features_df is the transaction ID, let's name it
    features_df.rename(columns={0: 'txId'}, inplace=True)

    # Merge features and classes to create the nodes dataframe
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')

    # The edgelist is already in the desired format, just needs to be saved
    # We can rename the columns to be more descriptive
    edgelist_df.rename(columns={'txId1': 'source', 'txId2': 'target'}, inplace=True)

    # Save the processed data
    nodes_df.to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)
    edgelist_df.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)

    print("Preprocessing complete. 'nodes.csv' and 'edges.csv' have been created.")

if __name__ == "__main__":
    # Download latest version
    print("Downloading the Elliptic dataset...")
    path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    print(f"Dataset downloaded to: {path}")

    # The actual data is in a subdirectory
    data_subdirectory = os.path.join(path, 'elliptic_bitcoin_dataset')

    print("Starting data preprocessing...")
    preprocess_data(data_subdirectory)
