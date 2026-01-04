"""
blockchain_baseline_model.py

Baseline model for blockchain anomaly detection using IsolationForest.
Compares performance with GNN models to demonstrate GNN's superiority
in detecting blockchain anomalies.

Features: Only uses node features (data.x), ignores graph structure (edge_index).
Model: sklearn.ensemble.IsolationForest
Metrics: Same evaluation metrics as GNN models (F1-Score, Recall, G-Mean)

Execution: Runs automatically with pre-configured parameters, no command-line arguments needed.
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

# Visualization tools
from visualization_tools import plot_confusion_matrix, plot_feature_visualization_2d, TrainingHistory

warnings.filterwarnings('ignore')


def load_elliptic_data_for_baseline(dataset_dir='../Dataset'):
    """
    Load Elliptic dataset specifically for baseline model.
    Returns only features and labels, ignores graph structure.
    """
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')

    if not all(os.path.exists(p) for p in [classes_path, features_path]):
        raise FileNotFoundError("Dataset files not found, please ensure Dataset folder contains required CSV files")

    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId'}, inplace=True)

    # Merge features and labels
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    feature_columns = nodes_df.columns[2:-1]  # Skip txId and timestep
    features = nodes_df[feature_columns].values
    labels = nodes_df['class'].apply(lambda c: 1 if c == '2' else (0 if c == '1' else -1))

    # Split train/validation/test sets
    timesteps = nodes_df.iloc[:, 1].values  # timestep column
    train_mask = timesteps < 35
    val_mask = (timesteps >= 35) & (timesteps < 42)
    test_mask = timesteps >= 42

    # Keep only labeled samples
    known_mask = labels != -1
    train_mask = train_mask & known_mask
    val_mask = val_mask & known_mask
    test_mask = test_mask & known_mask

    return {
        'features': features,
        'labels': labels.values,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'timesteps': timesteps
    }


def train_isolation_forest(X_train, contamination='auto', random_state=42, **kwargs):
    """
    Train IsolationForest model.

    Args:
        X_train: Training features
        contamination: Expected proportion of anomalies
        random_state: Random seed
        **kwargs: Other IsolationForest parameters

    Returns:
        trained_model: Trained IsolationForest model
        scaler: Feature scaler
    """
    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # If contamination is not specified, set automatically based on illicit transaction ratio in training set
    if contamination == 'auto':
        # Here we need labels to estimate contamination, but IsolationForest is unsupervised
        # We use a reasonable default value or specify manually
        contamination = 0.1  # Assume anomaly ratio is 10%

    # Create and train model
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        **kwargs
    )

    model.fit(X_train_scaled)

    return model, scaler


def predict_isolation_forest(model, scaler, X):
    """
    Make predictions using trained IsolationForest.

    Args:
        model: Trained IsolationForest model
        scaler: Feature scaler
        X: Features to predict

    Returns:
        predictions: Prediction results (-1: anomaly, 1: normal)
        scores: Anomaly scores (lower means more anomalous)
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    # IsolationForest returns -1 (anomaly) and 1 (normal)
    # We need to convert to our label format: 1 (illicit/anomaly), 0 (licit/normal)
    predictions_binary = (predictions == -1).astype(int)

    return predictions_binary, scores


def evaluate_baseline_model(y_true, y_pred, scores=None, detailed=False):
    """
    Evaluate baseline model performance using same metrics as GNN.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Anomaly scores (for AUC calculation)
        detailed: Whether to output detailed report

    Returns:
        metrics: Dictionary containing various metrics
    """
    # Basic metrics
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Metrics for illicit class (label=1)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro metrics
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC (if scores are available)
    if scores is not None:
        try:
            auc = roc_auc_score(y_true, -scores)  # IsolationForest scores are lower for anomalies, so take negative
            macro_auc = auc  # For binary classification, macro AUC equals regular AUC
        except:
            auc = float('nan')
            macro_auc = float('nan')
    else:
        auc = float('nan')
        macro_auc = float('nan')

    # G-Mean
    try:
        tpr = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # Illicit class recall
        tnr = recall_score(y_true, y_pred, pos_label=0, zero_division=0)  # Licit class recall (specificity)
        gmean = float((tpr * tnr) ** 0.5)
    except:
        gmean = 0.0

    if detailed:
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Legal', 'Illegal'], zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
        print(f"Illicit class - Precision: {precision_illicit:.4f}, Recall: {recall_illicit:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
        if not np.isnan(auc):
            print(f"AUC: {auc:.4f}")
        print(f"G-Mean: {gmean:.4f}")

    return {
        'f1': f1, 'accuracy': acc,
        'precision_illicit': precision_illicit, 'recall_illicit': recall_illicit,
        'macro_recall': macro_recall, 'macro_f1': macro_f1,
        'auc': auc, 'macro_auc': macro_auc, 'gmean': gmean
    }


def plot_baseline_results(y_true, y_pred, model_name="IsolationForest"):
    """
    Plot baseline model results visualization.
    """
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name=model_name,
                         save_path=f"{model_name.lower()}_confusion.png")

    # Simple metrics output
    metrics = evaluate_baseline_model(y_true, y_pred, detailed=True)

    return metrics


def main():
    # Hard-coded parameters for automatic execution
    dataset_dir = '../Dataset'
    contamination = 'auto'  # Automatically calculate anomaly ratio
    n_estimators = 100
    max_samples = 'auto'
    random_state = 42
    output_dir = 'baseline_results'

    print("=== Blockchain Anomaly Detection Baseline Model (IsolationForest) ===")
    print(f"contamination: {contamination}")
    print(f"n_estimators: {n_estimators}")
    print(f"max_samples: {max_samples}")

    # Load data
    print("Loading data...")
    data = load_elliptic_data_for_baseline(dataset_dir)

    X_train = data['features'][data['train_mask']]
    y_train = data['labels'][data['train_mask']]
    X_val = data['features'][data['val_mask']]
    y_val = data['labels'][data['val_mask']]
    X_test = data['features'][data['test_mask']]
    y_test = data['labels'][data['test_mask']]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # If contamination is 'auto', set based on anomaly ratio in training set
    if contamination == 'auto':
        illicit_ratio = np.mean(y_train == 1)
        contamination = illicit_ratio
        print(f"Anomaly ratio calculated from training set: {contamination:.4f}")

    # Train model
    print("Training IsolationForest model...")
    model, scaler = train_isolation_forest(
        X_train,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples
    )

    # Make predictions
    print("Making predictions...")

    # Validation set
    y_val_pred, val_scores = predict_isolation_forest(model, scaler, X_val)
    print("\n=== Validation Set Results ===")
    val_metrics = evaluate_baseline_model(y_val, y_val_pred, val_scores, detailed=True)

    # Test set
    y_test_pred, test_scores = predict_isolation_forest(model, scaler, X_test)
    print("\n=== Test Set Results ===")
    test_metrics = evaluate_baseline_model(y_test, y_test_pred, test_scores, detailed=True)

    # Generate visualizations
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # Validation set confusion matrix
    plot_confusion_matrix(y_val, y_val_pred,
                         model_name="IsolationForest_Val",
                         save_path=os.path.join(output_dir, "isolation_forest_val_confusion.png"))

    # Test set confusion matrix
    plot_confusion_matrix(y_test, y_test_pred,
                         model_name="IsolationForest_Test",
                         save_path=os.path.join(output_dir, "isolation_forest_test_confusion.png"))

    # Save results summary
    results_summary = {
        'model': 'IsolationForest',
        'parameters': {
            'contamination': contamination,
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'random_state': random_state
        },
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'data_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_dim': X_train.shape[1]
        }
    }

    # Output final summary
    print("\n=== IsolationForest Baseline Model Summary ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"contamination: {contamination:.4f}")
    print("Validation set metrics:")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")
    print(f"  Illicit class Recall: {val_metrics['recall_illicit']:.4f}")
    print(f"  G-Mean: {val_metrics['gmean']:.4f}")
    print("Test set metrics:")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Illicit class Recall: {test_metrics['recall_illicit']:.4f}")
    print(f"  G-Mean: {test_metrics['gmean']:.4f}")

    print("IsolationForest training completed!")
    print(f"Results saved to: {output_dir}")

    return results_summary


if __name__ == '__main__':
    # Execute all functions automatically
    results = main()
    print(f"\nBaseline model execution completed successfully!")
