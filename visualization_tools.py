"""
Student ID: 24027277d
Name: Yuen Tsz Ki
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json

# PyTorch support removed - torch functionality not used in current implementation

# Set font for better display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingHistory:
    """Record training history"""
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_acc': [],
            'test_f1': [],
            'test_acc': []
        }

    def add_epoch(self, epoch, train_loss, val_loss, val_f1, val_acc, test_f1, test_acc,
                  val_macro_recall=0.0, test_macro_recall=0.0, val_gmean=0.0, test_gmean=0.0,
                  val_macro_f1=0.0, test_macro_f1=0.0, val_macro_auc=0.0, test_macro_auc=0.0):
        """Add a record for one epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_f1'].append(val_f1)
        self.history['val_acc'].append(val_acc)
        self.history['test_f1'].append(test_f1)
        self.history['test_acc'].append(test_acc)
        # Additional metrics
        if 'val_macro_recall' not in self.history:
            self.history['val_macro_recall'] = []
            self.history['test_macro_recall'] = []
            self.history['val_gmean'] = []
            self.history['test_gmean'] = []
            self.history['val_macro_f1'] = []
            self.history['test_macro_f1'] = []
            self.history['val_macro_auc'] = []
            self.history['test_macro_auc'] = []
        self.history['val_macro_recall'].append(val_macro_recall)
        self.history['test_macro_recall'].append(test_macro_recall)
        self.history['val_gmean'].append(val_gmean)
        self.history['test_gmean'].append(test_gmean)
        self.history['val_macro_f1'].append(val_macro_f1)
        self.history['test_macro_f1'].append(test_macro_f1)
        self.history['val_macro_auc'].append(val_macro_auc)
        self.history['test_macro_auc'].append(test_macro_auc)

    def save(self, filepath):
        """Save training history to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def load(self, filepath):
        """Load training history from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name
        save_path: Save path
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['licit', 'illicit'],
                yticklabels=['licit', 'illicit'],
                cbar_kws={'label': 'Count'})

    plt.title(f'{model_name} Confusion Matrix', fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {model_name.lower()}_confusion_matrix.png")

    plt.close()


def plot_individual_metrics(results_collector, save_dir='results/metrics'):
    """
    Generate individual bar charts for each metric comparing all models

    Args:
        results_collector: Dictionary containing results for all models
        save_dir: Directory to save the individual metric charts
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the models to compare - check if this is for MixHopConv models
    if 'MixHop_Final_Ensemble' in results_collector:
        # This is for MixHopConv models
        models = ['MixHop_Final_Ensemble', 'MixHop_GCN_Single', 'MixHop_GAT_Single', 'MixHop_GIN_Single', 'MixHop_GraphSAGE_Single', 'IForest']
        model_labels = ['MixHop Final Ensemble', 'MixHop GCN Single', 'MixHop GAT Single', 'MixHop GIN Single', 'MixHop GraphSAGE Single', 'Isolation Forest']
    else:
        # This is for regular GNN models
        models = ['Final_Ensemble', 'GCN_Single', 'GAT_Single', 'GIN_Single', 'GraphSAGE_Single', 'IForest']
        model_labels = ['Final Ensemble', 'GCN Single', 'GAT Single', 'GIN Single', 'GraphSAGE Single', 'Isolation Forest']

    # Filter out models that don't exist in results_collector
    available_models = []
    available_labels = []
    for model, label in zip(models, model_labels):
        if model in results_collector:
            available_models.append(model)
            available_labels.append(label)

    models = available_models
    model_labels = available_labels

    # Define metrics to plot (all 9 metrics from calculate_all_metrics)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'macro_recall', 'macro_f1', 'auc', 'macro_auc', 'gmean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Macro Recall', 'Macro F1', 'ROC-AUC', 'Macro AUC', 'G-Mean']

    # Color palette for 6 models
    colors = ['#2E86AB', '#A23B72', '#27AE60', '#E67E22', '#9B59B6', '#F24236']
    # Blue, Purple, Green, Orange, Magenta, Red, Yellow, Teal, Dark Blue, Bright Red

    # Create individual plots for each metric
    for metric, metric_label in zip(metrics, metric_labels):
        fig, ax = plt.subplots(figsize=(16, 8))

        # Collect values for this metric
        values = []
        for model in models:
            value = results_collector[model].get(metric, np.nan)
            values.append(value)

        # Create bars
        bars = ax.bar(model_labels, values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5, width=0.25)

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom',
                       fontsize=12, fontweight='bold')

        # Customize the plot
        ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric_label}', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max([v for v in values if not np.isnan(v)]) * 1.15 if any(not np.isnan(v) for v in values) else 1)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3, axis='y')

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        filename = f'{metric}_comparison.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual metric chart saved to: {save_path}")

        plt.close()

    print(f"\nAll individual metric charts saved to: {save_dir}")


def generate_mixhop_visualizations(results_collector, training_histories, test_y_true, test_y_pred):
    """
    Generate all visualizations for MixHopConv-enhanced GNN models

    Args:
        results_collector: Dictionary containing results for all models
        training_histories: Dictionary containing training histories for all models
        test_y_true: True labels for test set
        test_y_pred: Predicted labels for test set
    """
    print("\n8. Generating visualizations and training curves...")

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/curves', exist_ok=True)

    # Plot confusion matrix
    plot_confusion_matrix(test_y_true, test_y_pred, model_name='MixHopConv Final Ensemble Model',
                         save_path='results/mixhop_confusion_matrix.png')

    # Plot individual metric charts
    print("Generating individual metric comparison charts...")

    # Update the model list for MixHopConv models
    mixhop_results_collector = {}
    mixhop_keys = ['IForest', 'MixHop_GCN_Single', 'MixHop_GAT_Single', 'MixHop_GIN_Single', 'MixHop_GraphSAGE_Single', 'MixHop_Final_Ensemble']
    for key, value in results_collector.items():
        if key in mixhop_keys:
            mixhop_results_collector[key] = value

    plot_individual_metrics(mixhop_results_collector, save_dir='results/mixhop_metrics')

    # Plot training curves for all models (showing fitting process)
    print("Generating training curves for all models...")

    # Models to generate training curves for
    models_to_plot = [
        'MixHop_GCN_Single', 'MixHop_GAT_Single', 'MixHop_GIN_Single', 'MixHop_GraphSAGE_Single',
        'MixHop_GCN_Bagging', 'MixHop_GAT_Bagging', 'MixHop_GIN_Bagging', 'MixHop_GraphSAGE_Bagging'
    ]

    for model_key in models_to_plot:
        if model_key in training_histories and training_histories[model_key] is not None:
            history = training_histories[model_key]
            model_name = model_key.replace('MixHop_', '').replace('_', ' ')

            epochs = history.history['epoch']
            train_losses = history.history['train_loss']
            val_losses = history.history['val_loss']
            val_f1_scores = history.history['val_f1']
            test_f1_scores = history.history['test_f1']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{model_name} Training Process', fontsize=16, fontweight='bold')

            # Loss curves
            ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2, color='#e74c3c', alpha=0.8)
            ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, color='#3498db', alpha=0.8)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # F1 Score curves
            ax2.plot(epochs, val_f1_scores, label='Validation F1', linewidth=2, color='#f39c12', alpha=0.8)
            ax2.plot(epochs, test_f1_scores, label='Test F1', linewidth=2, color='#27ae60', alpha=0.8)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Validation and Test F1 Score', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'{model_key.lower()}_training_curves.png'
            plt.savefig(f'results/curves/{filename}', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Training curves saved to: results/curves/{filename}")
        else:
            print(f"Warning: No training history available for {model_key}")

    # Create a summary plot for the final ensemble (no training history, just final performance)
    print("Generating ensemble performance summary...")

    # Get final metrics for ensemble
    ensemble_metrics = results_collector.get('MixHop_Final_Ensemble', {})
    if ensemble_metrics:
        metrics_names = ['f1', 'accuracy', 'precision', 'recall', 'macro_f1', 'auc']
        metrics_values = [ensemble_metrics.get(m, 0) for m in metrics_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_names, metrics_values, color='#9b59b6', alpha=0.8, width=0.6)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('MixHopConv Final Ensemble Performance Summary', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/mixhop_final_ensemble_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Ensemble summary saved to: results/mixhop_final_ensemble_summary.png")
    else:
        print("Warning: No ensemble metrics available for summary")


def generate_standard_gnn_visualizations(results_collector, training_histories, test_y_true, test_y_pred):
    """
    Generate all visualizations for standard GNN models (GCN, GAT, GIN, GraphSAGE)

    Args:
        results_collector: Dictionary containing results for all models
        training_histories: Dictionary containing training histories for all models
        test_y_true: True labels for test set
        test_y_pred: Predicted labels for test set
    """
    print("\n8. Generating visualizations and training curves...")

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/curves', exist_ok=True)

    # Plot confusion matrix
    plot_confusion_matrix(test_y_true, test_y_pred, model_name='Final Ensemble Model',
                         save_path='results/confusion_matrix.png')

    # Plot individual metric charts
    print("Generating individual metric comparison charts...")
    plot_individual_metrics(results_collector, save_dir='results/metrics')

    # Plot training curves for all models (showing fitting process)
    print("Generating training curves for all models...")

    # Models to generate training curves for
    models_to_plot = [
        'GCN_Single', 'GAT_Single', 'GIN_Single', 'GraphSAGE_Single'
    ]

    for model_key in models_to_plot:
        if model_key in training_histories and training_histories[model_key] is not None:
            history = training_histories[model_key]
            model_name = model_key.replace('_', ' ')

            epochs = history.history['epoch']
            train_losses = history.history['train_loss']
            val_losses = history.history['val_loss']
            val_f1_scores = history.history['val_f1']
            test_f1_scores = history.history['test_f1']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{model_name} Training Process', fontsize=16, fontweight='bold')

            # Loss curves
            ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2, color='#e74c3c', alpha=0.8)
            ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, color='#3498db', alpha=0.8)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # F1 Score curves
            ax2.plot(epochs, val_f1_scores, label='Validation F1', linewidth=2, color='#f39c12', alpha=0.8)
            ax2.plot(epochs, test_f1_scores, label='Test F1', linewidth=2, color='#27ae60', alpha=0.8)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Validation and Test F1 Score', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'{model_key.lower()}_training_curves.png'
            plt.savefig(f'results/curves/{filename}', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Training curves saved to: results/curves/{filename}")
        else:
            print(f"Warning: No training history available for {model_key}")

    # Create a summary plot for the final ensemble (no training history, just final performance)
    print("Generating ensemble performance summary...")

    # Get final metrics for ensemble
    ensemble_metrics = results_collector.get('Final_Ensemble', {})
    if ensemble_metrics:
        metrics_names = ['f1', 'accuracy', 'precision', 'recall', 'macro_f1', 'auc']
        metrics_values = [ensemble_metrics.get(m, 0) for m in metrics_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_names, metrics_values, color='#9b59b6', alpha=0.8, width=0.6)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Final Ensemble Performance Summary', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/final_ensemble_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Ensemble summary saved to: results/final_ensemble_summary.png")
    else:
        print("Warning: No ensemble metrics available for summary")


# if __name__ == '__main__':
#     # All visualization functions are now used through the main pipeline files
#     # No standalone execution needed
#     pass