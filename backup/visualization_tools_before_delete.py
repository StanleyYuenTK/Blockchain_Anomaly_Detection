"""
Student ID: 24027277d
Name: Yuen Tsz Ki
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from collections import defaultdict
import os
import json
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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


def plot_model_architecture(model_name='GCN', save_path=None):
    """
    Plot model architecture diagram

    Args:
        model_name: Model name ('GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid')
        save_path: Save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4F8',
        'conv': '#FFE5B4',
        'attention': '#FFB6C1',
        'output': '#D4EDDA',
        'fusion': '#F0E68C'
    }
    
    if model_name in ['GCN', 'GAT', 'GraphSAGE']:
        # Basic GNN model architecture
        y_positions = [8, 6, 4, 2]
        
        # Input layer
        input_box = FancyBboxPatch((3.5, y_positions[0]-0.4), 3, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, y_positions[0], 'Input Features\n(Node Features)',
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Hidden layers
        for i, y in enumerate(y_positions[1:-1]):
            layer_name = f'{model_name}Conv\nLayer {i+1}'
            if model_name == 'GAT':
                layer_name = f'GATConv\n(Multi-head Attention)\nLayer {i+1}'
            elif model_name == 'GraphSAGE':
                layer_name = f'SAGEConv\n(Sample Aggregate)\nLayer {i+1}'

            conv_box = FancyBboxPatch((3.5, y-0.4), 3, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor=colors['conv'],
                                      edgecolor='black', linewidth=2)
            ax.add_patch(conv_box)
            ax.text(5, y, layer_name, ha='center', va='center', fontsize=10)

            # Arrow
            arrow = FancyArrowPatch((5, y_positions[i]-0.4), (5, y+0.4),
                                   arrowstyle='->', mutation_scale=20,
                                   color='black', linewidth=1.5)
            ax.add_patch(arrow)
        
        # Output layer
        output_box = FancyBboxPatch((3.5, y_positions[-1]-0.4), 3, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['output'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(5, y_positions[-1], 'Classification Output\n(Log Softmax)',
                ha='center', va='center', fontsize=12, weight='bold')

        arrow = FancyArrowPatch((5, y_positions[-2]-0.4), (5, y_positions[-1]+0.4),
                               arrowstyle='->', mutation_scale=20,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)

        ax.text(5, 9.5, f'{model_name} Model Architecture',
                ha='center', va='top', fontsize=16, weight='bold')
    
    elif model_name == 'SGAT':
        # SGAT model architecture
        ax.text(5, 9.5, 'SGAT Model Architecture (STA + GAT)',
                ha='center', va='top', fontsize=16, weight='bold')

        # Input
        input_box = FancyBboxPatch((3.5, 8-0.3), 3, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, 8, 'Input Features', ha='center', va='center', fontsize=11, weight='bold')

        # Branches
        # STA branch
        sta_box = FancyBboxPatch((1, 6-0.3), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['attention'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(sta_box)
        ax.text(2.25, 6, 'STA\n(Spatio-Temporal\nAttention)', ha='center', va='center', fontsize=9)

        # GAT branch
        gat_box = FancyBboxPatch((6.5, 6-0.3), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['conv'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(gat_box)
        ax.text(7.75, 6, 'GAT\n(Graph Attention)', ha='center', va='center', fontsize=9)
        
        # Arrows
        arrow1 = FancyArrowPatch((4, 8-0.3), (2.25, 6+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow1)

        arrow2 = FancyArrowPatch((6, 8-0.3), (7.75, 6+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow2)

        # Fusion layer
        fusion_box = FancyBboxPatch((3.5, 4-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(fusion_box)
        ax.text(5, 4, 'Feature Fusion Layer', ha='center', va='center', fontsize=11, weight='bold')

        arrow3 = FancyArrowPatch((2.25, 6-0.3), (4.5, 4+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow3)

        arrow4 = FancyArrowPatch((7.75, 6-0.3), (5.5, 4+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow4)

        # Output
        output_box = FancyBboxPatch((3.5, 2-0.3), 3, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['output'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(5, 2, 'Classification Output', ha='center', va='center', fontsize=11, weight='bold')
        
        arrow5 = FancyArrowPatch((5, 4-0.3), (5, 2+0.3),
                                 arrowstyle='->', mutation_scale=15,
                                 color='black', linewidth=1.5)
        ax.add_patch(arrow5)
    
    elif model_name == 'Hybrid':
        # Hybrid model architecture
        ax.text(5, 9.5, 'Hybrid Model Architecture (SGAT + GraphSAGE + Temporal Features)',
                ha='center', va='top', fontsize=14, weight='bold')

        # Input
        input_box = FancyBboxPatch((3.5, 8.5-0.25), 3, 0.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, 8.5, 'Input Features', ha='center', va='center', fontsize=10, weight='bold')

        # Three branches
        # GraphSAGE
        sage_box = FancyBboxPatch((0.5, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['conv'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(sage_box)
        ax.text(1.5, 6.5, 'GraphSAGE\nEncoder', ha='center', va='center', fontsize=8)

        # Temporal features
        temp_box = FancyBboxPatch((4, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['attention'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(temp_box)
        ax.text(5, 6.5, 'Temporal Feature\nExtractor\n(GRU+Conv1D)', ha='center', va='center', fontsize=8)

        # Original features
        orig_box = FancyBboxPatch((7.5, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['input'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(orig_box)
        ax.text(8.5, 6.5, 'Original Features', ha='center', va='center', fontsize=8)
        
        # Arrows
        for x in [1.5, 5, 8.5]:
            arrow = FancyArrowPatch((x, 8.5-0.25), (x, 6.5+0.25),
                                   arrowstyle='->', mutation_scale=12,
                                   color='black', linewidth=1.2)
            ax.add_patch(arrow)

        # Merge
        merge_box = FancyBboxPatch((3, 5-0.25), 4, 0.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(merge_box)
        ax.text(5, 5, 'Feature Merge', ha='center', va='center', fontsize=10, weight='bold')

        for x in [1.5, 5, 8.5]:
            arrow = FancyArrowPatch((x, 6.5-0.25), (5, 5+0.25),
                                   arrowstyle='->', mutation_scale=12,
                                   color='black', linewidth=1.2)
            ax.add_patch(arrow)

        # SGAT
        sgat_box = FancyBboxPatch((3.5, 3.5-0.25), 3, 0.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['attention'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(sgat_box)
        ax.text(5, 3.5, 'SGAT Processing', ha='center', va='center', fontsize=10, weight='bold')

        arrow = FancyArrowPatch((5, 5-0.25), (5, 3.5+0.25),
                               arrowstyle='->', mutation_scale=12,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)

        # Classifier
        classifier_box = FancyBboxPatch((3.5, 2-0.25), 3, 0.5,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colors['output'],
                                       edgecolor='black', linewidth=2)
        ax.add_patch(classifier_box)
        ax.text(5, 2, 'Classifier', ha='center', va='center', fontsize=10, weight='bold')

        arrow = FancyArrowPatch((5, 3.5-0.25), (5, 2+0.25),
                               arrowstyle='->', mutation_scale=12,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture diagram saved to: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_architecture.png', dpi=300, bbox_inches='tight')
        print(f"Model architecture diagram saved to: {model_name.lower()}_architecture.png")
    
    plt.close()


def plot_training_curves(history, save_path=None, model_name='Model'):
    """
    Plot training curves

    Args:
        history: TrainingHistory object or dictionary
        save_path: Save path
        model_name: Model name
    """
    if isinstance(history, TrainingHistory):
        history = history.history
    
    epochs = history['epoch']
    
    # Changed to 3x2 to show additional metrics (Macro Recall, G-Mean)
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(f'{model_name} Training Process', fontsize=16, weight='bold')

    # Loss curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, color='#3498db')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve', fontsize=13, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # F1 score curve
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_f1'], label='Validation F1', linewidth=2, color='#2ecc71')
    ax2.plot(epochs, history['test_f1'], label='Test F1', linewidth=2, color='#9b59b6')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Curve', fontsize=13, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Accuracy curve
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, color='#f39c12')
    ax3.plot(epochs, history['test_acc'], label='Test Accuracy', linewidth=2, color='#1abc9c')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy Curve', fontsize=13, weight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Comprehensive performance - F1 vs Acc
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, history['val_f1'], label='Validation F1', linewidth=2, color='#2ecc71')
    line2 = ax4_twin.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, color='#f39c12', linestyle='--')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1 Score', fontsize=12, color='#2ecc71')
    ax4_twin.set_ylabel('Accuracy', fontsize=12, color='#f39c12')
    ax4.set_title('Comprehensive Performance Metrics', fontsize=13, weight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Third row: Macro Recall and G-Mean
    ax5 = axes[2, 0]
    val_macro = history.get('val_macro_recall', [0]*len(epochs))
    test_macro = history.get('test_macro_recall', [0]*len(epochs))
    ax5.plot(epochs, val_macro, label='Validation Macro Recall', linewidth=2, color='#8e44ad')
    ax5.plot(epochs, test_macro, label='Test Macro Recall', linewidth=2, color='#3498db', linestyle='--')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Macro Recall', fontsize=12)
    ax5.set_title('Macro Recall Curve', fontsize=13, weight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    val_g = history.get('val_gmean', [0]*len(epochs))
    test_g = history.get('test_gmean', [0]*len(epochs))
    ax6.plot(epochs, val_g, label='Validation G-Mean', linewidth=2, color='#e67e22')
    ax6.plot(epochs, test_g, label='Test G-Mean', linewidth=2, color='#2c3e50', linestyle='--')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('G-Mean', fontsize=12)
    ax6.set_title('G-Mean Curve', fontsize=13, weight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Fourth row: Macro F1 and Macro AUC
    ax7 = axes[3, 0]
    val_mf1 = history.get('val_macro_f1', [0]*len(epochs))
    test_mf1 = history.get('test_macro_f1', [0]*len(epochs))
    ax7.plot(epochs, val_mf1, label='Validation Macro F1', linewidth=2, color='#16a085')
    ax7.plot(epochs, test_mf1, label='Test Macro F1', linewidth=2, color='#e74c3c', linestyle='--')
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Macro F1', fontsize=12)
    ax7.set_title('Macro F1 Curve', fontsize=13, weight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    ax8 = axes[3, 1]
    val_mauc = history.get('val_macro_auc', [0]*len(epochs))
    test_mauc = history.get('test_macro_auc', [0]*len(epochs))
    ax8.plot(epochs, val_mauc, label='Validation Macro AUC', linewidth=2, color='#2c3e50')
    ax8.plot(epochs, test_mauc, label='Test Macro AUC', linewidth=2, color='#9b59b6', linestyle='--')
    ax8.set_xlabel('Epoch', fontsize=12)
    ax8.set_ylabel('Macro AUC', fontsize=12)
    ax8.set_title('Macro AUC Curve', fontsize=13, weight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {model_name.lower()}_training_curves.png")
    
    plt.close()


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot model performance comparison chart

    Args:
        results_dict: Dictionary in format {model_name: {metric: value}}
        save_path: Save path
    """
    models = list(results_dict.keys())
    metrics = ['val_f1', 'test_f1', 'val_acc', 'test_acc']
    metric_names = ['Validation F1', 'Test F1', 'Validation Accuracy', 'Test Accuracy']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, weight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        values = [results_dict[model].get(metric, 0) for model in models]
        colors = sns.color_palette("husl", len(models))

        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, weight='bold')

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=13, weight='bold')
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(models, rotation=15, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison chart saved to: {save_path}")
    else:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Model comparison chart saved to: model_comparison.png")
    
    plt.close()


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
                xticklabels=['Legal', 'Illegal'],
                yticklabels=['Legal', 'Illegal'],
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


def plot_feature_visualization_2d(features_2d, labels, title="Feature Visualization",
                                   method="PCA", save_path=None):
    """
    Plot 2D feature visualization

    Args:
        features_2d: 2D feature matrix [num_nodes, 2]
        labels: Node labels
        title: Chart title
        method: Dimensionality reduction method ('PCA' or 't-SNE')
        save_path: Save path
    """
    if HAS_TORCH:
        if isinstance(features_2d, torch.Tensor):
            features_2d = features_2d.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    # Only show nodes with known labels
    known_mask = labels != -1
    known_features = features_2d[known_mask]
    known_labels = labels[known_mask]

    # Plot points for different classes
    for label in [0, 1]:
        mask = known_labels == label
        label_name = "Legal" if label == 0 else "Illegal"
        color = '#3498db' if label == 0 else '#e74c3c'
        plt.scatter(known_features[mask, 0], known_features[mask, 1],
                   label=label_name, alpha=0.6, s=20, color=color, edgecolors='black', linewidths=0.5)

    xlabel = "First Principal Component" if method == "PCA" else "t-SNE Dimension 1"
    ylabel = "Second Principal Component" if method == "PCA" else "t-SNE Dimension 2"

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} ({method})', fontsize=14, weight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to: {save_path}")
    else:
        filename = f'feature_visualization_{method.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to: {filename}")
    
    plt.close()


def plot_architecture_comparison(save_path=None):
    """
    Plot architecture comparison for all models
    """
    models = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    
    fig, axes = plt.subplots(1, len(models), figsize=(20, 4))
    fig.suptitle('GNN Model Architecture Comparison', fontsize=16, weight='bold')

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Simplified architecture representation
        y_positions = [0.8, 0.6, 0.4, 0.2]

        # Input
        rect = mpatches.FancyBboxPatch((0.2, y_positions[0]-0.05), 0.6, 0.1,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#E8F4F8', edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.5, y_positions[0], 'Input', ha='center', va='center', fontsize=8)

        # Middle layers
        for i, y in enumerate(y_positions[1:-1]):
            if model_name == 'GCN':
                layer_name = 'GCN'
            elif model_name == 'GAT':
                layer_name = 'GAT'
            elif model_name == 'GraphSAGE':
                layer_name = 'SAGE'
            elif model_name == 'SGAT':
                layer_name = 'SGAT'
            elif model_name == 'Hybrid':
                layer_name = 'Hybrid'

            rect = mpatches.FancyBboxPatch((0.2, y-0.05), 0.6, 0.1,
                                           boxstyle="round,pad=0.02",
                                           facecolor='#FFE5B4', edgecolor='black')
            ax.add_patch(rect)
            ax.text(0.5, y, layer_name, ha='center', va='center', fontsize=8)

            # Arrow
            arrow = FancyArrowPatch((0.5, y_positions[i]-0.05), (0.5, y+0.05),
                                   arrowstyle='->', mutation_scale=10,
                                   color='black', linewidth=1)
            ax.add_patch(arrow)

        # Output
        rect = mpatches.FancyBboxPatch((0.2, y_positions[-1]-0.05), 0.6, 0.1,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#D4EDDA', edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.5, y_positions[-1], 'Output', ha='center', va='center', fontsize=8)

        arrow = FancyArrowPatch((0.5, y_positions[-2]-0.05), (0.5, y_positions[-1]+0.05),
                               arrowstyle='->', mutation_scale=10,
                               color='black', linewidth=1)
        ax.add_patch(arrow)

        ax.text(0.5, 0.95, model_name, ha='center', va='top',
               fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture comparison saved to: {save_path}")
    else:
        plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Architecture comparison saved to: architecture_comparison.png")
    
    plt.close()


def generate_all_visualizations(output_dir='visualizations'):
    """
    Generate all visualization charts

    Args:
        output_dir: Output directory
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Starting to generate visualization charts...")
    print("="*60)

    # 1. Generate architecture diagrams for all models
    print("\n1. Generating model architecture diagrams...")
    models = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    for model in models:
        plot_model_architecture(model,
                               os.path.join(output_dir, f'{model.lower()}_architecture.png'))

    # 2. Generate architecture comparison
    print("\n2. Generating architecture comparison...")
    plot_architecture_comparison(os.path.join(output_dir, 'architecture_comparison.png'))

    # 3. Generate example training curves (using simulated data)
    print("\n3. Generating example training curves...")
    example_history = TrainingHistory()
    for epoch in range(1, 101):
        train_loss = 0.5 * np.exp(-epoch/50) + 0.1 + np.random.normal(0, 0.02)
        val_loss = 0.6 * np.exp(-epoch/50) + 0.15 + np.random.normal(0, 0.02)
        val_f1 = 0.3 + 0.5 * (1 - np.exp(-epoch/30)) + np.random.normal(0, 0.02)
        val_acc = 0.4 + 0.4 * (1 - np.exp(-epoch/30)) + np.random.normal(0, 0.02)
        test_f1 = val_f1 - 0.05 + np.random.normal(0, 0.02)
        test_acc = val_acc - 0.05 + np.random.normal(0, 0.02)

        example_history.add_epoch(epoch, train_loss, val_loss,
                                 max(0, min(1, val_f1)), max(0, min(1, val_acc)),
                                 max(0, min(1, test_f1)), max(0, min(1, test_acc)))

    for model in models[:3]:  # Only generate examples for first three models
        plot_training_curves(example_history,
                            os.path.join(output_dir, f'{model.lower()}_training_curves.png'),
                            model)

    # 4. Generate example model comparison chart
    print("\n4. Generating model performance comparison chart...")
    example_results = {
        'GCN': {'val_f1': 0.75, 'test_f1': 0.72, 'val_acc': 0.78, 'test_acc': 0.75},
        'GAT': {'val_f1': 0.78, 'test_f1': 0.75, 'val_acc': 0.81, 'test_acc': 0.78},
        'GraphSAGE': {'val_f1': 0.76, 'test_f1': 0.73, 'val_acc': 0.79, 'test_acc': 0.76},
        'SGAT': {'val_f1': 0.82, 'test_f1': 0.79, 'val_acc': 0.85, 'test_acc': 0.82},
        'Hybrid': {'val_f1': 0.85, 'test_f1': 0.82, 'val_acc': 0.88, 'test_acc': 0.85}
    }
    plot_model_comparison(example_results,
                         os.path.join(output_dir, 'model_comparison.png'))

    print("\n" + "="*60)
    print(f"All visualization charts generated and saved to '{output_dir}' directory")
    print("="*60)


# ============================================================================
# Common data visualization charts
# ============================================================================

def plot_bar_chart(data, x_labels, title="Bar Chart", xlabel="Category", ylabel="Value",
                   colors=None, save_path=None, figsize=(10, 6), rotation=0):
    """
    Plot bar chart

    Args:
        data: Data values list or dictionary
        x_labels: X-axis label list
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Color list (optional)
        save_path: Save path
        figsize: Chart size
        rotation: X-axis label rotation angle
    """
    plt.figure(figsize=figsize)

    if isinstance(data, dict):
        x_labels = list(data.keys())
        values = list(data.values())
    else:
        values = data

    if colors is None:
        colors = sns.color_palette("husl", len(values))

    bars = plt.bar(x_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}' if isinstance(value, (int, float)) else str(value),
                ha='center', va='bottom', fontsize=10, weight='bold')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.xticks(rotation=rotation, ha='right' if rotation != 0 else 'center')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: {save_path}")
    else:
        plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: bar_chart.png")
    
    plt.close()


def plot_pie_chart(data, labels, title="Pie Chart", colors=None, save_path=None,
                   figsize=(10, 8), autopct='%1.1f%%', startangle=90):
    """
    Plot pie chart

    Args:
        data: Data values list or dictionary
        labels: Label list
        title: Chart title
        colors: Color list (optional)
        save_path: Save path
        figsize: Chart size
        autopct: Percentage format
        startangle: Starting angle
    """
    plt.figure(figsize=figsize)

    if isinstance(data, dict):
        labels = list(data.keys())
        values = list(data.values())
    else:
        values = data

    if colors is None:
        colors = sns.color_palette("husl", len(values))

    # Highlight the largest sector
    explode = [0.05 if i == np.argmax(values) else 0 for i in range(len(values))]

    wedges, texts, autotexts = plt.pie(values, labels=labels, colors=colors,
                                       autopct=autopct, startangle=startangle,
                                       explode=explode, shadow=True,
                                       textprops={'fontsize': 11, 'weight': 'bold'})

    # Beautify auto texts
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {save_path}")
    else:
        plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: pie_chart.png")
    
    plt.close()


def plot_scatter_distribution(x, y, labels=None, title="Scatter Data Distribution",
                             xlabel="X-axis", ylabel="Y-axis", save_path=None,
                             figsize=(10, 8), alpha=0.6, size=50, color_col=None):
    """
    Plot scatter data distribution

    Args:
        x: X-axis data
        y: Y-axis data
        labels: Labels (for different colors, optional)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Save path
        figsize: Chart size
        alpha: Transparency
        size: Point size
        color_col: Column for color (if data is DataFrame)
    """
    if HAS_TORCH:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    if labels is not None:
        # Use different colors based on labels
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f'Category {label}' if isinstance(label, (int, float)) else str(label)
            plt.scatter(x[mask], y[mask], label=label_name, alpha=alpha,
                       s=size, color=colors[i], edgecolors='black', linewidths=0.5)
        plt.legend(fontsize=10, loc='best')
    else:
        plt.scatter(x, y, alpha=alpha, s=size, c='steelblue',
                   edgecolors='black', linewidths=0.5)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    else:
        plt.savefig('scatter_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: scatter_distribution.png")
    
    plt.close()


def plot_histogram(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency",
                  save_path=None, figsize=(10, 6), color='steelblue', alpha=0.7):
    """
    Plot histogram

    Args:
        data: Data list or array
        bins: Number of bins
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Save path
        figsize: Chart size
        color: Color
        alpha: Transparency
    """
    if HAS_TORCH:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

    plt.figure(figsize=figsize)

    n, bins, patches = plt.hist(data, bins=bins, color=color, alpha=alpha,
                                edgecolor='black', linewidth=1.2)

    # Add mean line
    mean_val = np.mean(data)
    std_val = np.std(data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'±1 Std Dev')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    else:
        plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: histogram.png")
    
    plt.close()


def plot_boxplot(data, labels=None, title="Box Plot", ylabel="Value",
                save_path=None, figsize=(10, 6), vert=True):
    """
    Plot box plot

    Args:
        data: Data list (multiple groups) or single group data
        labels: Label list
        title: Chart title
        ylabel: Y-axis label
        save_path: Save path
        figsize: Chart size
        vert: Whether to display vertically
    """
    if HAS_TORCH:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], torch.Tensor):
            data = [d.cpu().numpy() for d in data]

    plt.figure(figsize=figsize)

    bp = plt.boxplot(data, labels=labels, vert=vert, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5))

    if vert:
        plt.ylabel(ylabel, fontsize=12)
    else:
        plt.xlabel(ylabel, fontsize=12)

    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3, axis='y' if vert else 'x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: {save_path}")
    else:
        plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: boxplot.png")
    
    plt.close()


def plot_heatmap(data, row_labels=None, col_labels=None, title="Heatmap",
                save_path=None, figsize=(10, 8), cmap='viridis', annot=True,
                fmt='.2f', cbar_label="Value"):
    """
    Plot heatmap

    Args:
        data: 2D data array or matrix
        row_labels: Row labels
        col_labels: Column labels
        title: Chart title
        save_path: Save path
        figsize: Chart size
        cmap: Color map
        annot: Whether to show value annotations
        fmt: Value format
        cbar_label: Color bar label
    """
    if HAS_TORCH:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

    plt.figure(figsize=figsize)

    sns.heatmap(data, xticklabels=col_labels, yticklabels=row_labels,
               cmap=cmap, annot=annot, fmt=fmt, cbar_kws={'label': cbar_label},
               linewidths=0.5, linecolor='gray')

    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: heatmap.png")
    
    plt.close()


def plot_line_chart(x, y, labels=None, title="Line Chart", xlabel="X-axis", ylabel="Y-axis",
                   save_path=None, figsize=(10, 6), marker='o', linestyle='-'):
    """
    Plot line chart

    Args:
        x: X-axis data (single line) or X-axis data list (multiple lines)
        y: Y-axis data (single line) or Y-axis data list (multiple lines)
        labels: Label list (used for multiple lines)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Save path
        figsize: Chart size
        marker: Marker style
        linestyle: Line style
    """
    if HAS_TORCH:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

    plt.figure(figsize=figsize)

    # Check if multiple lines
    if isinstance(y, list) or (isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[0] > 1):
        # Multiple lines
        if labels is None:
            labels = [f'Line {i+1}' for i in range(len(y))]

        colors = sns.color_palette("husl", len(y))
        for i, (yi, label) in enumerate(zip(y, labels)):
            if isinstance(x, list):
                xi = x[i]
            else:
                xi = x
            plt.plot(xi, yi, marker=marker, linestyle=linestyle,
                    label=label, linewidth=2, markersize=6, color=colors[i])
        plt.legend(fontsize=10, loc='best')
    else:
        # Single line
        plt.plot(x, y, marker=marker, linestyle=linestyle,
                linewidth=2, markersize=6, color='steelblue')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Line chart saved to: {save_path}")
    else:
        plt.savefig('line_chart.png', dpi=300, bbox_inches='tight')
        print(f"Line chart saved to: line_chart.png")
    
    plt.close()


def plot_class_distribution(labels, title="Class Distribution", save_path=None,
                           figsize=(10, 6), normalize=False):
    """
    Plot class distribution chart (bar chart + pie chart)

    Args:
        labels: Label array
        title: Chart title
        save_path: Save path
        figsize: Chart size
        normalize: Whether to normalize to percentage
    """
    if HAS_TORCH:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    if normalize:
        counts = counts / len(labels) * 100

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    colors = sns.color_palette("husl", len(unique))
    bars = ax1.bar(unique.astype(str), counts, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:.2f}{"%" if normalize else ""}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Percentage' if normalize else 'Count', fontsize=12)
    ax1.set_title(f'{title} - Bar Chart', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Pie chart
    explode = [0.05 if i == np.argmax(counts) else 0 for i in range(len(unique))]
    wedges, texts, autotexts = ax2.pie(counts, labels=unique.astype(str),
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=explode, shadow=True,
                                       textprops={'fontsize': 11, 'weight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax2.set_title(f'{title} - Pie Chart', fontsize=13, weight='bold')
    ax2.axis('equal')

    plt.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution chart saved to: {save_path}")
    else:
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Class distribution chart saved to: class_distribution.png")
    
    plt.close()


def generate_common_visualizations(output_dir='visualizations/common'):
    """
    Generate examples of common visualization charts

    Args:
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating examples of common visualization charts...")
    print("="*60)

    # 1. Bar chart example
    print("\n1. Generating bar chart...")
    model_scores = {
        'GCN': 0.75,
        'GAT': 0.78,
        'GraphSAGE': 0.76,
        'SGAT': 0.82,
        'Hybrid': 0.85
    }
    plot_bar_chart(model_scores, None, "Model Performance Comparison (F1 Score)",
                  "Model", "F1 Score",
                  save_path=os.path.join(output_dir, 'bar_chart_example.png'))

    # 2. Pie chart example
    print("\n2. Generating pie chart...")
    class_dist = {
        'Legal Transaction': 70,
        'Illegal Transaction': 25,
        'Unknown': 5
    }
    plot_pie_chart(class_dist, None, "Transaction Category Distribution",
                  save_path=os.path.join(output_dir, 'pie_chart_example.png'))

    # 3. Scatter plot example
    print("\n3. Generating scatter plot...")
    np.random.seed(42)
    n_samples = 500
    x1 = np.random.normal(0, 1, n_samples)
    y1 = np.random.normal(0, 1, n_samples)
    labels1 = np.zeros(n_samples)

    x2 = np.random.normal(3, 1, n_samples)
    y2 = np.random.normal(3, 1, n_samples)
    labels2 = np.ones(n_samples)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    labels = np.concatenate([labels1, labels2])

    plot_scatter_distribution(x, y, labels, "Data Distribution Scatter Plot",
                             "Feature 1", "Feature 2",
                             save_path=os.path.join(output_dir, 'scatter_distribution_example.png'))

    # 4. Histogram example
    print("\n4. Generating histogram...")
    data_hist = np.random.normal(100, 15, 1000)
    plot_histogram(data_hist, bins=30, title="Transaction Amount Distribution Histogram",
                  xlabel="Amount", ylabel="Frequency",
                  save_path=os.path.join(output_dir, 'histogram_example.png'))

    # 5. Box plot example
    print("\n5. Generating box plot...")
    data_box = [
        np.random.normal(100, 10, 100),
        np.random.normal(150, 15, 100),
        np.random.normal(120, 12, 100)
    ]
    plot_boxplot(data_box, ['Legal', 'Illegal', 'Unknown'],
                "Transaction Amount Distribution by Category Box Plot", "Amount",
                save_path=os.path.join(output_dir, 'boxplot_example.png'))

    # 6. Heatmap example
    print("\n6. Generating heatmap...")
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    plot_heatmap(correlation_matrix,
                row_labels=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
                col_labels=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
                title="Feature Correlation Heatmap",
                save_path=os.path.join(output_dir, 'heatmap_example.png'),
                cmap='coolwarm', cbar_label="Correlation")

    # 7. Line chart example
    print("\n7. Generating line chart...")
    epochs = np.arange(1, 101)
    train_loss = 0.5 * np.exp(-epochs/50) + 0.1 + np.random.normal(0, 0.02, 100)
    val_loss = 0.6 * np.exp(-epochs/50) + 0.15 + np.random.normal(0, 0.02, 100)

    plot_line_chart(epochs, [train_loss, val_loss], ['Training Loss', 'Validation Loss'],
                   "Training Loss Curve", "Epoch", "Loss",
                   save_path=os.path.join(output_dir, 'line_chart_example.png'))

    # 8. Class distribution chart example
    print("\n8. Generating class distribution chart...")
    labels_dist = np.random.choice([0, 1, -1], size=1000, p=[0.7, 0.25, 0.05])
    plot_class_distribution(labels_dist, "Transaction Category Distribution",
                           save_path=os.path.join(output_dir, 'class_distribution_example.png'),
                           normalize=True)

    print("\n" + "="*60)
    print(f"All common visualization chart examples generated and saved to '{output_dir}' directory")
    print("="*60)


def run_full_report(model, data, history, embeds=None, output_dir='visualizations/full_report', reduction_method='tsne'):
    """
    Generate complete report: model architecture, training curves, confusion matrix, embedding visualization, temporal dynamics, cumulative gains, etc.

    Args:
        model: Trained model object (PyTorch)
        data: PyG Data (must contain train/val/test masks and timesteps)
        history: TrainingHistory or equivalent dict
        embeds: If None, will try to extract from model; if high-dimensional embedding, use dimensionality reduction (TSNE/PCA)
        output_dir: Chart output folder
        reduction_method: 'tsne' or 'pca' for embedding dimensionality reduction
    """
    import os
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    os.makedirs(output_dir, exist_ok=True)

    model_name = type(model).__name__ if model is not None else "Model"

    # 1) Model architecture diagram
    try:
        plot_model_architecture(model_name, save_path=os.path.join(output_dir, f"{model_name.lower()}_architecture.png"))
    except Exception as e:
        print(f"[run_full_report] Unable to plot model architecture: {e}")

    # 2) Training curves
    try:
        plot_training_curves(history, save_path=os.path.join(output_dir, f"{model_name.lower()}_training_curves.png"), model_name=model_name)
    except Exception as e:
        print(f"[run_full_report] Unable to plot training curves: {e}")

    # 3) Confusion matrix (validation / test)
    try:
        if _torch is not None:
            model.eval()
            with _torch.no_grad():
                out = model(data)
                preds = out.argmax(dim=1).cpu().numpy()
        else:
            preds = None
        if hasattr(data, 'val_mask') and preds is not None:
            val_mask = data.val_mask.cpu().numpy()
            plot_confusion_matrix(data.y.cpu().numpy()[val_mask], preds[val_mask], model_name=f"{model_name}_val", save_path=os.path.join(output_dir, f"{model_name.lower()}_val_confusion.png"))
        if hasattr(data, 'test_mask') and preds is not None:
            test_mask = data.test_mask.cpu().numpy()
            plot_confusion_matrix(data.y.cpu().numpy()[test_mask], preds[test_mask], model_name=f"{model_name}_test", save_path=os.path.join(output_dir, f"{model_name.lower()}_test_confusion.png"))
    except Exception as e:
        print(f"[run_full_report] Unable to generate confusion matrix: {e}")

    # 4) Embedding visualization (if embeds not provided, try to extract from model)
    try:
        features = None
        if embeds is not None:
            features = embeds
        else:
            try:
                out_emb = model(data, return_embed=True)
                if isinstance(out_emb, tuple):
                    _, features = out_emb
                else:
                    features = out_emb
            except Exception:
                features = None

        if features is not None:
            # features may be torch tensor or numpy
            if _torch is not None and isinstance(features, _torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = _np.array(features)

            if features_np.ndim == 1:
                features_np = features_np.reshape(-1, 1)

            if features_np.shape[1] > 2:
                if reduction_method == 'pca':
                    reducer = _PCA(n_components=2, random_state=42)
                    reduced = reducer.fit_transform(features_np)
                else:
                    reducer = _TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
                    reduced = reducer.fit_transform(features_np)
            else:
                reduced = features_np

            plot_feature_visualization_2d(reduced, data.y, title=f"{model_name} Embedding Visualization", method=reduction_method.upper(), save_path=os.path.join(output_dir, f"{model_name.lower()}_embedding_{reduction_method}.png"))
    except Exception as e:
        print(f"[run_full_report] Unable to generate embedding visualization: {e}")

    # 5) Temporal dynamics: illicit ratio / F1 for each timestep (if timesteps exist)
    try:
        if hasattr(data, 'timesteps'):
            t = data.timesteps.cpu().numpy()
            y = data.y.cpu().numpy()
            unique_ts = _np.sort(_np.unique(t))
            ts_illicit_ratio = []
            ts_f1 = []
            for ts in unique_ts:
                mask = (t == ts) & (y != -1)
                if mask.sum() == 0:
                    ts_illicit_ratio.append(0.0)
                    ts_f1.append(0.0)
                    continue
                y_ts = y[mask]
                try:
                    if _torch is not None:
                        with _torch.no_grad():
                            out = model(data)
                            preds_ts = out.argmax(dim=1).cpu().numpy()[mask]
                    else:
                        preds_ts = _np.zeros_like(y_ts)
                    from sklearn.metrics import f1_score as _f1
                    ts_f1.append(float(_f1(y_ts, preds_ts, average='binary', pos_label=1, zero_division=0)))
                except Exception:
                    ts_f1.append(0.0)
                ts_illicit_ratio.append(float((y_ts == 1).sum() / max(1, len(y_ts))))

            # Plot
            try:
                plt.figure(figsize=(10,4))
                plt.plot(unique_ts, ts_illicit_ratio, marker='o', label='Illicit Ratio', color='red')
                plt.xlabel('Timestep'); plt.ylabel('Illicit Ratio'); plt.title(f'{model_name} - Temporal Illicit Ratio'); plt.grid(True); plt.legend()
                plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_temporal_illicit_ratio.png'), dpi=300); plt.close()
                plt.figure(figsize=(10,4))
                plt.plot(unique_ts, ts_f1, marker='o', label='F1 Score', color='blue')
                plt.xlabel('Timestep'); plt.ylabel('F1'); plt.title(f'{model_name} - Temporal F1 Score'); plt.grid(True); plt.legend()
                plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_temporal_f1.png'), dpi=300); plt.close()
            except Exception as e:
                print(f"[run_full_report] Unable to plot temporal dynamics: {e}")
    except Exception as e:
        print(f"[run_full_report] Temporal dynamics processing failed: {e}")

    # 6) Cumulative Gain
    try:
        if _torch is not None:
            with _torch.no_grad():
                out = model(data)
                probs = _torch.exp(out)[:, 1].cpu().numpy()
            y_true = data.y.cpu().numpy()
            mask_known = (y_true != -1)
            scores = probs[mask_known]
            labels_known = y_true[mask_known]
            order = _np.argsort(-scores)
            y_sorted = labels_known[order]
            cum_tp = _np.cumsum(y_sorted == 1)
            total_pos = max(1, (labels_known == 1).sum())
            perc_accounts = _np.arange(1, len(y_sorted)+1) / len(y_sorted) * 100
            gain = cum_tp / total_pos
            plt.figure(figsize=(8,6))
            plt.plot(perc_accounts, gain, label='Cumulative Gain', color='purple')
            plt.xlabel('Top X% Accounts'); plt.ylabel('Cumulative Illicit Ratio Caught'); plt.title(f'{model_name} - Cumulative Gain Chart'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_cumulative_gain.png'), dpi=300); plt.close()
    except Exception as e:
        print(f"[run_full_report] Cumulative gain chart generation failed: {e}")

    print(f"[run_full_report] Complete, report saved to: {output_dir}")


def plot_model_comparison(ensemble_results, gcn_results, iforest_results, save_path=None):
    """
    Plot model performance comparison using grouped bar chart

    Args:
        ensemble_results: Dict containing ensemble model metrics
        gcn_results: Dict containing GCN baseline metrics
        iforest_results: Dict containing Isolation Forest baseline metrics
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define metrics to plot
    metrics = ['macro_f1', 'macro_recall', 'macro_auc', 'gmean']
    metric_labels = ['Macro F1', 'Macro Recall', 'Macro AUC', 'G-Mean']

    # Prepare data for plotting
    ensemble_values = []
    gcn_values = []
    iforest_values = []

    for metric in metrics:
        ensemble_val = ensemble_results.get(metric, np.nan)
        gcn_val = gcn_results.get(metric, np.nan)
        iforest_val = iforest_results.get(metric, np.nan)

        ensemble_values.append(ensemble_val)
        gcn_values.append(gcn_val)
        iforest_values.append(iforest_val)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set positions for bars
    x = np.arange(len(metric_labels))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, ensemble_values, width, label='Final Ensemble',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, gcn_values, width, label='GCN Baseline',
                   color='#F24236', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, iforest_values, width, label='Isolation Forest',
                   color='#F5A623', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Customize the plot
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Final Ensemble vs GCN Baseline vs Isolation Forest',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")

    plt.show()


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

    # Define the models to compare
    models = ['Final_Ensemble', 'GCN_Bagging', 'GAT_Bagging', 'GIN_Bagging', 'GraphSAGE_Bagging',
              'GCN_Single', 'GAT_Single', 'GIN_Single', 'GraphSAGE_Single', 'IForest']
    model_labels = ['Final Ensemble', 'GCN Bagging', 'GAT Bagging', 'GIN Bagging', 'GraphSAGE Bagging',
                    'GCN Single', 'GAT Single', 'GIN Single', 'GraphSAGE Single', 'Isolation Forest']

    # Define metrics to plot (all 9 metrics from calculate_all_metrics)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'macro_recall', 'macro_f1', 'auc', 'macro_auc', 'gmean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Macro Recall', 'Macro F1', 'ROC-AUC', 'Macro AUC', 'G-Mean']

    # Color palette for 10 models
    colors = ['#2E86AB', '#A23B72', '#27AE60', '#E67E22', '#9B59B6', '#F24236', '#F5A623',
              '#1ABC9C', '#34495E', '#E74C3C']
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


if __name__ == '__main__':
    import torch

    # Generate all visualization charts
    generate_all_visualizations()

    # Generate common visualization chart examples
    generate_common_visualizations()

    # Can also generate specific charts individually
    # plot_model_architecture('Hybrid', 'hybrid_architecture.png')
    # plot_architecture_comparison('architecture_comparison.png')
    # plot_bar_chart({'A': 10, 'B': 20, 'C': 30}, None, "Example Bar Chart")
    # plot_pie_chart({'A': 10, 'B': 20, 'C': 30}, None, "Example Pie Chart")


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
    for key, value in results_collector.items():
        if key in ['IForest', 'MixHop_GCN_Single', 'MixHop_GAT_Single', 'MixHop_GIN_Single', 'MixHop_GraphSAGE_Single', 'MixHop_GCN_Bagging', 'MixHop_GAT_Bagging', 'MixHop_GIN_Bagging', 'MixHop_GraphSAGE_Bagging', 'MixHop_Final_Ensemble']:
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
        'GCN_Single', 'GAT_Single', 'GIN_Single', 'GraphSAGE_Single',
        'GCN_Bagging', 'GAT_Bagging', 'GIN_Bagging', 'GraphSAGE_Bagging'
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

