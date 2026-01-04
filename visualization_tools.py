"""
可視化工具：為 GNN 模型生成各種圖表
包括模型架構圖、訓練曲線、性能比較、特徵可視化等
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

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設置樣式
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingHistory:
    """記錄訓練歷史"""
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
        """添加一個 epoch 的記錄"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_f1'].append(val_f1)
        self.history['val_acc'].append(val_acc)
        self.history['test_f1'].append(test_f1)
        self.history['test_acc'].append(test_acc)
        # 新增的指標
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
        """保存訓練歷史到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath):
        """從文件加載訓練歷史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)


def plot_model_architecture(model_name='GCN', save_path=None):
    """
    繪製模型架構圖
    
    Args:
        model_name: 模型名稱 ('GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid')
        save_path: 保存路徑
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定義顏色
    colors = {
        'input': '#E8F4F8',
        'conv': '#FFE5B4',
        'attention': '#FFB6C1',
        'output': '#D4EDDA',
        'fusion': '#F0E68C'
    }
    
    if model_name in ['GCN', 'GAT', 'GraphSAGE']:
        # 基礎 GNN 模型架構
        y_positions = [8, 6, 4, 2]
        
        # 輸入層
        input_box = FancyBboxPatch((3.5, y_positions[0]-0.4), 3, 0.8,
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, y_positions[0], '輸入特徵\n(節點特徵)', 
                ha='center', va='center', fontsize=12, weight='bold')
        
        # 隱藏層
        for i, y in enumerate(y_positions[1:-1]):
            layer_name = f'{model_name}Conv\nLayer {i+1}'
            if model_name == 'GAT':
                layer_name = f'GATConv\n(多頭注意力)\nLayer {i+1}'
            elif model_name == 'GraphSAGE':
                layer_name = f'SAGEConv\n(採樣聚合)\nLayer {i+1}'
            
            conv_box = FancyBboxPatch((3.5, y-0.4), 3, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor=colors['conv'],
                                      edgecolor='black', linewidth=2)
            ax.add_patch(conv_box)
            ax.text(5, y, layer_name, ha='center', va='center', fontsize=10)
            
            # 箭頭
            arrow = FancyArrowPatch((5, y_positions[i]-0.4), (5, y+0.4),
                                   arrowstyle='->', mutation_scale=20,
                                   color='black', linewidth=1.5)
            ax.add_patch(arrow)
        
        # 輸出層
        output_box = FancyBboxPatch((3.5, y_positions[-1]-0.4), 3, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['output'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(5, y_positions[-1], '分類輸出\n(Log Softmax)', 
                ha='center', va='center', fontsize=12, weight='bold')
        
        arrow = FancyArrowPatch((5, y_positions[-2]-0.4), (5, y_positions[-1]+0.4),
                               arrowstyle='->', mutation_scale=20,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)
        
        ax.text(5, 9.5, f'{model_name} 模型架構', 
                ha='center', va='top', fontsize=16, weight='bold')
    
    elif model_name == 'SGAT':
        # SGAT 模型架構
        ax.text(5, 9.5, 'SGAT 模型架構 (STA + GAT)', 
                ha='center', va='top', fontsize=16, weight='bold')
        
        # 輸入
        input_box = FancyBboxPatch((3.5, 8-0.3), 3, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, 8, '輸入特徵', ha='center', va='center', fontsize=11, weight='bold')
        
        # 分支
        # STA 分支
        sta_box = FancyBboxPatch((1, 6-0.3), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['attention'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(sta_box)
        ax.text(2.25, 6, 'STA\n(空間-時序\n注意力)', ha='center', va='center', fontsize=9)
        
        # GAT 分支
        gat_box = FancyBboxPatch((6.5, 6-0.3), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['conv'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(gat_box)
        ax.text(7.75, 6, 'GAT\n(圖注意力)', ha='center', va='center', fontsize=9)
        
        # 箭頭
        arrow1 = FancyArrowPatch((4, 8-0.3), (2.25, 6+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch((6, 8-0.3), (7.75, 6+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow2)
        
        # 融合層
        fusion_box = FancyBboxPatch((3.5, 4-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(fusion_box)
        ax.text(5, 4, '特徵融合層', ha='center', va='center', fontsize=11, weight='bold')
        
        arrow3 = FancyArrowPatch((2.25, 6-0.3), (4.5, 4+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow3)
        
        arrow4 = FancyArrowPatch((7.75, 6-0.3), (5.5, 4+0.3),
                                arrowstyle='->', mutation_scale=15,
                                color='black', linewidth=1.5)
        ax.add_patch(arrow4)
        
        # 輸出
        output_box = FancyBboxPatch((3.5, 2-0.3), 3, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['output'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(5, 2, '分類輸出', ha='center', va='center', fontsize=11, weight='bold')
        
        arrow5 = FancyArrowPatch((5, 4-0.3), (5, 2+0.3),
                                 arrowstyle='->', mutation_scale=15,
                                 color='black', linewidth=1.5)
        ax.add_patch(arrow5)
    
    elif model_name == 'Hybrid':
        # Hybrid 模型架構
        ax.text(5, 9.5, 'Hybrid 模型架構 (SGAT + GraphSAGE + 時序特徵)', 
                ha='center', va='top', fontsize=14, weight='bold')
        
        # 輸入
        input_box = FancyBboxPatch((3.5, 8.5-0.25), 3, 0.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(5, 8.5, '輸入特徵', ha='center', va='center', fontsize=10, weight='bold')
        
        # 三個分支
        # GraphSAGE
        sage_box = FancyBboxPatch((0.5, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['conv'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(sage_box)
        ax.text(1.5, 6.5, 'GraphSAGE\n編碼器', ha='center', va='center', fontsize=8)
        
        # 時序特徵
        temp_box = FancyBboxPatch((4, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['attention'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(temp_box)
        ax.text(5, 6.5, '時序特徵\n提取器\n(GRU+Conv1D)', ha='center', va='center', fontsize=8)
        
        # 原始特徵
        orig_box = FancyBboxPatch((7.5, 6.5-0.25), 2, 0.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['input'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(orig_box)
        ax.text(8.5, 6.5, '原始特徵', ha='center', va='center', fontsize=8)
        
        # 箭頭
        for x in [1.5, 5, 8.5]:
            arrow = FancyArrowPatch((x, 8.5-0.25), (x, 6.5+0.25),
                                   arrowstyle='->', mutation_scale=12,
                                   color='black', linewidth=1.2)
            ax.add_patch(arrow)
        
        # 合併
        merge_box = FancyBboxPatch((3, 5-0.25), 4, 0.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(merge_box)
        ax.text(5, 5, '特徵合併', ha='center', va='center', fontsize=10, weight='bold')
        
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
        ax.text(5, 3.5, 'SGAT 處理', ha='center', va='center', fontsize=10, weight='bold')
        
        arrow = FancyArrowPatch((5, 5-0.25), (5, 3.5+0.25),
                               arrowstyle='->', mutation_scale=12,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)
        
        # 分類器
        classifier_box = FancyBboxPatch((3.5, 2-0.25), 3, 0.5,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colors['output'],
                                       edgecolor='black', linewidth=2)
        ax.add_patch(classifier_box)
        ax.text(5, 2, '分類器', ha='center', va='center', fontsize=10, weight='bold')
        
        arrow = FancyArrowPatch((5, 3.5-0.25), (5, 2+0.25),
                               arrowstyle='->', mutation_scale=12,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型架構圖已保存到: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_architecture.png', dpi=300, bbox_inches='tight')
        print(f"模型架構圖已保存到: {model_name.lower()}_architecture.png")
    
    plt.close()


def plot_training_curves(history, save_path=None, model_name='Model'):
    """
    繪製訓練曲線
    
    Args:
        history: TrainingHistory 對象或字典
        save_path: 保存路徑
        model_name: 模型名稱
    """
    if isinstance(history, TrainingHistory):
        history = history.history
    
    epochs = history['epoch']
    
    # 改為 3x2 以顯示額外指標 (Macro Recall, G-Mean)
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(f'{model_name} 訓練過程', fontsize=16, weight='bold')
    
    # 損失曲線
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='訓練損失', linewidth=2, color='#3498db')
    ax1.plot(epochs, history['val_loss'], label='驗證損失', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('損失曲線', fontsize=13, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # F1 分數曲線
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_f1'], label='驗證 F1', linewidth=2, color='#2ecc71')
    ax2.plot(epochs, history['test_f1'], label='測試 F1', linewidth=2, color='#9b59b6')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 分數曲線', fontsize=13, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 準確率曲線
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_acc'], label='驗證準確率', linewidth=2, color='#f39c12')
    ax3.plot(epochs, history['test_acc'], label='測試準確率', linewidth=2, color='#1abc9c')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('準確率曲線', fontsize=13, weight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 綜合性能 - F1 vs Acc
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, history['val_f1'], label='驗證 F1', linewidth=2, color='#2ecc71')
    line2 = ax4_twin.plot(epochs, history['val_acc'], label='驗證準確率', linewidth=2, color='#f39c12', linestyle='--')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1 Score', fontsize=12, color='#2ecc71')
    ax4_twin.set_ylabel('Accuracy', fontsize=12, color='#f39c12')
    ax4.set_title('綜合性能指標', fontsize=13, weight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 第三行： Macro Recall 和 G-Mean
    ax5 = axes[2, 0]
    val_macro = history.get('val_macro_recall', [0]*len(epochs))
    test_macro = history.get('test_macro_recall', [0]*len(epochs))
    ax5.plot(epochs, val_macro, label='驗證 Macro Recall', linewidth=2, color='#8e44ad')
    ax5.plot(epochs, test_macro, label='測試 Macro Recall', linewidth=2, color='#3498db', linestyle='--')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Macro Recall', fontsize=12)
    ax5.set_title('Macro Recall 曲線', fontsize=13, weight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    val_g = history.get('val_gmean', [0]*len(epochs))
    test_g = history.get('test_gmean', [0]*len(epochs))
    ax6.plot(epochs, val_g, label='驗證 G-Mean', linewidth=2, color='#e67e22')
    ax6.plot(epochs, test_g, label='測試 G-Mean', linewidth=2, color='#2c3e50', linestyle='--')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('G-Mean', fontsize=12)
    ax6.set_title('G-Mean 曲線', fontsize=13, weight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 第四行： Macro F1 和 Macro AUC
    ax7 = axes[3, 0]
    val_mf1 = history.get('val_macro_f1', [0]*len(epochs))
    test_mf1 = history.get('test_macro_f1', [0]*len(epochs))
    ax7.plot(epochs, val_mf1, label='驗證 Macro F1', linewidth=2, color='#16a085')
    ax7.plot(epochs, test_mf1, label='測試 Macro F1', linewidth=2, color='#e74c3c', linestyle='--')
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Macro F1', fontsize=12)
    ax7.set_title('Macro F1 曲線', fontsize=13, weight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    ax8 = axes[3, 1]
    val_mauc = history.get('val_macro_auc', [0]*len(epochs))
    test_mauc = history.get('test_macro_auc', [0]*len(epochs))
    ax8.plot(epochs, val_mauc, label='驗證 Macro AUC', linewidth=2, color='#2c3e50')
    ax8.plot(epochs, test_mauc, label='測試 Macro AUC', linewidth=2, color='#9b59b6', linestyle='--')
    ax8.set_xlabel('Epoch', fontsize=12)
    ax8.set_ylabel('Macro AUC', fontsize=12)
    ax8.set_title('Macro AUC 曲線', fontsize=13, weight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"訓練曲線已保存到: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"訓練曲線已保存到: {model_name.lower()}_training_curves.png")
    
    plt.close()


def plot_model_comparison(results_dict, save_path=None):
    """
    繪製模型性能比較圖
    
    Args:
        results_dict: 字典，格式為 {model_name: {metric: value}}
        save_path: 保存路徑
    """
    models = list(results_dict.keys())
    metrics = ['val_f1', 'test_f1', 'val_acc', 'test_acc']
    metric_names = ['驗證 F1', '測試 F1', '驗證準確率', '測試準確率']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('模型性能比較', fontsize=16, weight='bold')
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        values = [results_dict[model].get(metric, 0) for model in models]
        colors = sns.color_palette("husl", len(models))
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加數值標籤
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
        print(f"模型比較圖已保存到: {save_path}")
    else:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"模型比較圖已保存到: model_comparison.png")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """
    繪製混淆矩陣
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        model_name: 模型名稱
        save_path: 保存路徑
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['合法', '非法'],
                yticklabels=['合法', '非法'],
                cbar_kws={'label': '數量'})
    
    plt.title(f'{model_name} 混淆矩陣', fontsize=14, weight='bold')
    plt.ylabel('真實標籤', fontsize=12)
    plt.xlabel('預測標籤', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存到: {save_path}")
    else:
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存到: {model_name.lower()}_confusion_matrix.png")
    
    plt.close()


def plot_feature_visualization_2d(features_2d, labels, title="特徵可視化", 
                                   method="PCA", save_path=None):
    """
    繪製 2D 特徵可視化
    
    Args:
        features_2d: 2D 特徵矩陣 [num_nodes, 2]
        labels: 節點標籤
        title: 圖表標題
        method: 降維方法 ('PCA' 或 't-SNE')
        save_path: 保存路徑
    """
    if HAS_TORCH:
        if isinstance(features_2d, torch.Tensor):
            features_2d = features_2d.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    # 只顯示已知標籤的節點
    known_mask = labels != -1
    known_features = features_2d[known_mask]
    known_labels = labels[known_mask]
    
    # 繪製不同類別的點
    for label in [0, 1]:
        mask = known_labels == label
        label_name = "合法" if label == 0 else "非法"
        color = '#3498db' if label == 0 else '#e74c3c'
        plt.scatter(known_features[mask, 0], known_features[mask, 1], 
                   label=label_name, alpha=0.6, s=20, color=color, edgecolors='black', linewidths=0.5)
    
    xlabel = "第一主成分" if method == "PCA" else "t-SNE 維度 1"
    ylabel = "第二主成分" if method == "PCA" else "t-SNE 維度 2"
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} ({method})', fontsize=14, weight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特徵可視化圖已保存到: {save_path}")
    else:
        filename = f'feature_visualization_{method.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"特徵可視化圖已保存到: {filename}")
    
    plt.close()


def plot_architecture_comparison(save_path=None):
    """
    繪製所有模型的架構對比圖
    """
    models = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    
    fig, axes = plt.subplots(1, len(models), figsize=(20, 4))
    fig.suptitle('GNN 模型架構對比', fontsize=16, weight='bold')
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 簡化的架構表示
        y_positions = [0.8, 0.6, 0.4, 0.2]
        
        # 輸入
        rect = mpatches.FancyBboxPatch((0.2, y_positions[0]-0.05), 0.6, 0.1,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#E8F4F8', edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.5, y_positions[0], 'Input', ha='center', va='center', fontsize=8)
        
        # 中間層
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
            
            # 箭頭
            arrow = FancyArrowPatch((0.5, y_positions[i]-0.05), (0.5, y+0.05),
                                   arrowstyle='->', mutation_scale=10,
                                   color='black', linewidth=1)
            ax.add_patch(arrow)
        
        # 輸出
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
        print(f"架構對比圖已保存到: {save_path}")
    else:
        plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
        print(f"架構對比圖已保存到: architecture_comparison.png")
    
    plt.close()


def generate_all_visualizations(output_dir='visualizations'):
    """
    生成所有可視化圖表
    
    Args:
        output_dir: 輸出目錄
    """
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print("開始生成可視化圖表...")
    print("="*60)
    
    # 1. 生成所有模型的架構圖
    print("\n1. 生成模型架構圖...")
    models = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    for model in models:
        plot_model_architecture(model, 
                               os.path.join(output_dir, f'{model.lower()}_architecture.png'))
    
    # 2. 生成架構對比圖
    print("\n2. 生成架構對比圖...")
    plot_architecture_comparison(os.path.join(output_dir, 'architecture_comparison.png'))
    
    # 3. 生成示例訓練曲線（使用模擬數據）
    print("\n3. 生成示例訓練曲線...")
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
    
    for model in models[:3]:  # 只為前三個模型生成示例
        plot_training_curves(example_history, 
                            os.path.join(output_dir, f'{model.lower()}_training_curves.png'),
                            model)
    
    # 4. 生成示例模型比較圖
    print("\n4. 生成模型性能比較圖...")
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
    print(f"所有可視化圖表已生成並保存到 '{output_dir}' 目錄")
    print("="*60)


# ============================================================================
# 常見數據可視化圖表
# ============================================================================

def plot_bar_chart(data, x_labels, title="柱狀圖", xlabel="類別", ylabel="數值",
                   colors=None, save_path=None, figsize=(10, 6), rotation=0):
    """
    繪製柱狀圖
    
    Args:
        data: 數據值列表或字典
        x_labels: X軸標籤列表
        title: 圖表標題
        xlabel: X軸標籤
        ylabel: Y軸標籤
        colors: 顏色列表（可選）
        save_path: 保存路徑
        figsize: 圖表大小
        rotation: X軸標籤旋轉角度
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
    
    # 添加數值標籤
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
        print(f"柱狀圖已保存到: {save_path}")
    else:
        plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
        print(f"柱狀圖已保存到: bar_chart.png")
    
    plt.close()


def plot_pie_chart(data, labels, title="餅圖", colors=None, save_path=None, 
                   figsize=(10, 8), autopct='%1.1f%%', startangle=90):
    """
    繪製餅圖
    
    Args:
        data: 數據值列表或字典
        labels: 標籤列表
        title: 圖表標題
        colors: 顏色列表（可選）
        save_path: 保存路徑
        figsize: 圖表大小
        autopct: 百分比格式
        startangle: 起始角度
    """
    plt.figure(figsize=figsize)
    
    if isinstance(data, dict):
        labels = list(data.keys())
        values = list(data.values())
    else:
        values = data
    
    if colors is None:
        colors = sns.color_palette("husl", len(values))
    
    # 突出顯示最大的扇區
    explode = [0.05 if i == np.argmax(values) else 0 for i in range(len(values))]
    
    wedges, texts, autotexts = plt.pie(values, labels=labels, colors=colors,
                                       autopct=autopct, startangle=startangle,
                                       explode=explode, shadow=True,
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    
    # 美化自動文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"餅圖已保存到: {save_path}")
    else:
        plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
        print(f"餅圖已保存到: pie_chart.png")
    
    plt.close()


def plot_scatter_distribution(x, y, labels=None, title="散點數據分佈圖", 
                             xlabel="X軸", ylabel="Y軸", save_path=None,
                             figsize=(10, 8), alpha=0.6, size=50, color_col=None):
    """
    繪製散點數據分佈圖
    
    Args:
        x: X軸數據
        y: Y軸數據
        labels: 標籤（用於不同顏色，可選）
        title: 圖表標題
        xlabel: X軸標籤
        ylabel: Y軸標籤
        save_path: 保存路徑
        figsize: 圖表大小
        alpha: 透明度
        size: 點的大小
        color_col: 用於顏色的列（如果數據是DataFrame）
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
        # 根據標籤使用不同顏色
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f'類別 {label}' if isinstance(label, (int, float)) else str(label)
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
        print(f"散點圖已保存到: {save_path}")
    else:
        plt.savefig('scatter_distribution.png', dpi=300, bbox_inches='tight')
        print(f"散點圖已保存到: scatter_distribution.png")
    
    plt.close()


def plot_histogram(data, bins=30, title="直方圖", xlabel="數值", ylabel="頻率",
                  save_path=None, figsize=(10, 6), color='steelblue', alpha=0.7):
    """
    繪製直方圖
    
    Args:
        data: 數據列表或數組
        bins: 分箱數
        title: 圖表標題
        xlabel: X軸標籤
        ylabel: Y軸標籤
        save_path: 保存路徑
        figsize: 圖表大小
        color: 顏色
        alpha: 透明度
    """
    if HAS_TORCH:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    n, bins, patches = plt.hist(data, bins=bins, color=color, alpha=alpha, 
                                edgecolor='black', linewidth=1.2)
    
    # 添加均值線
    mean_val = np.mean(data)
    std_val = np.std(data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'均值: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'±1標準差')
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
        print(f"直方圖已保存到: {save_path}")
    else:
        plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
        print(f"直方圖已保存到: histogram.png")
    
    plt.close()


def plot_boxplot(data, labels=None, title="箱線圖", ylabel="數值",
                save_path=None, figsize=(10, 6), vert=True):
    """
    繪製箱線圖
    
    Args:
        data: 數據列表（多組數據）或單組數據
        labels: 標籤列表
        title: 圖表標題
        ylabel: Y軸標籤
        save_path: 保存路徑
        figsize: 圖表大小
        vert: 是否垂直顯示
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
        print(f"箱線圖已保存到: {save_path}")
    else:
        plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
        print(f"箱線圖已保存到: boxplot.png")
    
    plt.close()


def plot_heatmap(data, row_labels=None, col_labels=None, title="熱力圖",
                save_path=None, figsize=(10, 8), cmap='viridis', annot=True,
                fmt='.2f', cbar_label="數值"):
    """
    繪製熱力圖
    
    Args:
        data: 2D 數據數組或矩陣
        row_labels: 行標籤
        col_labels: 列標籤
        title: 圖表標題
        save_path: 保存路徑
        figsize: 圖表大小
        cmap: 顏色映射
        annot: 是否顯示數值註解
        fmt: 數值格式
        cbar_label: 顏色條標籤
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
        print(f"熱力圖已保存到: {save_path}")
    else:
        plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
        print(f"熱力圖已保存到: heatmap.png")
    
    plt.close()


def plot_line_chart(x, y, labels=None, title="折線圖", xlabel="X軸", ylabel="Y軸",
                   save_path=None, figsize=(10, 6), marker='o', linestyle='-'):
    """
    繪製折線圖
    
    Args:
        x: X軸數據（單條線）或 X軸數據列表（多條線）
        y: Y軸數據（單條線）或 Y軸數據列表（多條線）
        labels: 標籤列表（多條線時使用）
        title: 圖表標題
        xlabel: X軸標籤
        ylabel: Y軸標籤
        save_path: 保存路徑
        figsize: 圖表大小
        marker: 標記樣式
        linestyle: 線條樣式
    """
    if HAS_TORCH:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # 檢查是否為多條線
    if isinstance(y, list) or (isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[0] > 1):
        # 多條線
        if labels is None:
            labels = [f'線 {i+1}' for i in range(len(y))]
        
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
        # 單條線
        plt.plot(x, y, marker=marker, linestyle=linestyle, 
                linewidth=2, markersize=6, color='steelblue')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"折線圖已保存到: {save_path}")
    else:
        plt.savefig('line_chart.png', dpi=300, bbox_inches='tight')
        print(f"折線圖已保存到: line_chart.png")
    
    plt.close()


def plot_class_distribution(labels, title="類別分佈", save_path=None, 
                           figsize=(10, 6), normalize=False):
    """
    繪製類別分佈圖（柱狀圖 + 餅圖）
    
    Args:
        labels: 標籤數組
        title: 圖表標題
        save_path: 保存路徑
        figsize: 圖表大小
        normalize: 是否標準化為百分比
    """
    if HAS_TORCH:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
    # 統計類別
    unique, counts = np.unique(labels, return_counts=True)
    if normalize:
        counts = counts / len(labels) * 100
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 柱狀圖
    colors = sns.color_palette("husl", len(unique))
    bars = ax1.bar(unique.astype(str), counts, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:.2f}{"%" if normalize else ""}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_xlabel('類別', fontsize=12)
    ax1.set_ylabel('百分比' if normalize else '數量', fontsize=12)
    ax1.set_title(f'{title} - 柱狀圖', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 餅圖
    explode = [0.05 if i == np.argmax(counts) else 0 for i in range(len(unique))]
    wedges, texts, autotexts = ax2.pie(counts, labels=unique.astype(str), 
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=explode, shadow=True,
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax2.set_title(f'{title} - 餅圖', fontsize=13, weight='bold')
    ax2.axis('equal')
    
    plt.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"類別分佈圖已保存到: {save_path}")
    else:
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"類別分佈圖已保存到: class_distribution.png")
    
    plt.close()


def generate_common_visualizations(output_dir='visualizations/common'):
    """
    生成常見可視化圖表的示例
    
    Args:
        output_dir: 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成常見可視化圖表示例...")
    print("="*60)
    
    # 1. 柱狀圖示例
    print("\n1. 生成柱狀圖...")
    model_scores = {
        'GCN': 0.75,
        'GAT': 0.78,
        'GraphSAGE': 0.76,
        'SGAT': 0.82,
        'Hybrid': 0.85
    }
    plot_bar_chart(model_scores, None, "模型性能比較 (F1 Score)", 
                  "模型", "F1 Score",
                  save_path=os.path.join(output_dir, 'bar_chart_example.png'))
    
    # 2. 餅圖示例
    print("\n2. 生成餅圖...")
    class_dist = {
        '合法交易': 70,
        '非法交易': 25,
        '未知': 5
    }
    plot_pie_chart(class_dist, None, "交易類別分佈",
                  save_path=os.path.join(output_dir, 'pie_chart_example.png'))
    
    # 3. 散點圖示例
    print("\n3. 生成散點圖...")
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
    
    plot_scatter_distribution(x, y, labels, "數據分佈散點圖",
                             "特徵 1", "特徵 2",
                             save_path=os.path.join(output_dir, 'scatter_distribution_example.png'))
    
    # 4. 直方圖示例
    print("\n4. 生成直方圖...")
    data_hist = np.random.normal(100, 15, 1000)
    plot_histogram(data_hist, bins=30, title="交易金額分佈直方圖",
                  xlabel="金額", ylabel="頻率",
                  save_path=os.path.join(output_dir, 'histogram_example.png'))
    
    # 5. 箱線圖示例
    print("\n5. 生成箱線圖...")
    data_box = [
        np.random.normal(100, 10, 100),
        np.random.normal(150, 15, 100),
        np.random.normal(120, 12, 100)
    ]
    plot_boxplot(data_box, ['合法', '非法', '未知'],
                "不同類別交易金額分佈箱線圖", "金額",
                save_path=os.path.join(output_dir, 'boxplot_example.png'))
    
    # 6. 熱力圖示例
    print("\n6. 生成熱力圖...")
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    plot_heatmap(correlation_matrix,
                row_labels=['特徵1', '特徵2', '特徵3', '特徵4', '特徵5'],
                col_labels=['特徵1', '特徵2', '特徵3', '特徵4', '特徵5'],
                title="特徵相關性熱力圖",
                save_path=os.path.join(output_dir, 'heatmap_example.png'),
                cmap='coolwarm', cbar_label="相關性")
    
    # 7. 折線圖示例
    print("\n7. 生成折線圖...")
    epochs = np.arange(1, 101)
    train_loss = 0.5 * np.exp(-epochs/50) + 0.1 + np.random.normal(0, 0.02, 100)
    val_loss = 0.6 * np.exp(-epochs/50) + 0.15 + np.random.normal(0, 0.02, 100)
    
    plot_line_chart(epochs, [train_loss, val_loss], ['訓練損失', '驗證損失'],
                   "訓練損失曲線", "Epoch", "Loss",
                   save_path=os.path.join(output_dir, 'line_chart_example.png'))
    
    # 8. 類別分佈圖示例
    print("\n8. 生成類別分佈圖...")
    labels_dist = np.random.choice([0, 1, -1], size=1000, p=[0.7, 0.25, 0.05])
    plot_class_distribution(labels_dist, "交易類別分佈",
                           save_path=os.path.join(output_dir, 'class_distribution_example.png'),
                           normalize=True)
    
    print("\n" + "="*60)
    print(f"所有常見可視化圖表示例已生成並保存到 '{output_dir}' 目錄")
    print("="*60)


def run_full_report(model, data, history, embeds=None, output_dir='visualizations/full_report', reduction_method='tsne'):
    """
    生成完整報告：模型架構、訓練曲線、混淆矩陣、Embedding 可視化、時序動態、累積增益等。

    Args:
        model: 訓練後的模型物件（PyTorch）
        data: PyG Data（需含 train/val/test masks 與 timesteps）
        history: TrainingHistory 或等價 dict
        embeds: 若為 None 則會嘗試從 model 提取；若為高維嵌入則使用降維（TSNE/PCA）
        output_dir: 圖表輸出資料夾
        reduction_method: 'tsne' 或 'pca'，用於 embedding 降維
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

    # 1) 模型架構圖
    try:
        plot_model_architecture(model_name, save_path=os.path.join(output_dir, f"{model_name.lower()}_architecture.png"))
    except Exception as e:
        print(f"[run_full_report] 無法繪製模型架構圖: {e}")

    # 2) 訓練曲線
    try:
        plot_training_curves(history, save_path=os.path.join(output_dir, f"{model_name.lower()}_training_curves.png"), model_name=model_name)
    except Exception as e:
        print(f"[run_full_report] 無法繪製訓練曲線: {e}")

    # 3) 混淆矩陣（驗證 / 測試）
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
        print(f"[run_full_report] 無法生成混淆矩陣: {e}")

    # 4) Embedding 可視化（若 embeds 未提供則嘗試從 model 提取）
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
            # features 可能為 torch tensor 或 numpy
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
        print(f"[run_full_report] 無法生成 Embedding 可視化: {e}")

    # 5) Temporal dynamics: 每個 timestep 的非法比例 / F1（若有 timesteps）
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

            # 繪圖
            try:
                plt.figure(figsize=(10,4))
                plt.plot(unique_ts, ts_illicit_ratio, marker='o', label='非法比例', color='red')
                plt.xlabel('Timestep'); plt.ylabel('非法比例'); plt.title(f'{model_name} - 時序非法比例'); plt.grid(True); plt.legend()
                plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_temporal_illicit_ratio.png'), dpi=300); plt.close()
                plt.figure(figsize=(10,4))
                plt.plot(unique_ts, ts_f1, marker='o', label='F1 Score', color='blue')
                plt.xlabel('Timestep'); plt.ylabel('F1'); plt.title(f'{model_name} - 時序 F1 分數'); plt.grid(True); plt.legend()
                plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_temporal_f1.png'), dpi=300); plt.close()
            except Exception as e:
                print(f"[run_full_report] 無法繪製時序動態圖: {e}")
    except Exception as e:
        print(f"[run_full_report] 時序動態處理失敗: {e}")

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
            plt.plot(perc_accounts, gain, label='累積增益', color='purple')
            plt.xlabel('前 X% 帳戶'); plt.ylabel('累積抓到的非法比例'); plt.title(f'{model_name} - 累積增益圖'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_cumulative_gain.png'), dpi=300); plt.close()
    except Exception as e:
        print(f"[run_full_report] 累積增益圖生成失敗: {e}")

    print(f"[run_full_report] 完成，報告已保存到: {output_dir}")

if __name__ == '__main__':
    import torch
    
    # 生成所有可視化圖表
    generate_all_visualizations()
    
    # 生成常見可視化圖表示例
    generate_common_visualizations()
    
    # 也可以單獨生成特定圖表
    # plot_model_architecture('Hybrid', 'hybrid_architecture.png')
    # plot_architecture_comparison('architecture_comparison.png')
    # plot_bar_chart({'A': 10, 'B': 20, 'C': 30}, None, "示例柱狀圖")
    # plot_pie_chart({'A': 10, 'B': 20, 'C': 30}, None, "示例餅圖")

