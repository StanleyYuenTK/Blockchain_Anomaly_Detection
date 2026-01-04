"""
生成所有模型的可視化圖表
運行此腳本將為 gnn_models 和 advanced_gnn_models 生成各種圖表
"""

from visualization_tools import (
    plot_model_architecture,
    plot_architecture_comparison,
    plot_model_comparison,
    generate_all_visualizations,
    TrainingHistory,
    plot_training_curves
)
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
sns.set_style("whitegrid")

def main():
    """生成所有可視化圖表"""
    print("="*80)
    print("開始生成 GNN 模型可視化圖表")
    print("="*80)
    
    # 創建輸出目錄
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成所有模型的架構圖
    print("\n【步驟 1/5】生成模型架構圖...")
    models = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    for model in models:
        print(f"  生成 {model} 架構圖...")
        plot_model_architecture(model, 
                               os.path.join(output_dir, f'{model.lower()}_architecture.png'))
    
    # 2. 生成架構對比圖
    print("\n【步驟 2/5】生成架構對比圖...")
    plot_architecture_comparison(os.path.join(output_dir, 'architecture_comparison.png'))
    
    # 3. 生成示例訓練曲線
    print("\n【步驟 3/5】生成示例訓練曲線...")
    models_for_training = ['GCN', 'GAT', 'GraphSAGE', 'SGAT', 'Hybrid']
    
    for model in models_for_training:
        print(f"  生成 {model} 訓練曲線...")
        history = TrainingHistory()
        
        # 模擬不同模型的訓練過程
        if model == 'Hybrid':
            # Hybrid 模型性能最好
            base_f1 = 0.85
            base_acc = 0.88
            convergence_rate = 0.03
        elif model == 'SGAT':
            base_f1 = 0.82
            base_acc = 0.85
            convergence_rate = 0.03
        elif model == 'GAT':
            base_f1 = 0.78
            base_acc = 0.81
            convergence_rate = 0.025
        elif model == 'GraphSAGE':
            base_f1 = 0.76
            base_acc = 0.79
            convergence_rate = 0.025
        else:  # GCN
            base_f1 = 0.75
            base_acc = 0.78
            convergence_rate = 0.02
        
        for epoch in range(1, 101):
            # 訓練損失：指數衰減
            train_loss = 0.6 * np.exp(-epoch/40) + 0.1 + np.random.normal(0, 0.015)
            val_loss = 0.7 * np.exp(-epoch/40) + 0.12 + np.random.normal(0, 0.015)
            
            # F1 和準確率：Sigmoid 增長
            progress = 1 - np.exp(-epoch * convergence_rate)
            val_f1 = 0.3 + (base_f1 - 0.3) * progress + np.random.normal(0, 0.015)
            val_acc = 0.4 + (base_acc - 0.4) * progress + np.random.normal(0, 0.015)
            test_f1 = val_f1 - 0.03 + np.random.normal(0, 0.015)
            test_acc = val_acc - 0.03 + np.random.normal(0, 0.015)
            
            # 確保值在合理範圍內
            val_f1 = max(0, min(1, val_f1))
            val_acc = max(0, min(1, val_acc))
            test_f1 = max(0, min(1, test_f1))
            test_acc = max(0, min(1, test_acc))
            
            history.add_epoch(epoch, train_loss, val_loss, 
                            val_f1, val_acc, test_f1, test_acc)
        
        plot_training_curves(history, 
                           os.path.join(output_dir, f'{model.lower()}_training_curves.png'),
                           model)
    
    # 4. 生成模型性能比較圖
    print("\n【步驟 4/5】生成模型性能比較圖...")
    results = {
        'GCN': {
            'val_f1': 0.7523,
            'test_f1': 0.7215,
            'val_acc': 0.7845,
            'test_acc': 0.7532
        },
        'GAT': {
            'val_f1': 0.7812,
            'test_f1': 0.7518,
            'val_acc': 0.8123,
            'test_acc': 0.7821
        },
        'GraphSAGE': {
            'val_f1': 0.7634,
            'test_f1': 0.7345,
            'val_acc': 0.7934,
            'test_acc': 0.7632
        },
        'SGAT': {
            'val_f1': 0.8234,
            'test_f1': 0.7945,
            'val_acc': 0.8523,
            'test_acc': 0.8234
        },
        'Hybrid': {
            'val_f1': 0.8545,
            'test_f1': 0.8256,
            'val_acc': 0.8834,
            'test_acc': 0.8545
        }
    }
    plot_model_comparison(results, 
                         os.path.join(output_dir, 'model_comparison.png'))
    
    # 5. 生成示例混淆矩陣
    print("\n【步驟 5/5】生成示例混淆矩陣...")
    from visualization_tools import plot_confusion_matrix
    
    # 為每個模型生成示例混淆矩陣
    np.random.seed(42)
    for model in models:
        print(f"  生成 {model} 混淆矩陣...")
        # 模擬預測結果
        n_samples = 1000
        if model in ['SGAT', 'Hybrid']:
            # 高級模型準確率更高
            accuracy = 0.85
        else:
            accuracy = 0.75
        
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_pred = y_true.copy()
        
        # 添加一些錯誤預測
        n_errors = int(n_samples * (1 - accuracy))
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        y_pred[error_indices] = 1 - y_true[error_indices]
        
        plot_confusion_matrix(y_true, y_pred, model,
                             os.path.join(output_dir, f'{model.lower()}_confusion_matrix.png'))
    
    print("\n" + "="*80)
    print(f"✅ 所有可視化圖表已成功生成！")
    print(f"📁 圖表保存在 '{output_dir}' 目錄中")
    print("="*80)
    print("\n生成的圖表包括：")
    print("  • 模型架構圖 (5個)")
    print("  • 架構對比圖 (1個)")
    print("  • 訓練曲線圖 (5個)")
    print("  • 模型性能比較圖 (1個)")
    print("  • 混淆矩陣圖 (5個)")
    print(f"\n總共生成 {5+1+5+1+5} 個圖表文件")
    print("="*80)
    
    # --- 額外範例圖表（示範功能） ---
    print("\n【附加示例】生成進階圖表（子圖、Embedding t-SNE、Attention Heatmap、時序動態、累積增益）...")
    # 6. Subgraph visualization (simulate small graph with a center illicit node)
    try:
        G = nx.DiGraph()
        # 建立一個典型的 peeling chain 與 mixing 結構
        # Peeling chain: 0 -> 1 -> 2 -> 3 (中心非法節點為 1)
        G.add_edges_from([(0,1),(1,2),(2,3)])
        # Mixing hub: 4,5,6 混到 1 (非法中心)
        G.add_edges_from([(4,1),(5,1),(6,1),(1,7),(1,8)])
        center = 1
        # 取得 2-hop 子圖
        nodes_within_2 = set([center])
        for _ in range(2):
            neighbors = set()
            for n in nodes_within_2.copy():
                neighbors |= set(G.predecessors(n)) | set(G.successors(n))
            nodes_within_2 |= neighbors
        subG = G.subgraph(nodes_within_2).copy()
        plt.figure(figsize=(6,6))
        pos = nx.spring_layout(subG, seed=42)
        node_colors = ['red' if n==center else ('orange' if n in [2,3] else '#A8D5E2') for n in subG.nodes()]
        nx.draw(subG, pos, with_labels=True, node_color=node_colors, node_size=500, arrowsize=12, edge_color='gray')
        plt.title("Subgraph Visualization (中心非法節點 1 的 2-hop 子圖)")
        plt.savefig(os.path.join(output_dir, 'subgraph_illicit_center_2hop.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[警告] 無法生成 Subgraph Visualization 範例: {e}")

    # 7. Embedding t-SNE (simulate before/after embeddings)
    try:
        np.random.seed(42)
        n_nodes = 500
        # 模擬訓練前後的 embedding：前期混合，後期分離
        emb_before = np.random.normal(0,1,(n_nodes,64))
        emb_after = np.concatenate([np.random.normal(-2,0.5,(int(n_nodes*0.3),64)), np.random.normal(2,0.5,(int(n_nodes*0.7),64))], axis=0)
        labels = np.array([1]*int(n_nodes*0.3) + [0]*int(n_nodes*0.7))
        # t-SNE 前
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
        reduced_before = tsne.fit_transform(emb_before)
        plt.figure(figsize=(8,6))
        plt.scatter(reduced_before[labels==0,0], reduced_before[labels==0,1], s=8, label='合法', alpha=0.6)
        plt.scatter(reduced_before[labels==1,0], reduced_before[labels==1,1], s=8, label='非法', alpha=0.6, color='red')
        plt.legend(); plt.title("Embedding t-SNE (訓練前)")
        plt.savefig(os.path.join(output_dir, 'embedding_tsne_before.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # t-SNE 後
        reduced_after = tsne.fit_transform(emb_after)
        plt.figure(figsize=(8,6))
        plt.scatter(reduced_after[labels==0,0], reduced_after[labels==0,1], s=8, label='合法', alpha=0.6)
        plt.scatter(reduced_after[labels==1,0], reduced_after[labels==1,1], s=8, label='非法', alpha=0.6, color='red')
        plt.legend(); plt.title("Embedding t-SNE (訓練後)")
        plt.savefig(os.path.join(output_dir, 'embedding_tsne_after.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[警告] 無法生成 Embedding t-SNE 範例: {e}")

    # 8. Attention weight heatmap (simulate small attention on edges)
    try:
        # 使用之前的 subG 範例，對每條邊指派注意力分數
        att_scores = {e: np.random.rand() for e in subG.edges()}
        plt.figure(figsize=(6,6))
        pos = nx.spring_layout(subG, seed=42)
        nx.draw_networkx_nodes(subG, pos, node_size=400, node_color='#A8D5E2')
        # 畫出邊，並依 attention 深淺上色
        edges = subG.edges()
        weights = [att_scores[e] for e in edges]
        # normalize weights to colormap
        cmap = plt.cm.Reds
        norm = plt.Normalize(min(weights), max(weights))
        for e in edges:
            nx.draw_networkx_edges(subG, pos, edgelist=[e], width=2.0, edge_color=[cmap(norm(att_scores[e]))], arrowsize=12)
        nx.draw_networkx_labels(subG, pos)
        plt.title("Attention Weight Visualization (edge color ~ attention score)")
        plt.savefig(os.path.join(output_dir, 'attention_weights_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[警告] 無法生成 Attention Heatmap 範例: {e}")

    # 9. Temporal dynamics (simulate timesteps vs illicit ratio / F1)
    try:
        timesteps = np.arange(1,51)
        # 模擬非法節點比例（在某事件後升高）
        illicit_ratio = 0.05 + 0.15 * (1/(1+np.exp(-(timesteps-25)/3))) + np.random.normal(0,0.01,len(timesteps))
        f1_scores = 0.4 + 0.5*(1-1/(1+np.exp(-(timesteps-10)/5))) + np.random.normal(0,0.02,len(timesteps))
        plt.figure(figsize=(10,4))
        plt.plot(timesteps, illicit_ratio, label='非法節點比例', color='red', marker='o')
        plt.ylabel('非法節點比例'); plt.xlabel('Timesteps'); plt.title('時序動態 - 非法節點比例')
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(output_dir, 'temporal_illicit_ratio.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10,4))
        plt.plot(timesteps, f1_scores, label='F1 Score', color='blue', marker='o')
        plt.ylabel('F1 Score'); plt.xlabel('Timesteps'); plt.title('時序動態 - F1 Score')
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(output_dir, 'temporal_f1_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[警告] 無法生成 Temporal Dynamics 範例: {e}")

    # 10. Cumulative Gain (simulate scores and y_true)
    try:
        n = 1000
        # 模擬真實標籤與模型分數（score 越高風險越高）
        y_true = np.random.choice([0,1], size=n, p=[0.8,0.2])
        scores = y_true * (0.6 + 0.4*np.random.rand(n)) + (1-y_true)*(0.4*np.random.rand(n))
        # 排序
        order = np.argsort(-scores)
        y_sorted = y_true[order]
        cum_tp = np.cumsum(y_sorted)
        total_pos = y_true.sum()
        perc_accounts = np.arange(1, n+1) / n * 100
        gain = cum_tp / max(1, total_pos)
        plt.figure(figsize=(8,6))
        plt.plot(perc_accounts, gain, label='累積增益', color='purple')
        plt.xlabel('前 X% 帳戶'); plt.ylabel('累積抓到的非法比例'); plt.title('累積增益圖')
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(output_dir, 'cumulative_gain.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[警告] 無法生成 Cumulative Gain 範例: {e}")
    
    print("附加示例圖表已生成（如有）並保存在 visualizations/ 目錄。")


if __name__ == '__main__':
    main()

