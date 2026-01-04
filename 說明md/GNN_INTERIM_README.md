# GNN Interim Version - 區塊鏈異常檢測框架

## 概述

`gnn_interim_version.py` 是從 `main_gnn_framework.py` 簡化而來的版本，專注於區塊鏈異常檢測任務。

## 主要功能

### 🎯 模型架構
- **GCN**: 圖卷積網路
- **GAT**: 圖注意力網路
- **GIN**: 圖同構網路
- **GraphSAGE**: 圖采樣聚合網路
- **STA**: 空間-時間注意力網路 (學習多跳鄰居信息)

### 🔧 進階技術
- **Ensemble + Bagging**: 結合 GAT、GIN、GraphSAGE 三個模型
  - Bagging 保持異常節點的不平衡比例
  - 多模型軟投票集成
- **Hyperopt**: Bayesian Optimization with TPE
  - 自動超參數優化
  - 搜索空間包括模型類型、隱藏層、dropout、學習率等
- **GNNExplainer**: 模型解釋
  - 識別重要圖結構和節點特徵
  - 視覺化異常檢測原因

### 📊 評估指標
- **Macro AUC**, **Macro F1 Score**, **Macro Recall**
- **G-Mean**, **F1**, **Accuracy**, **Precision**, **Recall**

## 使用方法

### 環境準備
```bash
pip install -r requirements.txt
```

### 執行完整流程
```bash
python gnn_interim_version.py
```

程式將自動執行以下步驟：

1. **數據載入**: 載入 Elliptic 區塊鏈數據集
2. **超參數優化**: 使用 Hyperopt 進行 30 次 Bayesian 優化
3. **集成訓練**: 訓練 GAT、GIN、GraphSAGE 的 Bagging 集成
4. **最終集成**: 結合三個集成模型
5. **性能評估**: 計算所有評估指標
6. **視覺化**: 生成混淆矩陣
7. **模型解釋**: 使用 GNNExplainer 分析異常節點

## 輸出結果

- **控制台輸出**: 詳細的訓練過程和評估指標
- **results/** 目錄:
  - `confusion_matrix.png`: 混淆矩陣視覺化
  - `explanations/`: GNNExplainer 解釋結果
    - 節點特徵重要性圖
    - 子圖結構視覺化

## 關鍵創新點

### 1. Bagging with Class Balance Preservation
```python
# 保持原始數據中的類別不平衡比例
def create_balanced_bags(self, data, n_bags=5):
    # 計算原始比例並應用到每個袋子
```

### 2. Multi-Model Ensemble
```python
# 最終集成結合三個不同的 GNN 架構
class FinalEnsemble:
    def predict(self, data, device):
        # GAT + GIN + GraphSAGE 軟投票
```

### 3. Hyperopt Integration
```python
# Bayesian 優化搜索空間
space = {
    'model': hp.choice('model', ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'STA']),
    'hidden_channels': hp.choice('hidden_channels', [32, 64, 128, 256]),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    # ...
}
```

### 4. Comprehensive Evaluation
```python
# 完整評估套件
metrics = {
    'Macro AUC': macro_auc,
    'Macro F1': macro_f1,
    'Macro Recall': macro_recall,
    'G-Mean': gmean,
    'F1': f1,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall
}
```

## 技術細節

### STA 模型 (Spatial-Temporal Attention)
- **空間注意力**: 學習圖結構中的節點相似性
- **時間注意力**: 處理時序依賴
- **多跳信息**: 通過注意力機制聚合遠距離鄰居信息

### Bagging 策略
- **樣本策略**: 有放回抽樣
- **類別保持**: 每個袋子保持原始類別分布
- **集成方式**: 軟投票 (概率平均)

### Hyperopt 配置
- **優化器**: TPE (Tree Parzen Estimator)
- **評估次數**: 30 次迭代
- **目標**: 最大化測試 F1 分數

## 依賴套件

```
torch>=2.9.1
torch-geometric>=2.7.0
hyperopt>=0.2.7
networkx>=2.8.0
scikit-learn>=1.7.2
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 注意事項

1. **計算資源**: Hyperopt 優化可能需要較長時間，建議使用 GPU
2. **數據要求**: 需要 `../Dataset/` 目錄中的 Elliptic 數據集
3. **輸出目錄**: 自動創建 `results/` 和 `results/explanations/` 目錄

## 擴展性

該框架易於擴展：
- 添加新的 GNN 模型
- 修改評估指標
- 調整 Hyperopt 搜索空間
- 集成其他解釋方法
