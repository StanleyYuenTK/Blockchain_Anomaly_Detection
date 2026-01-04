# 可視化工具使用說明

本目錄包含用於生成 GNN 模型可視化圖表的工具。

## 文件說明

- `visualization_tools.py`: 核心可視化工具庫
- `generate_plots.py`: 自動生成所有圖表的腳本

## 快速開始

### 生成所有圖表

運行以下命令即可生成所有可視化圖表：

```bash
python generate_plots.py
```

這將在 `visualizations/` 目錄中生成以下圖表：

1. **模型架構圖** (5個)
   - `gcn_architecture.png`
   - `gat_architecture.png`
   - `graphsage_architecture.png`
   - `sgat_architecture.png`
   - `hybrid_architecture.png`

2. **架構對比圖** (1個)
   - `architecture_comparison.png`

3. **訓練曲線圖** (5個)
   - `gcn_training_curves.png`
   - `gat_training_curves.png`
   - `graphsage_training_curves.png`
   - `sgat_training_curves.png`
   - `hybrid_training_curves.png`

4. **模型性能比較圖** (1個)
   - `model_comparison.png`

5. **混淆矩陣圖** (5個)
   - `gcn_confusion_matrix.png`
   - `gat_confusion_matrix.png`
   - `graphsage_confusion_matrix.png`
   - `sgat_confusion_matrix.png`
   - `hybrid_confusion_matrix.png`

## 單獨使用可視化函數

### 1. 生成模型架構圖

```python
from visualization_tools import plot_model_architecture

# 生成單個模型的架構圖
plot_model_architecture('GCN', 'gcn_architecture.png')
plot_model_architecture('Hybrid', 'hybrid_architecture.png')
```

### 2. 繪製訓練曲線

```python
from visualization_tools import plot_training_curves, TrainingHistory

# 創建訓練歷史記錄
history = TrainingHistory()
history.add_epoch(1, 0.5, 0.6, 0.7, 0.75, 0.68, 0.73)
history.add_epoch(2, 0.4, 0.5, 0.75, 0.78, 0.73, 0.76)
# ... 添加更多 epoch

# 繪製訓練曲線
plot_training_curves(history, 'training_curves.png', 'GCN')
```

### 3. 模型性能比較

```python
from visualization_tools import plot_model_comparison

results = {
    'GCN': {'val_f1': 0.75, 'test_f1': 0.72, 'val_acc': 0.78, 'test_acc': 0.75},
    'GAT': {'val_f1': 0.78, 'test_f1': 0.75, 'val_acc': 0.81, 'test_acc': 0.78},
    'Hybrid': {'val_f1': 0.85, 'test_f1': 0.82, 'val_acc': 0.88, 'test_acc': 0.85}
}

plot_model_comparison(results, 'comparison.png')
```

### 4. 混淆矩陣

```python
from visualization_tools import plot_confusion_matrix
import numpy as np

y_true = np.array([0, 0, 1, 1, 0, 1, ...])
y_pred = np.array([0, 1, 1, 1, 0, 0, ...])

plot_confusion_matrix(y_true, y_pred, 'GCN', 'confusion_matrix.png')
```

### 5. 特徵可視化

```python
from visualization_tools import plot_feature_visualization_2d
import numpy as np

# 2D 特徵矩陣
features_2d = np.random.randn(1000, 2)
labels = np.random.choice([0, 1], size=1000)

plot_feature_visualization_2d(features_2d, labels, 
                             '特徵可視化', 'PCA', 'features_pca.png')
```

## 在訓練過程中記錄歷史

修改 `gnn_models.py` 或 `advanced_gnn_models.py` 的訓練函數：

```python
from visualization_tools import TrainingHistory

def train_and_evaluate(...):
    # ... 模型初始化 ...
    
    history = TrainingHistory()
    
    for epoch in range(1, epochs + 1):
        # 訓練
        loss = train_model(model, data, optimizer, criterion, device)
        
        # 評估
        (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc) = evaluate_model(...)
        
        # 記錄歷史
        history.add_epoch(epoch, loss, val_loss, val_f1, val_acc, test_f1, test_acc)
        
        # ... 其他代碼 ...
    
    # 保存歷史並生成圖表
    history.save('training_history.json')
    plot_training_curves(history, 'training_curves.png', model_name)
    
    return best_epoch_stats
```

## 依賴項

確保安裝以下 Python 包：

```bash
pip install matplotlib seaborn numpy scikit-learn
```

如果使用 PyTorch 相關功能，還需要：

```bash
pip install torch
```

## 圖表說明

### 模型架構圖
- 顯示模型的層次結構
- 不同顏色代表不同類型的層（輸入、卷積、注意力、輸出等）
- 箭頭表示數據流向

### 訓練曲線
- **損失曲線**: 顯示訓練和驗證損失隨時間的變化
- **F1 分數曲線**: 顯示驗證和測試 F1 分數
- **準確率曲線**: 顯示驗證和測試準確率
- **綜合性能**: 同時顯示 F1 和準確率

### 模型性能比較
- 並排比較不同模型的性能指標
- 包括驗證/測試 F1 分數和準確率

### 混淆矩陣
- 顯示分類結果的詳細對照
- 幫助理解模型的錯誤類型

### 特徵可視化
- 使用 PCA 或 t-SNE 將高維特徵降維到 2D
- 可視化不同類別節點的分佈

## 自定義

所有函數都支持自定義參數，可以調整：
- 圖表大小 (`figsize`)
- 顏色方案 (`colors`, `palette`)
- 保存路徑 (`save_path`)
- 標題和標籤

查看 `visualization_tools.py` 中的函數文檔以獲取更多詳細信息。

## 注意事項

1. 中文字體：如果圖表中文字顯示為方框，請確保系統安裝了中文字體（如 SimHei、Microsoft YaHei）
2. 圖表分辨率：默認保存為 300 DPI，可在 `savefig()` 中調整
3. 文件格式：默認保存為 PNG，可改為 PDF、SVG 等格式

## 問題排查

如果遇到問題：

1. **ImportError**: 確保所有依賴項已安裝
2. **字體問題**: 檢查系統是否支持中文字體
3. **路徑問題**: 確保輸出目錄存在或具有寫入權限

## 示例輸出

運行 `generate_plots.py` 後，您將在 `visualizations/` 目錄中看到所有生成的圖表文件。

