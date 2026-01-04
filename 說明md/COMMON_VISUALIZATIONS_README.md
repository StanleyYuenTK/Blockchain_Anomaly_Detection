# 常見可視化圖表功能說明

本文檔說明 `visualization_tools.py` 中新增的常見數據可視化圖表功能。

## 新增功能

已添加以下常見的可視化圖表類型：

### 1. **柱狀圖 (Bar Chart)**
- 函數：`plot_bar_chart()`
- 用途：比較不同類別的數值
- 示例：
```python
from visualization_tools import plot_bar_chart

# 使用字典
data = {'GCN': 0.75, 'GAT': 0.78, 'Hybrid': 0.85}
plot_bar_chart(data, None, "模型性能比較", "模型", "F1 Score", 
               save_path='bar_chart.png')

# 使用列表
values = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
plot_bar_chart(values, labels, "示例柱狀圖")
```

### 2. **餅圖 (Pie Chart)**
- 函數：`plot_pie_chart()`
- 用途：顯示比例分佈
- 示例：
```python
from visualization_tools import plot_pie_chart

data = {'合法交易': 70, '非法交易': 25, '未知': 5}
plot_pie_chart(data, None, "交易類別分佈", save_path='pie_chart.png')
```

### 3. **散點數據分佈圖 (Scatter Plot)**
- 函數：`plot_scatter_distribution()`
- 用途：顯示兩個變量之間的關係和數據分佈
- 示例：
```python
from visualization_tools import plot_scatter_distribution
import numpy as np

x = np.random.randn(100)
y = np.random.randn(100)
labels = np.random.choice([0, 1], 100)

plot_scatter_distribution(x, y, labels, "數據分佈散點圖",
                         "特徵 1", "特徵 2", save_path='scatter.png')
```

### 4. **直方圖 (Histogram)**
- 函數：`plot_histogram()`
- 用途：顯示數值數據的分佈
- 示例：
```python
from visualization_tools import plot_histogram
import numpy as np

data = np.random.normal(100, 15, 1000)
plot_histogram(data, bins=30, title="交易金額分佈直方圖",
              xlabel="金額", ylabel="頻率", save_path='histogram.png')
```

### 5. **箱線圖 (Box Plot)**
- 函數：`plot_boxplot()`
- 用途：顯示數據的分佈、中位數、四分位數和異常值
- 示例：
```python
from visualization_tools import plot_boxplot
import numpy as np

data = [
    np.random.normal(100, 10, 100),
    np.random.normal(150, 15, 100),
    np.random.normal(120, 12, 100)
]
plot_boxplot(data, ['合法', '非法', '未知'],
            "不同類別交易金額分佈箱線圖", "金額", save_path='boxplot.png')
```

### 6. **熱力圖 (Heatmap)**
- 函數：`plot_heatmap()`
- 用途：顯示矩陣數據的相關性或其他關係
- 示例：
```python
from visualization_tools import plot_heatmap
import numpy as np

# 相關性矩陣
correlation = np.random.rand(5, 5)
correlation = (correlation + correlation.T) / 2
np.fill_diagonal(correlation, 1.0)

plot_heatmap(correlation,
            row_labels=['特徵1', '特徵2', '特徵3', '特徵4', '特徵5'],
            col_labels=['特徵1', '特徵2', '特徵3', '特徵4', '特徵5'],
            title="特徵相關性熱力圖", save_path='heatmap.png')
```

### 7. **折線圖 (Line Chart)**
- 函數：`plot_line_chart()`
- 用途：顯示數據隨時間或其他連續變量的變化趨勢
- 示例：
```python
from visualization_tools import plot_line_chart
import numpy as np

epochs = np.arange(1, 101)
train_loss = 0.5 * np.exp(-epochs/50) + 0.1
val_loss = 0.6 * np.exp(-epochs/50) + 0.15

plot_line_chart(epochs, [train_loss, val_loss], 
               ['訓練損失', '驗證損失'],
               "訓練損失曲線", "Epoch", "Loss", save_path='line_chart.png')
```

### 8. **類別分佈圖 (Class Distribution)**
- 函數：`plot_class_distribution()`
- 用途：同時顯示柱狀圖和餅圖，展示類別分佈
- 示例：
```python
from visualization_tools import plot_class_distribution
import numpy as np

labels = np.random.choice([0, 1, -1], size=1000, p=[0.7, 0.25, 0.05])
plot_class_distribution(labels, "交易類別分佈",
                       save_path='class_distribution.png', normalize=True)
```

## 快速生成所有示例圖表

運行以下命令可一次性生成所有常見可視化圖表的示例：

```python
from visualization_tools import generate_common_visualizations

# 生成所有示例圖表到 'visualizations/common' 目錄
generate_common_visualizations()
```

或者在命令行中：

```bash
python visualization_tools.py
```

這將在 `visualizations/common/` 目錄中生成：
- `bar_chart_example.png` - 柱狀圖示例
- `pie_chart_example.png` - 餅圖示例
- `scatter_distribution_example.png` - 散點圖示例
- `histogram_example.png` - 直方圖示例
- `boxplot_example.png` - 箱線圖示例
- `heatmap_example.png` - 熱力圖示例
- `line_chart_example.png` - 折線圖示例
- `class_distribution_example.png` - 類別分佈圖示例

## 函數參數說明

所有函數都支持以下通用參數：

- `save_path`: 圖表保存路徑（可選，默認保存在當前目錄）
- `figsize`: 圖表大小，格式為 `(寬度, 高度)`（可選）
- `title`: 圖表標題（可選）

每個函數還有其特定的參數，詳細說明請查看函數文檔字符串。

## 使用場景

### 區塊鏈交易數據分析

1. **交易類別分佈**：使用餅圖或類別分佈圖
2. **模型性能比較**：使用柱狀圖
3. **特徵分佈分析**：使用直方圖、箱線圖
4. **特徵相關性**：使用熱力圖、散點圖
5. **訓練過程監控**：使用折線圖

## 完整示例

```python
from visualization_tools import *
import numpy as np

# 1. 模型性能柱狀圖
model_performance = {
    'GCN': 0.7523,
    'GAT': 0.7812,
    'GraphSAGE': 0.7634,
    'SGAT': 0.8234,
    'Hybrid': 0.8545
}
plot_bar_chart(model_performance, None, 
              "模型性能比較 (F1 Score)", "模型", "F1 Score",
              save_path='model_performance_bar.png')

# 2. 交易類別餅圖
class_distribution = {
    '合法交易': 70.5,
    '非法交易': 24.3,
    '未知': 5.2
}
plot_pie_chart(class_distribution, None, 
              "交易類別分佈", save_path='class_pie.png')

# 3. 特徵相關性熱力圖
features = np.random.randn(100, 5)
correlation_matrix = np.corrcoef(features.T)
plot_heatmap(correlation_matrix,
            row_labels=[f'特徵{i+1}' for i in range(5)],
            col_labels=[f'特徵{i+1}' for i in range(5)],
            title="特徵相關性熱力圖",
            save_path='correlation_heatmap.png')

# 4. 訓練曲線折線圖
epochs = np.arange(1, 101)
train_acc = 0.3 + 0.5 * (1 - np.exp(-epochs/30))
val_acc = train_acc - 0.05

plot_line_chart(epochs, [train_acc, val_acc],
               ['訓練準確率', '驗證準確率'],
               "訓練準確率曲線", "Epoch", "Accuracy",
               save_path='accuracy_curve.png')
```

## 注意事項

1. **中文字體**：確保系統安裝了中文字體（如 SimHei、Microsoft YaHei）
2. **圖表分辨率**：默認保存為 300 DPI，可在 `savefig()` 中調整
3. **數據格式**：
   - 支持 NumPy 數組
   - 支持 PyTorch Tensor（會自動轉換）
   - 支持 Python 列表和字典
4. **顏色方案**：使用 Seaborn 的調色板，可自定義

## 依賴項

確保安裝以下 Python 包：

```bash
pip install matplotlib seaborn numpy
```

如果使用 PyTorch 相關功能：

```bash
pip install torch
```

## 與現有功能的整合

所有新的可視化函數都可以與現有的模型訓練和評估流程整合：

```python
from visualization_tools import plot_class_distribution, plot_bar_chart
from gnn_models import load_elliptic_data

# 加載數據
data = load_elliptic_data()

# 繪製類別分佈
plot_class_distribution(data.y.numpy(), "數據集類別分佈",
                       save_path='dataset_distribution.png')
```

這些新功能大大擴展了數據可視化的能力，使得分析結果更加直觀和易於理解！

