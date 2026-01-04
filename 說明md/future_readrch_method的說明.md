[2 tools called]

我已經幫你新增一個 `future_research_methods.py`，把你列出的 8 點研究方向整理成可直接引用的「方法原型」與接口。重點如下：

### 1. **密文上的頻繁項集 / 關聯規則挖掘**
- 檔案內的 `EncryptedPatternMiner`：
  - `mine_frequent_itemsets_plain(...)`：明文環境下的簡化頻繁項集挖掘（作為 baseline）。
  - `encrypt_transactions(...)`、`encrypted_frequency_count(...)`：預留 **searchable encryption / homomorphic encryption** 的接口，方便之後接真正的密碼學函式庫。

### 2. **動態圖與異質圖**
- `build_dynamic_snapshots(nodes_df, edges_df, ...)`  
  根據 `timestep` 把交易圖切成多個時間快照：`{t: {"nodes": ..., "edges": ...}}`，對應你提到的「動態圖深入探索」。

### 3. **克服遺失交易連結**
- `heuristic_missing_edge_completion(data, top_k=5)`  
  用節點特徵的餘弦相似度，為每個節點補上 top‑k 相似鄰居邊，作為 **missing links 補全** 的啟發式原型（真正實驗可以換成 GNN link prediction）。

### 4. **時序特徵異常檢測**
- `TemporalAnomalyDetector`（GRU 為主）：
  - `forward(seq)` 接收 `[B, T, F]` 時序特徵，輸出每個時間步的異常分數，對應你提到的 **GRU / 時序異常檢測** 方向。

### 5. **豐富時序與交易特徵**
- `build_rich_temporal_features(tx_df, window="1D", ...)`  
  以地址為單位，在時間窗內計算：
  - 交易筆數、金額總和、平均金額等統計量，可之後與圖節點特徵 join，當作額外輸入訊號。

### 6. **交易場景專用數據增強**
- `augment_transaction_sequences(sequences, labels, ...)`  
  對少數類（非法交易）時序樣本加噪音生成新樣本，作為簡易版 **time-series SMOTE / data augmentation**。

### 7. **圖增強（Graph Augmentation）**
- `simple_graph_augmentation(data, edge_drop_rate, feature_noise_level)`  
  對整張圖做：
  - 隨機刪邊（edge dropping）
  - 對節點特徵加噪音  
  對應你提到的「在建構圖之後直接做啟發式圖增強」。

### 8. **嚴重類別不平衡**
- `compute_class_weights_from_labels(labels, ...)`：依標籤頻率算 class weights。
- `FocalLoss`：可直接在訓練時替換 CrossEntropy，提升 **少數類（非法交易）** 的辨識能力。

你可以在其他檔案中這樣使用例如：

```python
from future_research_methods import (
    EncryptedPatternMiner,
    build_dynamic_snapshots,
    TemporalAnomalyDetector,
    simple_graph_augmentation,
    FocalLoss,
)
```

如果你希望，我也可以幫你把其中某一塊（例如 FocalLoss + 圖增強）直接整合進 `advanced_gnn_models.py` 的訓練流程。