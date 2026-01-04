以下是三個核心檔案的整理說明，方便你快速安裝依賴、了解用途並執行：

## 1. `advanced_gnn_models.py`
- **用途**：整合 SGAT、GraphSAGE 編碼器、時序特徵提取（GRU+MHA+Conv1D）、半監督自訓練與集成學習，用於區塊鏈異常偵測。
- **需安裝套件**：
  - `torch`, `torchvision`, `torchaudio`
  - `torch-geometric`（需搭配對應的 PyTorch CUDA 版本）
  - `scikit-learn`, `pandas`, `numpy`
- **執行方式**（於專案根目錄）：
  1. `cd "4913_2025_bc-feat-data-preparation-script"`
  2. `python advanced_gnn_models.py --model Hybrid --epochs 100 --use_self_training`
     （可改 `--model SGAT` 或加上 `--use_ensemble`）

## 2. `future_research_methods.py`
- **用途**：整理未來研究方向的原型接口，包含：
  - 可搜索/同態加密環境下的頻繁項集挖掘介面
  - 動態/異質圖切片，遺失連結補全
  - 時序異常偵測模型、豐富時序特徵生成
  - 交易序列與圖增強、類別不平衡處理（Focal Loss）
- **需安裝套件**（若只閱讀不執行可略）：
  - `pandas`, `numpy`
  - 若要跑 Torch/Graph 範例則需 `torch`, `torch-geometric`
- **執行建議**：
  - 主要作為模組引用；可在互動式 Python 測試：  
    `python -i future_research_methods.py`
  - 或在其他腳本 `from future_research_methods import ...` 引用對應工具。

## 3. `gnn_models.py`
- **用途**：基礎 GCN / GAT / GraphSAGE 訓練與特徵降維（PCA、t-SNE）、特徵可視化與混淆矩陣等。
- **需安裝套件**：
  - `torch`, `torch-geometric`, `scikit-learn`, `pandas`, `numpy`
  - 若需繪圖：`matplotlib`
- **執行方式**：
  1. `cd "4913_2025_bc-feat-data-preparation-script"`
  2. `python gnn_models.py --model GAT --epochs 100 --reduction_method tsne`
     （可用 `--model GraphSAGE --no_extract_features` 來跳過特徵提取）

---

### 安裝依賴範例
```bash
pip install -r requirements.txt
pip install torch-geometric   # 需依照官網指示選擇對應版本
```

如需補充圖表，可先確保 `matplotlib seaborn` 安裝完成。