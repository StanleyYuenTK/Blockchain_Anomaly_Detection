"""
未來研究方法原型實作：

1. 可搜索加密 + 同態加密環境下的頻繁項集 / 關聯規則挖掘
2. 動態圖與異質圖建模
3. 克服遺失交易連結（missing links）的影響
4. 考慮時序特徵的異常檢測
5. 結合豐富的時序與交易特徵
6. 交易場景專用的數據增強
7. 圖增強（Graph Augmentation）
8. 嚴重類別不平衡下的學習策略

說明：
這個檔案不是完整可用的系統，而是為論文 / 專題準備的「方法原型與接口設計」，
方便之後逐步填入真正的密碼學與模型實作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
except ImportError:  # pragma: no cover - 方便在無 PyTorch 環境下閱讀
    torch = None  # type: ignore
    nn = object  # type: ignore
    Dataset = object  # type: ignore
    Data = object  # type: ignore


# =============================================================================
# 1. 可搜索加密 + 同態加密下的頻繁項集 / 關聯規則挖掘
# =============================================================================


class EncryptedPatternMiner:
    """
    在「密文域」上進行頻繁項集與關聯規則挖掘的原型接口。

    真正部署時，需替換為實際的 searchable encryption / homomorphic encryption 函式庫。
    這裡只提供：
      - 介面設計
      - 演算法流程的「明文版本」實作，方便先做實驗 / ablation
    """

    def __init__(self) -> None:
        # TODO: 接上真正的加密函式庫，如：PySEAL、HElib 的 Python 綁定等
        self.crypto_backend: Optional[Any] = None

    # ----------------------------- 明文版雛形 ----------------------------- #

    def mine_frequent_itemsets_plain(
        self,
        transactions: List[List[str]],
        min_support: float = 0.05,
    ) -> Dict[frozenset, float]:
        """
        在明文條件下的頻繁項集挖掘（簡化版 Apriori），作為「密文版」的對照基線。
        """
        num_tx = len(transactions)
        item_counts: Dict[str, int] = {}
        for tx in transactions:
            for item in set(tx):
                item_counts[item] = item_counts.get(item, 0) + 1

        # 1-項集
        frequent: Dict[frozenset, float] = {}
        for item, cnt in item_counts.items():
            support = cnt / num_tx
            if support >= min_support:
                frequent[frozenset([item])] = support

        # 2-項集（示意，未實作完整 k-項集遞迴）
        items = list(item_counts.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                cnt = 0
                for tx in transactions:
                    s = set(tx)
                    if a in s and b in s:
                        cnt += 1
                support = cnt / num_tx
                if support >= min_support:
                    frequent[frozenset([a, b])] = support

        return frequent

    # -------------------------- 密文版接口設計 --------------------------- #

    def encrypt_transactions(self, transactions: List[List[str]]) -> Any:
        """
        將交易資料加密。

        回傳值的具體型別，將依賴實際採用的同態 / 可搜索加密函式庫。
        """
        raise NotImplementedError(
            "此為接口設計，實際部署時請接入同態 / 可搜索加密函式庫。"
        )

    def encrypted_frequency_count(self, encrypted_tx: Any, encrypted_query: Any) -> Any:
        """
        在密文上計算「出現次數」的同態操作接口。
        """
        raise NotImplementedError(
            "請在此實作同態加法 / 乘法，用以計算頻繁模式的支持度。"
        )


# =============================================================================
# 2. 動態圖與異質圖建模
# =============================================================================


def build_dynamic_snapshots(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    timestep_col: str = "timestep",
    tx_id_col: str = "txId",
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    根據時間步 (timestep) 將 Elliptic 類型的交易數據切成「動態快照」。

    回傳：
        {t: {"nodes": nodes_df_at_t, "edges": edges_df_involving_t_nodes}}
    """
    snapshots: Dict[int, Dict[str, pd.DataFrame]] = {}

    if timestep_col not in nodes_df.columns:
        raise ValueError(f"nodes_df 缺少 timestep 欄位：{timestep_col}")

    timesteps = sorted(nodes_df[timestep_col].unique())
    for t in timesteps:
        nodes_t = nodes_df[nodes_df[timestep_col] == t].copy()
        tx_ids = set(nodes_t[tx_id_col].values)
        edges_t = edges_df[
            (edges_df.iloc[:, 0].isin(tx_ids)) & (edges_df.iloc[:, 1].isin(tx_ids))
        ].copy()
        snapshots[int(t)] = {"nodes": nodes_t, "edges": edges_t}

    return snapshots


# =============================================================================
# 3. 克服遺失交易連結（missing links）
# =============================================================================


def heuristic_missing_edge_completion(
    data: "Data",
    top_k: int = 5,
) -> "Data":
    """
    非嚴格的「連結補全」雛形：
    - 使用節點特徵的餘弦相似度，為每個節點補上 top_k 個相似鄰居。
    - 真正研究可改為：GNN-based link prediction / matrix completion / knowledge graph completion。

    注意：這裡只是一個簡化示範，避免直接在真實實驗裡使用。
    """
    if torch is None or not isinstance(data, Data):
        raise RuntimeError("需要安裝 torch 與 torch_geometric 才能使用此功能。")

    x = data.x  # [N, F]
    N = x.size(0)

    # 正規化後做餘弦相似度
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    sim = x_norm @ x_norm.t()  # [N, N]
    sim.fill_diagonal_(-1.0)  # 不考慮自己

    # 為每個節點選 top_k 相似節點
    _, topk_idx = torch.topk(sim, k=min(top_k, N - 1), dim=1)

    new_edges = []
    for i in range(N):
        for j in topk_idx[i]:
            new_edges.append([i, int(j)])

    if len(new_edges) == 0:
        return data

    new_edges = torch.tensor(new_edges, dtype=torch.long, device=x.device).t()
    merged_edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    merged_edge_index = to_undirected(merged_edge_index)

    data.edge_index = merged_edge_index
    return data


# =============================================================================
# 4. 考慮時序特徵的異常檢測（時間序列 / RNN）
# =============================================================================


class TemporalAnomalyDetector(nn.Module):  # type: ignore[misc]
    """
    利用 GRU 進行交易序列的異常檢測雛形。

    可以重用 / 對照你在 `advanced_gnn_models.py` 中的 GRU-MHA + Conv1D，
    這裡則專注在「單純時序」層面。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        if torch is None:
            raise RuntimeError("需要安裝 torch 才能使用 TemporalAnomalyDetector。")

        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, T, F] 交易時序特徵

        Returns:
            scores: [B, T] 每個時間步的「異常分數」，數值越大代表越可疑。
        """
        h, _ = self.gru(seq)  # [B, T, 2H]
        scores = self.scorer(h).squeeze(-1)
        return scores


# =============================================================================
# 5. 結合豐富的時序與交易特徵
# =============================================================================


def build_rich_temporal_features(
    tx_df: pd.DataFrame,
    addr_col: str = "address",
    ts_col: str = "timestamp",
    amount_col: str = "amount",
    window: str = "1D",
) -> pd.DataFrame:
    """
    針對地址 / 節點，計算「一段時間內」的統計特徵：
      - 交易頻率
      - 交易金額總和 / 平均值
      - 入站 / 出站比（如果有 direction 資訊，可擴充）

    回傳一個可以與圖節點特徵 join 的 DataFrame。
    """
    if ts_col not in tx_df.columns:
        raise ValueError(f"tx_df 缺少時間欄位：{ts_col}")

    df = tx_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.set_index(ts_col, inplace=True)

    grouped = (
        df.groupby(addr_col)
        .resample(window)
        .agg(
            tx_count=(amount_col, "size"),
            tx_sum=(amount_col, "sum"),
            tx_mean=(amount_col, "mean"),
        )
        .reset_index()
    )

    # 填補 NaN
    grouped[["tx_sum", "tx_mean"]] = grouped[["tx_sum", "tx_mean"]].fillna(0.0)
    return grouped


# =============================================================================
# 6. 交易情境專用的數據增強
# =============================================================================


def augment_transaction_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    noise_level: float = 0.01,
    minority_label: int = 1,
    ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    針對「非法交易」（少數類）做簡單的數據增強：
      - 對少數類樣本加高斯噪音，產生新的序列
      - 可以視為最簡版的 time-series SMOTE

    Args:
        sequences: [N, T, F]
        labels: [N]
        noise_level: 噪音強度
        minority_label: 少數類標籤（預設 1 代表非法）
        ratio: 增強後新增樣本數 / 原少數類樣本數
    """
    minority_idx = np.where(labels == minority_label)[0]
    if len(minority_idx) == 0:
        return sequences, labels

    num_new = int(len(minority_idx) * ratio)
    chosen = np.random.choice(minority_idx, size=num_new, replace=True)

    new_seq = sequences[chosen] + noise_level * np.random.randn(
        num_new, *sequences.shape[1:]
    )
    new_labels = np.full(num_new, minority_label, dtype=labels.dtype)

    aug_seq = np.concatenate([sequences, new_seq], axis=0)
    aug_labels = np.concatenate([labels, new_labels], axis=0)
    return aug_seq, aug_labels


# =============================================================================
# 7. 圖增強（Graph Augmentation）
# =============================================================================


def simple_graph_augmentation(
    data: "Data",
    edge_drop_rate: float = 0.1,
    feature_noise_level: float = 0.01,
) -> "Data":
    """
    對整張圖做啟發式增強：
      - 隨機刪除部分邊（edge dropping）
      - 對節點特徵加少量噪音

    真正研究可以擴充為 GAPLG 之類的圖增強框架。
    """
    if torch is None or not isinstance(data, Data):
        raise RuntimeError("需要安裝 torch 與 torch_geometric 才能使用此功能。")

    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > edge_drop_rate
    new_edge_index = edge_index[:, keep_mask]

    new_x = data.x + feature_noise_level * torch.randn_like(data.x)

    aug = Data(
        x=new_x,
        y=data.y,
        edge_index=new_edge_index,
    )
    for attr in ["train_mask", "val_mask", "test_mask", "timesteps"]:
        if hasattr(data, attr):
            setattr(aug, attr, getattr(data, attr))
    return aug


# =============================================================================
# 8. 嚴重類別不平衡下的學習策略
# =============================================================================


def compute_class_weights_from_labels(
    labels: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    根據類別頻率計算 class weights，用於加權損失或抽樣。

    Args:
        labels: 一維標籤陣列
        smoothing: 平滑項，避免極端比率造成權重爆炸
    """
    values, counts = np.unique(labels, return_counts=True)
    freqs = counts / counts.sum()
    inv = 1.0 / (freqs + smoothing)
    inv /= inv.sum()

    weights = np.zeros_like(labels, dtype=float)
    for v, w in zip(values, inv):
        weights[labels == v] = w
    return weights


class FocalLoss(nn.Module):  # type: ignore[misc]
    """
    用於嚴重不平衡分類的 Focal Loss。

    gamma > 0 時，會降低「容易樣本」對損失的貢獻，讓模型專注少數類 / 困難樣本。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        if torch is None:
            raise RuntimeError("需要安裝 torch 才能使用 FocalLoss。")
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] 未經 softmax 的輸出
            targets: [N] 類別索引
        """
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )  # [N]
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


__all__ = [
    "EncryptedPatternMiner",
    "build_dynamic_snapshots",
    "heuristic_missing_edge_completion",
    "TemporalAnomalyDetector",
    "build_rich_temporal_features",
    "augment_transaction_sequences",
    "simple_graph_augmentation",
    "compute_class_weights_from_labels",
    "FocalLoss",
]



