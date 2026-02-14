#!/usr/bin/env python3
"""
測試 PGExplainer 實現的簡單腳本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.explain import PGExplainer

# 簡單的 GNN 模型
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def test_pg_explainer():
    """測試 PGExplainer 的基本功能"""
    print("=== 測試 PGExplainer 實現 ===")

    # 創建簡單的測試數據
    num_nodes = 100
    num_edges = 300
    in_channels = 10
    hidden_channels = 32
    out_channels = 2

    # 隨機生成數據
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.rand(num_nodes) < 0.6
    data.val_mask = (torch.rand(num_nodes) < 0.2) & (~data.train_mask)
    data.test_mask = ~(data.train_mask | data.val_mask)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SimpleGCN(in_channels, hidden_channels, out_channels).to(device)
    data = data.to(device)

    # 訓練模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    print("訓練模型...")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 測試 PGExplainer
    print("\n測試 PGExplainer...")
    try:
        pg_explainer = PGExplainer(
            model=model,
            in_channels=in_channels,
            device=device,
            epochs=10,  # 減少訓練時間用於測試
            lr=0.003,
            num_hops=2,
            batch_size=32
        )

        # 訓練 PGExplainer
        train_indices = torch.where(data.train_mask)[0]
        pg_explainer.train_explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=data.y,
            index=train_indices[:20]  # 只用前20個樣本測試
        )

        # 生成解釋
        test_node = torch.where(data.test_mask)[0][0]  # 選擇第一個測試節點
        explanation = pg_explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=data.y,
            index=test_node
        )

        print(f"✅ PGExplainer 測試成功!")
        print(f"解釋的節點: {test_node.item()}")
        print(f"邊重要性形狀: {explanation.edge_mask.shape}")
        print(f"邊重要性範圍: [{explanation.edge_mask.min().item():.4f}, {explanation.edge_mask.max().item():.4f}]")

    except Exception as e:
        print(f"❌ PGExplainer 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pg_explainer()
