#!/usr/bin/env python3
"""
Simple test script to check if our modifications work correctly
"""

def test_imports():
    """Test if all required imports work"""
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP
        print("✓ PyTorch Geometric imports successful")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
        return False

    try:
        # Test our custom classes
        from gnn_interim_version import FocalLoss, SpatialTemporalAttention, STAModel
        print("✓ Custom classes import successful")
    except ImportError as e:
        print(f"✗ Custom classes import failed: {e}")
        return False

    print("✓ All imports successful!")
    return True

def test_focal_loss():
    """Test FocalLoss implementation"""
    try:
        import torch
        from gnn_interim_version import FocalLoss

        criterion = FocalLoss(alpha=1, gamma=2)
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))

        loss = criterion(inputs, targets)
        print(f"✓ FocalLoss test successful, loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ FocalLoss test failed: {e}")
        return False

def test_spatial_temporal_attention():
    """Test SpatialTemporalAttention implementation"""
    try:
        import torch
        from gnn_interim_version import SpatialTemporalAttention

        model = SpatialTemporalAttention(in_channels=16, hidden_channels=32, num_heads=4)
        x = torch.randn(10, 16)  # 10 nodes, 16 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Simple triangle

        output = model(x, edge_index)
        expected_shape = (10, 32)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

        print(f"✓ SpatialTemporalAttention test successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ SpatialTemporalAttention test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing GNN Interim Version modifications...")
    print("=" * 50)

    success = True
    success &= test_imports()
    success &= test_focal_loss()
    success &= test_spatial_temporal_attention()

    print("=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
