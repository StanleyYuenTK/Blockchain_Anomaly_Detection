#!/usr/bin/env python3
"""
Test script to verify the updated load_elliptic_data function
"""

from interim_no_STA_ver import load_elliptic_data
import torch

def test_data_loading():
    print("Testing updated data loading function...")

    # Load data
    data = load_elliptic_data()

    # Basic checks
    print(f"\nBasic statistics:")
    print(f"Features shape: {data.x.shape}")
    print(f"Labels shape: {data.y.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Timesteps range: {data.timesteps.min().item()}-{data.timesteps.max().item()}")

    # Check feature count (should be 166)
    expected_features = 166
    actual_features = data.x.size(1)
    print(f"\nFeature count check:")
    print(f"Expected: {expected_features} features")
    print(f"Actual: {actual_features} features")
    if actual_features == expected_features:
        print("✅ Feature count matches expected value")
    else:
        print("❌ Feature count does not match!")

    # Check label distribution
    total_nodes = len(data.y)
    legal_count = (data.y == 0).sum().item()
    illicit_count = (data.y == 1).sum().item()
    unknown_count = (data.y == -1).sum().item()

    print(f"\nLabel distribution verification:")
    print(f"Total nodes: {total_nodes}")
    print(f"Legal (0): {legal_count} ({legal_count/total_nodes*100:.1f}%)")
    print(f"Illicit (1): {illicit_count} ({illicit_count/total_nodes*100:.1f}%)")
    print(f"Unknown (-1): {unknown_count} ({unknown_count/total_nodes*100:.1f}%)")

    # Expected from dataset description: 21% legal, 2% illicit
    expected_legal_pct = 21.0
    expected_illicit_pct = 2.0

    if abs(legal_count/total_nodes*100 - expected_legal_pct) < 1.0:
        print("✅ Legal percentage matches expected ~21%")
    else:
        print("❌ Legal percentage does not match expected!")

    if abs(illicit_count/total_nodes*100 - expected_illicit_pct) < 0.5:
        print("✅ Illicit percentage matches expected ~2%")
    else:
        print("❌ Illicit percentage does not match expected!")

    # Check data splits
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()

    print(f"\nData split verification:")
    print(f"Train: {train_count} nodes")
    print(f"Validation: {val_count} nodes")
    print(f"Test: {test_count} nodes")
    print(f"Total known labels: {train_count + val_count + test_count}")

    # Check that unknown nodes are excluded from splits
    known_in_splits = train_count + val_count + test_count
    total_known = total_nodes - unknown_count
    if known_in_splits == total_known:
        print("✅ All known labels are included in train/val/test splits")
    else:
        print("❌ Mismatch in known label counts!")

    print("\nTest completed!")

if __name__ == "__main__":
    test_data_loading()
