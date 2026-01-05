import numpy as np
import torch

# Simulate the scenario
total_nodes = 216374
test_nodes = 8841

# Create a mock test_mask (216374 elements, 8841 True values)
test_mask = torch.zeros(total_nodes, dtype=torch.bool)
test_mask[:test_nodes] = True  # Just set first 8841 as test nodes for simplicity
print(f"test_mask sum: {test_mask.sum().item()}")

# Create mock predictions (216374 predictions)
all_predictions = np.random.randint(0, 2, size=total_nodes)
print(f"all_predictions shape: {all_predictions.shape}")

# Test boolean indexing
test_mask_np = test_mask.cpu().numpy()
print(f"test_mask_np sum: {test_mask_np.sum()}")

test_y_pred_bool = all_predictions[test_mask_np]
print(f"test_y_pred_bool shape: {test_y_pred_bool.shape}")

# Test integer indexing
test_indices = torch.where(test_mask)[0].cpu().numpy()
print(f"test_indices shape: {test_indices.shape}")

test_y_pred_int = all_predictions[test_indices]
print(f"test_y_pred_int shape: {test_y_pred_int.shape}")

# Check if they are the same
print(f"Results equal: {np.array_equal(test_y_pred_bool, test_y_pred_int)}")