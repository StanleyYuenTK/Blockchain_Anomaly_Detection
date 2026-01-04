import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import argparse
from data_loader import load_graph_data
from gcn_model import GCN

def train(model, data, optimizer, criterion):
    """
    Trains the GCN model for one epoch.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    """
    Evaluates the GCN model on the validation and test sets.
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # Validation
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
    val_f1 = f1_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu(), average='binary', pos_label=1)
    val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())

    # Test
    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
    test_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='binary', pos_label=1)
    test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

    return (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc)

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='GCN training for Elliptic dataset')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels in the GCN.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer.')
    args = parser.parse_args()

    # --- Data and Model Setup ---
    data = load_graph_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=data.num_node_features, hidden_channels=args.hidden_channels, out_channels=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    # --- Training Loop ---
    best_val_f1 = 0
    best_epoch_stats = {}

    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, criterion)
        (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc) = test(model, data)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_stats = {
                'epoch': epoch,
                'loss': loss,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_f1': test_f1,
                'test_acc': test_acc
            }

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}')

    # --- Final Results ---
    print("\nTraining complete!")
    print(f"Best validation F1: {best_epoch_stats['val_f1']:.4f} at epoch {best_epoch_stats['epoch']}")
    print(f"Test F1: {best_epoch_stats['test_f1']:.4f}, Test Accuracy: {best_epoch_stats['test_acc']:.4f}")

if __name__ == '__main__':
    main()
