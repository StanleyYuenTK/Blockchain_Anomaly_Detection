"""Use pygad.torchga to optimize a PyG MixHopConv model.

Example:
    python pygad_torchga_mixhop.py --dataset e1 --generations 30 --sol-per-pop 12
"""

import argparse
import random

import numpy as np
import pygad
import pygad.torchga
import torch
from sklearn.metrics import f1_score

import dataset_zoo
import gnn_zoo

RANDOM_SEED = 24027277


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(dataset: str):
    if dataset == "e1":
        return dataset_zoo.load_elliptic_data()
    if dataset == "e2":
        return dataset_zoo.load_ethereum_data()
    raise ValueError(f"Unsupported dataset: {dataset}")


def model_logits(model, data):
    return model(data.x, data.edge_index)


def assign_weights(model, solution_vector):
    weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution_vector)
    model.load_state_dict(weights_dict)


def evaluate_macro_f1(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model_logits(model, data)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    return f1_score(y_true, preds, average="macro")


def run_torchga_mixhop(
    data,
    device,
    hidden_channels=128,
    dropout=0.5,
    powers="[0, 1, 2]",
    generations=30,
    sol_per_pop=12,
    num_parents_mating=6,
):
    params = {
        "in_channels": data.x.size(1),
        "out_channels": 2,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "powers": powers,
    }
    model = gnn_zoo.MixHop_Model(params).to(device)
    data = data.to(device)

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=sol_per_pop)

    def fitness_func(ga_instance, solution, sol_idx):
        assign_weights(model, solution)
        macro_f1 = evaluate_macro_f1(model, data, data.val_mask)
        # pygad maximizes fitness, keep it positive and smooth.
        return macro_f1 + 1e-8

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=num_parents_mating,
        initial_population=torch_ga.population_weights,
        fitness_func=fitness_func,
        mutation_percent_genes=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        random_seed=RANDOM_SEED,
        keep_parents=2,
        suppress_warnings=True,
    )

    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    assign_weights(model, best_solution)

    train_f1 = evaluate_macro_f1(model, data, data.train_mask)
    val_f1 = evaluate_macro_f1(model, data, data.val_mask)
    test_f1 = evaluate_macro_f1(model, data, data.test_mask)

    return {
        "best_val_macro_f1": float(best_fitness),
        "train_macro_f1": float(train_f1),
        "val_macro_f1": float(val_f1),
        "test_macro_f1": float(test_f1),
        "config": {
            "hidden_channels": hidden_channels,
            "dropout": dropout,
            "powers": powers,
            "generations": generations,
            "sol_per_pop": sol_per_pop,
            "num_parents_mating": num_parents_mating,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="pygad.torchga + PyG MixHopConv example")
    parser.add_argument("--dataset", type=str, default="e1", choices=["e1", "e2"])
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--powers", type=str, default="[0, 1, 2]")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--sol-per-pop", type=int, default=12)
    parser.add_argument("--num-parents-mating", type=int, default=6)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_data(args.dataset)
    result = run_torchga_mixhop(
        data=data,
        device=device,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        powers=args.powers,
        generations=args.generations,
        sol_per_pop=args.sol_per_pop,
        num_parents_mating=args.num_parents_mating,
    )

    print("\n=== TorchGA MixHop 結果 ===")
    print(f"Dataset: {args.dataset}")
    print(f"Best fitness (val macro F1): {result['best_val_macro_f1']:.6f}")
    print(f"Train macro F1: {result['train_macro_f1']:.6f}")
    print(f"Val macro F1:   {result['val_macro_f1']:.6f}")
    print(f"Test macro F1:  {result['test_macro_f1']:.6f}")
    print(f"Config: {result['config']}")


if __name__ == "__main__":
    main()
