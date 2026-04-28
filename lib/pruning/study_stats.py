import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

def main():

    parser = ArgumentParser()
    parser.add_argument("--stats_dir", type=Path, required=True)
    parser.add_argument("--num_heads", type=int, default=16, required=True)
    parser.add_argument("--num_blocks", type=int, default=24, required=True)

    parser.add_argument("--prune_heads", type=int, required=True)
    parser.add_argument("--prune_blocks", type=int, required=True)

    args = parser.parse_args()

    mean_tensors = {}
    std_tensors = {}

    # Load dino stats
    stats_dir = Path(args.stats_dir)

    max_std = 0
    for layer in range(args.num_blocks):
        mean_file = stats_dir / f"dino_{layer}_mean.pt"
        if mean_file.exists():
            mean = torch.load(mean_file)
            mean_tensors[layer] = mean

        std_file = stats_dir / f"dino_{layer}_std.pt"
        if std_file.exists():
            std = torch.load(std_file)
            std_tensors[layer] = std

            head_stds = [round(std_tensors[layer][head].sum().item(), 2) for head in range(args.num_heads)]
            max_std = max(max_std, max(head_stds))  # Update max

    std_all = {}
    for layer in range(args.num_blocks):
        mean = mean_tensors.get(layer, None)
        std = std_tensors.get(layer, None)

        if mean is None or std is None:
            continue

        head_stds = [round(std[head].sum().item(), 2) for head in range(args.num_heads)]
        std_all[layer] = head_stds

    # Compute means for each entry
    means = {key: np.mean(values) for key, values in std_all.items()}

    # Extract keys and values for plotting
    if args.prune_blocks is not None:
        keys = list(means.keys())
        values = list(means.values())

        prune_blocks = np.argsort(values)[:args.prune_blocks]
        print(prune_blocks)

    # Flatten the array and get the indices of the k smallest values
    if args.prune_heads is not None:
        ph = np.stack(list(std_all.values()))
        flat_indices = np.argpartition(ph.ravel(), args.prune_heads)[:args.prune_heads]  # Get top k smallest indices (unsorted)
        sorted_indices = flat_indices[np.argsort(ph.ravel()[flat_indices])]  # Sort them
        indices_2d = np.column_stack(np.unravel_index(sorted_indices, ph.shape))

        id_scattered = defaultdict(list)
        for b, h in indices_2d:
            id_scattered[b.item()].append(h.item())

        print(dict(id_scattered))


if __name__ == "__main__":
    main()