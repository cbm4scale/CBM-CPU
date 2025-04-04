import argparse
from time import time
from torch import inference_mode, ones, zeros, randint, arange, empty, tensor
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=[
        "cbm-ax", "cbm-adx", "cbm-dadx", "cbm-mm-add",
        "mkl-ax", "mkl-adx", "mkl-dadx", "mkl-mm-add"
    ], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, help="Number of columns to use in X. If not set, the original number of columns will be used.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of matrix multiplications.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations.")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    if args.columns:
        dataset.x = ones((dataset.num_nodes, args.columns))
        dataset.num_classes = 2
        dataset.y = zeros((dataset.num_nodes, 2))
        dataset.y[arange(dataset.num_nodes), randint(0, 2, (dataset.num_nodes,))] = 1
        print_dataset_info(f"[FAKE] {args.dataset}", dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # adjacency matrix
    a, a_t = set_adjacency_matrix(args.operation, dataset.edge_index, alpha)
    del dataset.edge_index

    performance = []

    with inference_mode():
        x = dataset.x
        y = empty((dataset.num_nodes, dataset.num_features))
        for iterations in range(1, args.warmup + args.iterations + 1):
            time_start = time()

            # matrix multiplication
            a.matmul(x, y)

            time_end = time()
            performance.append(time_end - time_start)
    
    performance = tensor(performance[args.warmup:])
    print(f"[{args.operation}] [{'FAKE' if args.columns else 'REAL'} {args.dataset}] [{alpha}] [{args.columns if args.columns else dataset.num_features}]   Mean: {performance.mean():.6f} s   |   Std: {performance.std():.6f} s   |   Min: {performance.min():.6f} s   |   Max: {performance.max():.6f} s")