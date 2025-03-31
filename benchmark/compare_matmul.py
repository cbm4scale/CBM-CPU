import argparse
from torch import inference_mode, empty, rand, testing
from utilities import load_dataset, set_adjacency_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", choices=["ax", "adx", "dadx", "mm-add"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, help="Number of columns to use in X. If not set, the original number of columns will be used.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of matrix multiplications.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # adjacency matrix
    cbm_a, _ = set_adjacency_matrix(f"cbm-{args.nn}", dataset.edge_index, alpha)
    mkl_a, _ = set_adjacency_matrix(f"mkl-{args.nn}", dataset.edge_index, alpha)
    del dataset.edge_index

    print("------------------------------------------------------------")
    with inference_mode():
        for iteration in range(1, args.iterations + 1):
            x = rand((dataset.num_nodes, args.columns)) if args.columns else dataset.x
            cbm_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features))    # this doesn't need to be done here but if we want to vary the number of columns, we need to create a new empty tensor
            mkl_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features))    # this doesn't need to be done here but if we want to vary the number of columns, we need to create a new empty tensor

            # matrix multiplication
            cbm_a.matmul(x, cbm_y)
            mkl_a.matmul(x, mkl_y)

            # compare
            try:
                testing.assert_close(cbm_y, mkl_y, atol=args.atol, rtol=args.rtol)
                print(f"[{iteration}/{args.iterations}] PASSED")
            except AssertionError as e:
                print(f"[{iteration}/{args.iterations}] FAILED: {e}")
            print("------------------------------------------------------------")