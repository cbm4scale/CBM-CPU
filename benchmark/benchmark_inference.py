import argparse
from time import time
from torch import inference_mode, ones, tensor
from torch.nn import Module, ModuleList
from utilities import set_layer, load_dataset, print_dataset_info, set_adjacency_matrix


class NodePrediction(Module):
    def __init__(self, layer: str, in_features: int, hidden_features: int, out_features: int, num_hidden_layers: int, bias: bool, a, a_t=None):
        super(NodePrediction, self).__init__()

        # input layer
        self.layers = [set_layer(layer, in_features, hidden_features, bias, a, a_t)]
        # hidden layers
        self.layers += [set_layer(layer, hidden_features, hidden_features, bias, a, a_t) for _ in range(num_hidden_layers)]
        # output layer
        self.layers += [set_layer(layer, hidden_features, out_features, bias, a, a_t)]

        self.layers = ModuleList(self.layers)   # required for training so that the optimizer can find the parameters
    
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = x.relu()
        out = self.layers[-1](x, edge_index)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", choices=[   # pyg gnns removed because they inherintly use autograd therefore it would not be a fair comparison
        "cbm-gcn-inference", "cbm-gin-inference", "cbm-sage-inference", 
        "mkl-gcn-inference", "mkl-gin-inference", "mkl-sage-inference"
    ], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh", choices=["ca-HepPh", "ca-AstroPh", "ogbn-proteins-ignore", "ogbn-proteins-raw", "PubMed", "Cora", "coPapersCiteseer", "coPapersDBLP", "COLLAB"], help="Dataset to use for inference.")
    parser.add_argument("--hidden_features", type=int, default=10, help="Number of hidden features to use in the model. If --fake is set, this will also be the number of input features.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers to use in the model.")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to use in inference.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--fake", action="store_true", help="Fake X and Y.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs.")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    if args.fake:   # if fake, set X and Y to ones and use hidden_features as both the number of input features and the number of classes
        dataset.x = ones((dataset.num_nodes, args.hidden_features))
        dataset.num_classes = args.hidden_features
        dataset.y = ones((dataset.num_nodes, args.hidden_features))
        print_dataset_info(f"[FAKE] {args.dataset}", dataset)

    if args.alpha is not None:
        alpha = args.alpha

    # adjacency matrix
    a, a_t = set_adjacency_matrix(args.nn, dataset.edge_index, alpha)
    del dataset.edge_index
    # model
    model = NodePrediction(args.nn, dataset.num_features, args.hidden_features, dataset.num_classes, args.num_hidden_layers, args.bias, a, a_t)

    performance = []

    model.eval()
    with inference_mode():
        for epoch in range(1, args.warmup + args.epochs + 1):
            time_start = time()

            # forward pass
            y = model(x=dataset.x, edge_index=None)

            time_end = time()
            performance.append(time_end - time_start)
    
    # remove warmup
    performance = tensor(performance[args.warmup:])
    print(f"[{args.nn}] [{'FAKE' if args.fake else 'REAL'} {args.dataset}] [{alpha}] [{args.num_hidden_layers}] [{args.hidden_features}]   Mean: {performance.mean():.6f} s   |   Std: {performance.std():.6f} s   |   Min: {performance.min():.6f} s   |   Max: {performance.max():.6f} s")