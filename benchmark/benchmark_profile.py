import argparse
from os import makedirs
from os.path import exists
from torch import inference_mode, ones, zeros, randint, arange
from torch.nn import Module, ModuleList
from utilities import set_layer, load_dataset, print_dataset_info, set_adjacency_matrix, criterions, optimizers
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function


class NodePrediction(Module):
    def __init__(self, layer, in_features, hidden_features, out_features, num_hidden_layers, bias, a, a_t=None):
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
    parser.add_argument("--nn", choices=[
        "pyg-gcn", "pyg-gin", "pyg-sage", 
        "cbm-gcn", "cbm-gin", "cbm-sage", 
        "mkl-gcn", "mkl-gin", "mkl-sage"
    ], required=True)
    parser.add_argument("--train", action="store_true", help="Train the model (inference is the default).")
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--hidden_features", type=int, default=10, help="Number of hidden features to use in the model. If --fake is set, this will also be the number of input features.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers to use in the model.")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to use in training/inference.")
    parser.add_argument("--criterion", type=str, default="mse", help="Criterion to use in training (loss function).")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use in training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--fake", action="store_true", help="Fake X and Y.")
    args = parser.parse_args()

    profiler_dir = "./logs/profiler/"
    if not exists(profiler_dir):
        makedirs(profiler_dir)

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    if args.fake:
        dataset.x = ones((dataset.num_nodes, args.hidden_features))
        dataset.num_classes = 2
        dataset.y = zeros((dataset.num_nodes, 2))
        dataset.y[arange(dataset.num_nodes), randint(0, 2, (dataset.num_nodes,))] = 1
        print_dataset_info(f"[FAKE] {args.dataset}", dataset)

    if args.alpha is not None:
        alpha = args.alpha

    # adjacency matrix
    a, a_t = set_adjacency_matrix(args.nn, dataset.edge_index, alpha)
    # model
    model = NodePrediction(args.nn, dataset.num_features, args.hidden_features, dataset.num_classes, args.num_hidden_layers, args.bias, a, a_t)

    if args.train:
        criterion = criterions[args.criterion]()
        optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr)

        model.train()
        with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler(profiler_dir), record_shapes=True, profile_memory=True, with_stack=True) as prof:
            for epoch in range(1, args.epochs + 1):
                # forward pass
                with record_function("Forward Pass"):
                    y = model(x=dataset.x, edge_index=dataset.edge_index)
                # backward pass
                with record_function("Loss Computation"):
                    loss = criterion(y, dataset.y)
                print(f'[{epoch}/{args.epochs}] Loss: {loss.item():.6f}')
                with record_function("Backward Pass"):
                    optimizer.zero_grad()   # clear previous gradients
                    loss.backward()         # compute new gradients
                    optimizer.step()        # update learnable parameters
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))
    else:
        model.eval()
        with inference_mode():
            for epoch in range(1, args.epochs + 1):
                # forward pass
                y = model(x=dataset.x, edge_index=dataset.edge_index)
