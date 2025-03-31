import argparse
from torch import ones, zeros, randint, arange, testing
from torch.nn import Module, ModuleList
from utilities import set_layer, load_dataset, print_dataset_info, set_adjacency_matrix, criterions, optimizers


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
    parser.add_argument("--nn", choices=["gcn", "gin", "sage"], required=True)
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
    a1, a_t1 = set_adjacency_matrix(f"cbm-{args.nn}", dataset.edge_index, alpha)
    a2, a_t2 = set_adjacency_matrix(f"mkl-{args.nn}", dataset.edge_index, alpha)
    a3, a_t3 = set_adjacency_matrix(f"pyg-{args.nn}", dataset.edge_index, alpha)
    # model
    model1 = NodePrediction(f"cbm-{args.nn}", dataset.num_features, args.hidden_features, dataset.num_classes, args.num_hidden_layers, args.bias, a1, a_t1)
    model2 = NodePrediction(f"mkl-{args.nn}", dataset.num_features, args.hidden_features, dataset.num_classes, args.num_hidden_layers, args.bias, a2, a_t2)
    model3 = NodePrediction(f"pyg-{args.nn}", dataset.num_features, args.hidden_features, dataset.num_classes, args.num_hidden_layers, args.bias, a3, a_t3)

    # set state dict for comparison
    model1.load_state_dict(model3.state_dict())
    model2.load_state_dict(model3.state_dict())

    models = [model1, model2, model3]
    losses = [[],[],[]]

    for i in range(3):
        criterion = criterions[args.criterion]()
        optimizer = optimizers[args.optimizer](models[i].parameters(), lr=args.lr)

        models[i].train()
        for epoch in range(1, args.epochs + 1):
            # forward pass
            y = models[i](x=dataset.x, edge_index=dataset.edge_index)
            # backward pass
            loss = criterion(y, dataset.y)
            optimizer.zero_grad()   # clear previous gradients
            loss.backward()         # compute new gradients
            optimizer.step()        # update learnable parameters

            losses[i].append(loss.item())

    print(f"loss-pyg: {losses[2]}")
    print(f"loss-cbm: {losses[0]}")
    print(f"loss-mkl: {losses[1]}")
    
    testing.assert_close(losses[2], losses[0], atol=1e-4, rtol=1e-4)
    testing.assert_close(losses[2], losses[1], atol=1e-4, rtol=1e-4)
