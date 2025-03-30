from cbm.mkl4gin import mkl4gin
from torch import empty
from torch.nn import Module, Parameter


class MKLGINInference(Module):
    def __init__(self, in_features, a, eps, train_eps, nn):
        super(MKLGINInference, self).__init__()
        assert isinstance(a, mkl4gin), "the adjacency matrix should be an instance of mkl4gin"
        self.a = a
        self.y = empty((a.num_nodes, in_features))
        self.dloss_dx = empty((a.num_nodes, in_features))
        self.nn = nn
        self.train_eps = train_eps
        self.initial_eps = eps
        if self.train_eps:
            self.eps = Parameter(empty(1))
        else:
            self.register_buffer('eps', empty(1))
        self.eps.data.fill_(self.initial_eps)
        

    def forward(self, x, edge_index):
        # A @ X + (1 + eps) * X
        self.a.matmul_add(x, self.y, self.eps)
        # NeuralNetwork(AX+(1+eps)X)
        out = self.nn(self.y)
        return out
