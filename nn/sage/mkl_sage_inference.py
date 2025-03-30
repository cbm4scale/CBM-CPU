from cbm.mkl4sage import mkl4sage
from torch import empty
from torch.nn import Module, Linear


class MKLSAGEInference(Module):
    def __init__(self, in_features, out_features, bias, a):
        super(MKLSAGEInference, self).__init__()
        assert isinstance(a, mkl4sage), "the adjacency matrix should be an instance of mkl4sage"
        self.a = a
        self.y = empty((a.num_nodes, out_features))
        self.dloss_dx = empty((a.num_nodes, out_features))  # here x is in fact XW_l therefore dLoss_dX -> dLoss_dXW_l -> shape: (num_nodes, out_features)
        self.lin_l = Linear(in_features, out_features, bias)
        self.lin_r = Linear(in_features, out_features, False)

    def forward(self, x, edge_index):
        # X @ W_l
        x_l = self.lin_l(x)
        # D^(-1)A @ XW_l
        self.a.matmul(x_l, self.y)
        # X @ W_r
        out_r = self.lin_r(x)
        # DAXW_l + XW_r
        out = self.y + out_r
        return out
