from cbm.mkl4gcn import mkl4gcn
from torch import empty
from torch.nn import Module, Linear
from torch.autograd import Function


class MKLGCNAutograd(Function):
    @staticmethod
    def forward(ctx, x, a, y, dloss_dx):
        ctx.save_for_backward(dloss_dx)
        ctx.a = a
        a.matmul(x, y)
        return y

    @staticmethod
    def backward(ctx, dloss_dy):
        dloss_dx = ctx.saved_tensors[0]
        a = ctx.a
        # dLoss_dXW = dY_dXW @ dLoss_dY = DAD @ dLoss_dY
        a.matmul(dloss_dy, dloss_dx)    # here x is in fact XW because X @ W is handled by the default torch.autograd (not by our custom injection)
        return dloss_dx, None, None, None


class MKLGCN(Module):
    def __init__(self, in_features, out_features, bias, a):
        super(MKLGCN, self).__init__()
        assert isinstance(a, mkl4gcn), "the adjacency matrix should be an instance of mkl4gcn"
        self.a = a
        self.y = empty((a.num_nodes, out_features))
        self.dloss_dx = empty((a.num_nodes, out_features))  # here x is in fact XW therefore dLoss_dX -> dLoss_dXW -> shape: (num_nodes, out_features)
        self.lin = Linear(in_features, out_features, False)

    def forward(self, x, edge_index):
        # X @ W
        x = self.lin(x)
        # DAD @ XW
        out = MKLGCNAutograd.apply(x, self.a, self.y, self.dloss_dx)
        return out
