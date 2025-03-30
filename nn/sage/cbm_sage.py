from cbm.cbm4sage import cbm4sage
from torch import empty
from torch.nn import Module, Linear
from torch.autograd import Function


class CBMSAGEAutograd(Function):
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
        # dLoss_dXW_l = dY_dXW_l @ dLoss_dY = (DA).T @ dLoss_dY
        a.matmul_t(dloss_dy, dloss_dx)    # here x is in fact XW_l because X @ W_l is handled by the default torch.autograd (not by our custom injection)
        return dloss_dx, None, None, None


class CBMSAGE(Module):
    def __init__(self, in_features, out_features, bias, a):
        super(CBMSAGE, self).__init__()
        assert isinstance(a, cbm4sage), "the adjacency matrix should be an instance of cbm4sage"
        self.a = a
        self.y = empty((a.num_nodes, out_features))
        self.dloss_dx = empty((a.num_nodes, out_features))  # here x is in fact XW_l therefore dLoss_dX -> dLoss_dXW_l -> shape: (num_nodes, out_features)
        self.lin_l = Linear(in_features, out_features, bias)
        self.lin_r = Linear(in_features, out_features, False)

    def forward(self, x, edge_index):
        # X @ W_l
        x_l = self.lin_l(x)
        # D^(-1)A @ XW_l
        out_l = CBMSAGEAutograd.apply(x_l, self.a, self.y, self.dloss_dx)
        # X @ W_r
        out_r = self.lin_r(x)
        # DAXW_l + XW_r
        out = out_l + out_r
        return out
