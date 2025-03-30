from cbm.cbm4gin import cbm4gin
from torch import empty, empty_like
from torch.nn import Module, Parameter
from torch.autograd import Function


class CBMGINAutograd(Function):
    @staticmethod
    def forward(ctx, x, a, eps, train_eps, y, dloss_dx):
        ctx.save_for_backward(x, dloss_dx)
        ctx.a = a
        ctx.eps = eps
        ctx.train_eps = train_eps
        a.matmul_add(x, y, eps)
        return y

    @staticmethod
    def backward(ctx, dloss_dy):
        x = ctx.saved_tensors[0]
        dloss_dx = ctx.saved_tensors[1]
        a = ctx.a
        eps = ctx.eps
        train_eps = ctx.train_eps
        # dLoss_dX = dY_dX @ dLoss_dY = kron(I,(A+(1 + eps)I)) @ dLoss_dY = (A+(1 + eps)) @ dLoss_dY = A @ dLoss_dY + (1 + eps) * dLoss_dY
        a.matmul_add(dloss_dy, dloss_dx, eps)
        # dLoss_deps = dY_deps @ dLoss_dY = X * dLoss_dY (where * is element-wise multiplication, also known as Hadamard product)
        dloss_deps = (dloss_dy * x) if train_eps else None
        return dloss_dx, None, dloss_deps, None, None, None


class CBMGIN(Module):
    def __init__(self, in_features, a, eps, train_eps, nn):
        super(CBMGIN, self).__init__()
        assert isinstance(a, cbm4gin), "the adjacency matrix should be an instance of cbm4gin"
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
        x = CBMGINAutograd.apply(x, self.a, self.eps, self.train_eps, self.y, self.dloss_dx)
        # NeuralNetwork(AX+(1+eps)X)
        out = self.nn(x)
        return out
