from torch import int32, float32, ones, zeros, sparse_coo_tensor, sparse_csr_tensor
from cbm.mkl4mm import mkl4mm


class mkl4dad(mkl4mm):

    def __init__(self, edge_index):
        # get number of nodes
        self.num_nodes = edge_index.max().item() + 1
        row, col = edge_index[0], edge_index[1]
        out = zeros((self.num_nodes))
        one = ones((col.size(0)))
        deg = out.scatter_add_(0, col, one)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self.a = sparse_coo_tensor(edge_index, 
                                   edge_weight, 
                                   size=(self.num_nodes, self.num_nodes)).to_sparse_csr()
        
