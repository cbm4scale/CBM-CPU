from torch import sparse_coo_tensor, ones, int32, float32
from cbm import cbm_mkl_cpp as mkl
from cbm.mkl4mm import mkl4mm


class mkl4gin(mkl4mm):

    def __init__(self, edge_index):
        self.num_nodes = edge_index.max().item() + 1
        self.a = sparse_coo_tensor(edge_index.to(int32), ones(edge_index.size(1)), (self.num_nodes, self.num_nodes)).to_sparse_csr()
    
    def matmul_add(self, x, y, epsilon):
        row_ptr_s = self.a.crow_indices()[:-1].to(int32)
        row_ptr_e = self.a.crow_indices()[1:].to(int32)
        col_ptr = self.a.col_indices().to(int32)
        val_ptr = self.a.values().to(float32)
        mkl.s_spmm_add_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, epsilon, y)
    
    def matmul(self, x, y):
        self.matmul_add(x, y, 1)