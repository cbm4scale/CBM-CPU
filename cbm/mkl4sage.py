from torch import sparse_coo_tensor, sparse_csr_tensor, ones, float32, int32, stack, tensor
from cbm import cbm_mkl_cpp as mkl
from cbm.mkl4mm import mkl4mm


class mkl4sage(mkl4mm):

    def __init__(self, edge_index):
        self.num_nodes = edge_index.max().item() + 1
        # A
        mkl_csr = sparse_coo_tensor(edge_index, ones(len(edge_index[0]), dtype=float32), size=(self.num_nodes, self.num_nodes)).to_sparse_csr()
        d = [1.0 / (mkl_csr.crow_indices()[row_idx + 1] - mkl_csr.crow_indices()[row_idx]) for row_idx in range(0, len(mkl_csr.crow_indices()) - 1)]
        new_values = mkl_csr.values()    
        for row_idx in range(0, len(mkl_csr.crow_indices()) - 1):
            row_ptr_s = mkl_csr.crow_indices()[row_idx].item()
            row_ptr_e = mkl_csr.crow_indices()[row_idx + 1].item()
            new_values[row_ptr_s:row_ptr_e] *= d[row_idx]
        self.a = sparse_coo_tensor(edge_index, new_values, size=(self.num_nodes, self.num_nodes)).to_sparse_csr()
        # A.T
        edge_index_t = stack([edge_index[1], edge_index[0]])
        mkl_csc = sparse_coo_tensor(edge_index_t, ones(len(edge_index_t[0]), dtype=float32), size=(self.num_nodes, self.num_nodes)).to_sparse_csr()
        column_indices_t = mkl_csc.col_indices()
        d = tensor(d, dtype=float32)
        new_values_t = mkl_csc.values() * d[column_indices_t]
        self.a_t = sparse_csr_tensor(mkl_csc.crow_indices(), mkl_csc.col_indices(), new_values_t, (self.num_nodes, self.num_nodes)).to(float32)
    
    def matmul(self, x, y):
        row_ptr_s = self.a.crow_indices()[:-1].to(int32)
        row_ptr_e = self.a.crow_indices()[1:].to(int32)
        col_ptr = self.a.col_indices().to(int32)
        val_ptr = self.a.values().to(float32)
        mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)
    
    def matmul_t(self, x, y):
        row_ptr_s = self.a_t.crow_indices()[:-1].to(int32)
        row_ptr_e = self.a_t.crow_indices()[1:].to(int32)
        col_ptr = self.a_t.col_indices().to(int32)
        val_ptr = self.a_t.values().to(float32)
        mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)