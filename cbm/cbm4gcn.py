import torch
import time
from cbm.cbm4mm import cbm4mm
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4gcn(cbm4mm):

    def __init__(self, edge_index, edge_values, alpha=0):
        super().__init__(edge_index, edge_values, alpha)

        # for GCNConv Â = D^{⁻1/2} A D^{⁻1/2}, when D is the degree matrix of A 
        num_rows = self.deltas.size()[0]
        d = torch.zeros(num_rows,1)
        x = torch.ones(num_rows,1)

        # resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # compute D^{⁻1/2} and flattens 
        self.D = (d ** (-1/2)).view(-1)

        # get csr column indices of matrix of deltas
        column_indices = self.deltas.col_indices()
        
        # scale columns of matrix of deltas
        new_values = self.deltas.values() * self.D[column_indices]

        # find nodes that are "not" in the mca
        missing_nodes = set(range(num_rows)) - set(self.deltas.col_indices())

        # scale missing mca nodes (rows)
        for row_idx in missing_nodes:
            row_ptr_s = self.deltas.crow_indices()[row_idx].item()
            row_ptr_e = self.deltas.crow_indices()[row_idx + 1].item()
            new_values[row_ptr_s:row_ptr_e] *= self.D[row_idx]


        self.deltas = torch.sparse_csr_tensor(self.deltas.crow_indices(),
                                              self.deltas.col_indices(),
                                              new_values, 
                                              (num_rows, num_rows)).to(torch.float32)


    def matmul(self, x, y):
        """
        Matrix multiplication with CBM format:

        Computes the product between the matrix of delta (self.deltas) and a 
        dense real-valued matrix. The result of this product is stored in 
        another dense real-valued matrix y. Matrix y is subsequently updated
        according to the compression tree (self.mca_src_idx /  self.mca_dst_idx) 
        that was obtained during the construction of the CBM format, and the 
        scaling factors (self.D) required due to normalization. 
        
        Notes: -This method wraps C++ code and resorts to Intel MKL sparse BLAS.
               -Use OpenMP environment variables to control parallelism
            
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        """

        cbm_cpp.s_spmm_fused_update_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32),
            self.deltas.crow_indices()[1: ].to(torch.int32),
            self.deltas.col_indices().to(torch.int32),
            self.deltas.values().to(torch.float32),
            x,
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32),
            self.D.to(torch.float32), 
            y)


    def update(self, y):
        """
        Helper / Debugging / Benchmarking method:

        Computes the update stage of CBM format, according to the compression 
        tree (self.mca_src_idx / self.mca_dst_idx) that was obtained during the 
        construction of the format.

        Note: -Use OpenMP environment variables to control parallelism.

        Args:
            y (pytorch.Tensor): matrix to be updated.     
        """
        
        cbm_cpp.s_fused_update_csr_int32(
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32),
            self.D, y)