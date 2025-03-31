import torch
import time
from cbm.cbm4mm import cbm4mm
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4gin(cbm4mm):

    def __init__(self, edge_index, edge_values, alpha=0):
        super().__init__(edge_index, edge_values, alpha)

    def matmul_add(self, x, y, epsilon):
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

        cbm_cpp.s_spmm_update_add_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32),
            self.deltas.crow_indices()[1: ].to(torch.int32),
            self.deltas.col_indices().to(torch.int32),
            self.deltas.values().to(torch.float32),
            x, epsilon,
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32),
            y)
    
    def matmul(self, x, y):
        self.matmul_add(x, y, 1)


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
        
        cbm_cpp.s_update_csr_int32(
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32),
            y)