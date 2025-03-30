import torch
import time
from cbm.cbm4mm import cbm4mm
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4sage(cbm4mm):

    def __init__(self, edge_index, edge_values, alpha=0):
        # get number of rows in input dataset 
        num_rows = max(edge_index[0].max(), edge_index[1].max()) + 1

        # represent input dataset in CBM
        cbm_data = cbm_cpp.init(edge_index[0],  # row indices
                                edge_index[1],  # column indices
                                edge_values,    # value of nnz's
                                num_rows,       # number of rows
                                alpha)          # prunning param

        # unpack resulting data
        delta_edge_index = torch.stack([cbm_data[0], cbm_data[1]])
        delta_values = cbm_data[2]
        self.mca_branches = cbm_data[3]
        self.mca_row_idx = cbm_data[4]
        self.mca_col_idx = cbm_data[5] 

        # convert matrix of deltas to COO tensor (torch.float32)
        coo_tensor = torch.sparse_coo_tensor(delta_edge_index, 
                                             delta_values, 
                                             (num_rows, num_rows))
        
        # convert matrix of deltas to CSR tensor (torch.float32)
        self.deltas = coo_tensor.to(torch.float32).coalesce().to_sparse_csr()
        
        num_rows = self.deltas.size()[0]
        d = torch.zeros(num_rows,1)
        x = torch.ones(num_rows,1)

        # resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # computes D^{⁻1} and flattens 
        self.D = (1.0 / d).view(-1)

        # scale columns of matrix of deltas
        new_values = self.deltas.values()

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

        #compress A^t in CBM

        # represent input dataset in CBM
        cbm_data_t = cbm_cpp.init(edge_index[1],  # row indices
                                  edge_index[0],  # column indices
                                  edge_values,    # value of nnz's
                                  num_rows,       # number of rows
                                  alpha)          # prunning param

        # unpack resulting data
        delta_edge_index_t = torch.stack([cbm_data_t[0], cbm_data_t[1]])
        delta_values_t = cbm_data_t[2]
        self.mca_branches_t = cbm_data_t[3]
        self.mca_row_idx_t = cbm_data_t[4]
        self.mca_col_idx_t = cbm_data_t[5] 

        # convert matrix of deltas to COO tensor (torch.float32)
        coo_tensor_t = torch.sparse_coo_tensor(delta_edge_index_t, 
                                               delta_values_t, 
                                               (num_rows, num_rows))
        
        # convert matrix of deltas to CSR tensor (torch.float32)
        self.deltas_t = coo_tensor.to(torch.float32).coalesce().to_sparse_csr()
        
        # get csr column indices of matrix of deltas
        column_indices_t = self.deltas_t.col_indices()

        # print("cbm")
        # print(type(self.D))
        # print(column_indices_t)
        
        # scale columns of matrix of deltas
        new_values_t = self.deltas_t.values() * self.D[column_indices_t]

        self.deltas_t = torch.sparse_csr_tensor(self.deltas_t.crow_indices(),
                                                self.deltas_t.col_indices(),
                                                new_values_t, 
                                                (num_rows, num_rows)).to(torch.float32)
        
        self.num_nodes = num_rows

    def matmul(self, x, y):
        """
        Matrix multiplication with CBM format:

        Computes the product between the matrix of delta (self.deltas) and a 
        dense real-valued matrix. The result of this product is stored in 
        another dense real-valued matrix y. Matrix y is subsequently updated
        according to the compression tree (self.mca_row_ptr /  self.mca_col_idx) 
        that was obtained during the construction of the CBM format, and the 
        scaling factors (self.D) required for normalization. 
        
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
            self.mca_row_idx.to(torch.int32), 
            self.mca_col_idx.to(torch.int32),
            self.D.to(torch.float32), 
            y)

    def matmul_t(self, x, y):
        """
        Transposed matrix multiplication with CBM format:

        Computes the product between the matrix of tranposed deltas (self.deltas_t) and a 
        dense real-valued matrix. The result of this product is stored in 
        another dense real-valued matrix y. Matrix y is subsequently updated
        according to the compression tree (self.mca_row_ptr /  self.mca_col_idx) 
        that was obtained during the construction of the CBM format, and the 
        scaling factors (self.D) required for normalization. 
        
        Notes: -This method wraps C++ code and resorts to Intel MKL sparse BLAS.
               -Use OpenMP environment variables to control parallelism
            
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        """

        cbm_cpp.s_spmm_update_csr_int32(
            self.deltas_t.crow_indices()[:-1].to(torch.int32),
            self.deltas_t.crow_indices()[1: ].to(torch.int32),
            self.deltas_t.col_indices().to(torch.int32),
            self.deltas_t.values().to(torch.float32),
            x,
            self.mca_branches_t.to(torch.int32), 
            self.mca_row_idx_t.to(torch.int32), 
            self.mca_col_idx_t.to(torch.int32),
            y)

    def update(self, y):
        """
        Helper / Debugging / Benchmarking method:

        Computes the update stage of CBM format, according to the compression 
        tree (self.mca_row_ptr / self.mca_col_idx) that was obtained during the 
        construction of the format.

        Note: -Use OpenMP environment variables to control parallelism.

        Args:
            y (pytorch.Tensor): matrix to be updated.     
        """
        
        cbm_cpp.s_fused_update_csr_int32(
            self.mca_branches.to(torch.int32), 
            self.mca_row_idx.to(torch.int32), 
            self.mca_col_idx.to(torch.int32),
            self.D, y)