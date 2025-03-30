#include <omp.h>
#include <torch/extension.h>

#include "mkl.h"
#include "mkl_spblas.h"

#define CHECK_DTYPE(x, dtype) \
    TORCH_CHECK(x.scalar_type() == dtype,  \
    "\"" #x "\" is not a tensor of type \"" #dtype "\"")

static inline void check_alloc(void* ptr) {
    if (ptr == NULL) {
            std::cerr << "Memory allocation failed." << std::endl;
            exit(-1);
            // Handle allocation failure
        } else {
            // Allocation was successful, use mkl_array as needed
            std::cout << "Memory allocation successful." << std::endl;
    }
}

static inline void check_status(sparse_status_t s) {
    if (s == SPARSE_STATUS_SUCCESS) {
        return;
    }
    else if (s == SPARSE_STATUS_NOT_INITIALIZED) {
        printf("MKL: Not Initialized\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_ALLOC_FAILED) {
        printf("MKL: Not Alloc'ed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INVALID_VALUE) {
        printf("MKL: Invalid Value\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_EXECUTION_FAILED) {
        printf("MKL: Execution Failed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INTERNAL_ERROR) {
        printf("MKL: Internal Error\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_NOT_SUPPORTED) {
        printf("MKL: Not Supported\n");
        exit(-1);
    }
}

 
static inline void omp_s_spmm_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                         const at::Tensor& lhs_row_ptr_e, 
                                         const at::Tensor& lhs_col_idx, 
                                         const at::Tensor& lhs_values,
                                         const at::Tensor& rhs,
                                         at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          dst.size(0), 
                                          rhs.size(1),
                                          lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                          lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                          lhs_col_idx.data_ptr<MKL_INT>(),
                                          lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);
}

// AX + (1+epsilon)IX
static inline void omp_s_spmm_add_v1_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                const at::Tensor& lhs_row_ptr_e, 
                                                const at::Tensor& lhs_col_idx, 
                                                const at::Tensor& lhs_values,
                                                const at::Tensor& rhs,
                                                const float epsilon,
                                                at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         dst.size(0), 
                                         rhs.size(1),
                                         lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                         lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                         lhs_col_idx.data_ptr<MKL_INT>(),
                                         lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);

    // tread dst and rhs as vectors to compute AX + (1+epsilon)IX
    cblas_saxpy(dst.size(0) * dst.size(1), 1 + epsilon, rhs_data, 1, dst_data, 1);
}


// (A + (1+epsilon)I)X (assumes A contains diagonal entries or self-loops!)
static inline void omp_s_spmm_add_v2_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                const at::Tensor& lhs_row_ptr_e, 
                                                const at::Tensor& lhs_col_idx,
                                                const at::Tensor& lhs_dia_idx, 
                                                const at::Tensor& lhs_values,
                                                const at::Tensor& rhs,
                                                const float epsilon,
                                                at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_dia_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);
    
    MKL_INT n_rows = dst.size(0);
    MKL_INT *lhs_col_idx_data = lhs_col_idx.data_ptr<MKL_INT>();
    MKL_INT *lhs_dia_idx_data = lhs_dia_idx.data_ptr<MKL_INT>();

    #pragma omp parallel for
    for (MKL_INT d = 0; d < n_rows; d++) {
        lhs_col_idx_data[lhs_dia_idx_data[d]] = 1 + epsilon;
    }

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         dst.size(0), 
                                         rhs.size(1),
                                         lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                         lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                         lhs_col_idx_data,
                                         lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);
}

// add self-loops and get diagonal entries indices within lhs_col_idx
// TODO: should we do it here or on python's side? Ask Tomás



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("s_spmm_csr_int32", &omp_s_spmm_csr_int32_);
    m.def("s_spmm_add_v1_csr_int32", &omp_s_spmm_add_v1_csr_int32_);
    m.def("s_spmm_add_v2_csr_int32", &omp_s_spmm_add_v2_csr_int32_);
}
