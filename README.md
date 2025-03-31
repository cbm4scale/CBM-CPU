# CBM4Scale

CBM4Scale is a high-performance implementation of Graph Neural Networks (GNNs) using the Compressed Binary Matrix (CBM) format, optimized for Intel CPUs. This project provides efficient implementations of popular GNN architectures including Graph Convolutional Networks (GCN), GraphSAGE, and Graph Isomorphism Networks (GIN) using both parallel and sequential processing.

Key features:
- Custom CBM format implementation optimized for sparse matrix operations
- Seamless integration with PyTorch's autograd framework
- Native-like usage similar to PyTorch Geometric (PyG) neural network modules
- Optimized CSR-format implementation using Intel MKL for benchmarking
- Comprehensive benchmarking suite comparing against PyG's native implementations
- Support for both parallel and sequential processing

The project provides researchers and practitioners with flexibility in their implementation choices, offering optimized sparse matrix operations through our custom CBM format, Intel MKL-optimized CSR operations, and native PyG implementations. This enables comprehensive performance comparisons and optimization across different graph neural network architectures while maintaining full compatibility with existing PyTorch workflows.

## Setup

1. **Install Intel oneAPI Base Toolkit**  
   Download and install the Intel oneAPI Base Toolkit following the instructions provided [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/overview.html).

2. **Create a Conda Environment**  
   Set up a new Conda environment and install the necessary dependencies:
   ```bash
   conda create -n cbm python=3.11
   conda activate cbm
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip uninstall numpy
   pip install numpy==1.24.3
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
   pip install ogb
   conda install cmake ninja wget prettytable scipy
    ```
4. **Clone and Install the Repository**  
   Clone the repository and set up the project:
   ```bash
   git clone https://github.com/cbm4scale/CBM4Scale.git --recursive
   cd CBM4Scale/
   git submodule init
   git submodule update
   python conda_setup.py  # If Intel oneAPI is not installed in the default directory, use: --setvars_path PATH_TO_ONEAPI/setvars.sh
   export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
   export PYTHONPATH=./:$PYTHONPATH
   ```

## Usage

### `./scripts/alpha_searcher.sh`
This script calculates the execution time of the matrix multiplication method defined in `cbm/cbm4mm.py` via `benchmark/benchmark_matmul.py` for each combination of alpha values specified in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array. 

Upon completion, the script generates a results file named `results/alpha_searcher_results.txt`, which records the matrix multiplication runtime, in seconds, for each combination of alpha values and datasets using the CBM format. Additionally, the resulting file includes the execution time of the matrix multiplication method from `cbm/mkl4mm.py`, which converts the datasets to CSR format and serves as the baseline for comparison.

> **Note:** `cbm/cbm4mm.py` and `cbm/mkl4mm.py` contain python classes to store matrix **A** in CBM and CSR format, and support matrix products of the form **A** @ **X**.
> Here, **A** is the adjacency matrix of the dataset and **X** is a dense real-valued matrix. 


#### How to Run:
1. Open the `scripts/alpha_searcher.sh` and modify the following variables:
   - `MAX_THREADS=...`  
        Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
        Include in this array the thread counts you want to experiment with.
      
2. Run `./scripts/alpha_searcher.sh` inside the `CBM4Scale/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.

   - `ALPHAS=(...)`  
       Include in this array the alpha values to be considered.

### `./scripts/compression_metrics.sh`
This script evaluates the performance of CBM's compression algorithm using `cbm/cbm4mm.py` via `benchmark/cbm_construction`. Specifically, it measures the time required to convert a matrix to CBM format and calculates the compression ratio relative to the CSR format for each combination of alpha values defined in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array.  

Upon completion, the script generates a results file named `results/compression_metrics_results.txt`, which records the compression time, in seconds, and the achieved compression ratio for each alpha value and dataset combination.

#### How to Run:
1. Open the `scripts/compression_metrics.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
      
2. Run `./scripts/compression_metrics.sh` inside the `CBM4Scale/` direction.

Other configuration options (use default values to reproduce our experiments):   
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
     
   - `ITERATIONS=(...)`  
        Include in this array the number of times dataset should be converted to CBM format..

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.

   - `ALPHAS=(...)`  
       Include in this array the alpha values to be considered.


### `./scripts/matmul.sh`
This script evaluates the performance of different matrix multiplication methods with both CBM and CSR formats using:  
   - `cbm/cbm4{mm,ad,dad}.py` and `cbm/mkl4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.
   - The alpha values used to convert the dataset to CBM format are defined in `benchmark/utilities.py`.

Upon completion, the script generates a results file named `results/matmul_results.txt`, which records time related metrics for matrix multiplication.

> **Note:** `cbm/cbm4ad.py` and `cbm/mkl4ad.py` contain python classes to store matrix **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 

> **Note:** `cbm/cbm4dad.py` and `cbm/mkl4dad.py` contain python classes to store matrix **D^(-1/2)** @ **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **D^(-1/2)** @ **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 


#### How to Run:
1. Open the `scripts/matmul.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  

2. Run `./scripts/matmul.sh` inside the `CBM4Scale/` direction.  

Other configuration options (use default values to reproduce our experiments):    
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.



### `./scripts/inference.sh`
This script evaluates the performance of the CBM format in the context of Graph Convolutional Neural Network (GCN) inference:  
- The graph's laplacian is represented in CBM (`cbm/cbm4dad}.py`) or CSR (`cbm/mkl4dad}.py`) formats.
- The inference itself is executed by `benchmark/benchmark_inference.py`.  
- The alpha values used to convert the dataset to CBM format are defined in `benchmark/utilities.py`.

Upon completion, the script generates a results file named `results/inference_results.txt`, which records the time related metrics for GCN inference.

#### How to Run:
1. Open the `scripts/inference.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
       
2. Run `./scripts/inference.sh` inside the `CBM4Scale/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
        Include in this array the datasets that should be considered..  

   - `NUM_HIDDEN_LAYERS=(...)`  
        Include in this array the number of hidden layers to be added to the GCN.
   
   - `HIDDEN FEATURES=(...)`  
        Include in this array the number of columns to be used in the feature and learnable matrices.
     
   - `EPOCHS=(...)`  
        Include in this array the number of GCN inferences to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup epochs to be run before recording starts.

### `./scripts/validate.sh`
This script validates the correction different matrix multiplication methods with CBM formats using: 
- `cbm/cbm4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.

This validation is performed by comparing the resulting matrices (element-wise) between the classes mentioned above and `cbm/mkl4{mm,ad,dad}.py`.
Again, the alpha values used are the ones set in `benchmark/utilities.py`.

#### How to Run:
1. Open the `scripts/validate.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
       
2. Run `./scripts/valiate.sh` inside the `CBM4Scale/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `RTOL=...`  
        Specifies the relative tolerance interval to be considered during validation.

   - `ATOL=...`  
        Specifies the absolute tolerance interval to be considered in the validation.
