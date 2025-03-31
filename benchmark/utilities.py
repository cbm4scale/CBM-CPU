from cbm.cbm4gcn import cbm4gcn
from cbm.cbm4gin import cbm4gin
from cbm.cbm4sage import cbm4sage
from cbm.cbm4mm import cbm4mm
from cbm.cbm4ad import cbm4ad
from cbm.cbm4dad import cbm4dad
from cbm.mkl4gcn import mkl4gcn
from cbm.mkl4gin import mkl4gin
from cbm.mkl4sage import mkl4sage
from cbm.mkl4mm import mkl4mm
from cbm.mkl4ad import mkl4ad
from cbm.mkl4dad import mkl4dad
from nn.gcn import CBMGCN, CBMGCNInference, MKLGCN, MKLGCNInference
from nn.gin import CBMGIN, CBMGINInference, MKLGIN, MKLGINInference
from nn.sage import CBMSAGE, CBMSAGEInference, MKLSAGE, MKLSAGEInference
from ogb.nodeproppred import PygNodePropPredDataset
from torch import sparse_coo_tensor, sparse_csr_tensor, Tensor, zeros, ones, int32, float32, arange, randint, stack, tensor
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, SuiteSparseMatrixCollection, Planetoid
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch_geometric.transforms import OneHotDegree
from torch.nn import MSELoss, CrossEntropyLoss, Sequential, Linear, BatchNorm1d, ReLU
from torch.nn.functional import one_hot
from torch.optim import SGD, Adam


criterions = {
    'mse': MSELoss,
    'cross-entropy': CrossEntropyLoss
}

optimizers = {
    'sgd': SGD,
    'adam': Adam
}


def set_adjacency_matrix(layer, edge_index, alpha):
    if layer in ["pyg-gcn", "pyg-gin", "pyg-sage"]:
        return None, None
    elif layer == "cbm-gcn" or layer == "cbm-gcn-inference":
        return cbm4gcn(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "mkl-gcn" or layer == "mkl-gcn-inference":
        return mkl4gcn(edge_index), None
    elif layer == "cbm-gin" or layer == "cbm-gin-inference":
        return cbm4gin(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "mkl-gin" or layer == "mkl-gin-inference":
        return mkl4gin(edge_index), None
    elif layer == "cbm-sage" or layer == "cbm-sage-inference":
        return cbm4sage(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "mkl-sage" or layer == "mkl-sage-inference":
        return mkl4sage(edge_index), None
    elif layer == "cbm-ax":
        return cbm4mm(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "cbm-adx":
        return cbm4ad(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "cbm-dadx":
        return cbm4dad(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "cbm-mm-add":
        return cbm4gin(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha), None
    elif layer == "mkl-ax":
        return mkl4mm(edge_index), None
    elif layer == "mkl-adx":
        return mkl4ad(edge_index), None
    elif layer == "mkl-dadx":
        return mkl4dad(edge_index), None
    elif layer == "mkl-mm-add":
        return mkl4gin(edge_index), None
    else:
        raise NotImplementedError(f"Layer {layer} is not valid")

def set_layer(layer, in_features, out_features, bias, a, a_t=None):
    # GCN
    if layer == "pyg-gcn":
        return GCNConv(in_channels=in_features, out_channels=out_features, improved=False, cached=True, normalize=True, add_self_loops=False, bias=False)
    elif layer == "cbm-gcn":
        return CBMGCN(in_features, out_features, bias, a)
    elif layer == "mkl-gcn":
        return MKLGCN(in_features, out_features, bias, a)
    elif layer == "cbm-gcn-inference":
        return CBMGCNInference(in_features, out_features, bias, a)
    elif layer == "mkl-gcn-inference":
        return MKLGCNInference(in_features, out_features, bias, a)
    # GIN
    elif layer in ["pyg-gin", "cbm-gin", "mkl-gin", "cbm-gin-inference", "mkl-gin-inference"]:
        eps = 0.
        train_eps = True
        nn = Sequential(
            Linear(in_features, out_features, bias), 
            BatchNorm1d(out_features), 
            ReLU(), 
            Linear(out_features, out_features, bias), 
            ReLU()
        )
        if layer == "pyg-gin":
            return GINConv(nn=nn, eps=eps, train_eps=train_eps)
        elif layer == "cbm-gin":
            return CBMGIN(in_features, a, eps, train_eps, nn)
        elif layer == "mkl-gin":
            return MKLGIN(in_features, a, eps, train_eps, nn)
        elif layer == "cbm-gin-inference":
            return CBMGINInference(in_features, a, eps, train_eps, nn)
        elif layer == "mkl-gin-inference":
            return MKLGINInference(in_features, a, eps, train_eps, nn)
    # SAGE
    elif layer == "pyg-sage":
        return SAGEConv(in_channels=in_features, out_channels=out_features, normalize=False, aggr='mean', root_weight=True, project=False, bias=bias)
    elif layer == "cbm-sage":
        return CBMSAGE(in_features, out_features, bias, a)
    elif layer == "mkl-sage":
        return MKLSAGE(in_features, out_features, bias, a)
    elif layer == "cbm-sage-inference":
        return CBMSAGEInference(in_features, out_features, bias, a)
    elif layer == "mkl-sage-inference":
        return MKLSAGEInference(in_features, out_features, bias, a)
    else:
        raise NotImplementedError(f"Layer {layer} is not valid")


############################################################
######################### DATASETS #########################
############################################################

# examples:
# load_tudataset('PROTEINS', 'node')
# load_tudataset('PROTEINS', 'graph')
# load_tudataset('COLLAB', 'node')
# load_tudataset('COLLAB', 'graph')
# ...
def print_dataset_info(name, dataset):
    print('------------------------------------------------------')
    print(f'Dataset: {name}')
    print(f'Number of Nodes: {dataset.num_nodes}')
    print(f'Number of Edges: {dataset.num_edges}')
    print(f'Number of Nodes Features: {dataset.num_features}')
    print(f'Number of Edges Features: {dataset.num_edge_features}')
    print(f'Number of Classes: {dataset.num_classes}')
    print(f'X: {dataset.x.shape if dataset.x is not None else None}')
    print(f'Y: {dataset.y.shape if dataset.y is not None else None}')
    print('------------------------------------------------------')

def load_tudataset(name, task):     # graph and node prediction
    dataset = Batch.from_data_list(TUDataset(root="../data", name=name))

    # add node features (x) if not present
    if dataset.num_node_features == 0:
        degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
        max_degree = degrees.max().item()
        transform = OneHotDegree(max_degree=max_degree)
        dataset = transform(dataset)
    
    # tudataset is for graph prediction (y.shape = (num_graphs) or (num_graphs, num_classes))
    # if task == "node" then y must be converted to node prediction (y.shape = (num_nodes) or (num_nodes, num_classes))
    # we follow a random approach
    if task == "node":
        # dataset.y = randint(0, 4, (dataset.num_nodes,), dtype=long)
        # dataset.y = ones((dataset.num_nodes, 1), dtype=float32)
        # dataset.num_classes = 4
        # dataset.num_classes = 1
        dataset.num_classes = 2
        y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
        y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
        dataset.y = y
    # assert that num_classes is set (somes datasets loose this information after batching)
    elif task == "graph":
        dataset.num_classes = dataset.y.unique().size(0)
    
    print_dataset_info(name, dataset)
    
    return dataset

def load_snap(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root="../data", name=name, group='SNAP')[0]

    # add node features (x)
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)

    # add node prediction (y)
    dataset.num_classes = 2
    y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
    y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
    dataset.y = y
    
    print_dataset_info(name, dataset)

    return dataset

# examples:
# load_ogbn_proteins(0, 0.5)
# load_ogbn_proteins(1, 0.5)
# ...
# load_ogbn_proteins(7, 0.5)
# load_ogbn_proteins('all', None)
# load_ogbn_proteins(None, None)
def load_ogbn_proteins(edge_attr_feature, edge_attr_threshold):     # node prediction
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root="../data")[0]

    # pick 1 feature and threshold the edge_attr
    if isinstance(edge_attr_feature, int):
        dataset.edge_index = dataset.edge_index[:, dataset.edge_attr[:, edge_attr_feature] >= edge_attr_threshold]
    # average all features row-wise and threshold the edge_attr
    elif edge_attr_feature == 'all':
        # 0.049 is the weighted mean of edge_attr
        edge_attr_threshold = 0.04856166988611221
        dataset.edge_index = dataset.edge_index[:, dataset.edge_attr.mean(dim=1) >= edge_attr_threshold]
    # else just ignore edge_attr and use existing edges
    dataset.edge_attr = None
    dataset.num_edge_features = 0
    dataset.num_edges = dataset.edge_index.size(1)
    dataset.num_classes = dataset.y.size(1)     # this dataset has y.shape = (num_nodes, num_classes)
    dataset.y = dataset.y.to(float32)

    # while graphs that are meant for graph prediction can be converted to node prediction
    # graphs meant for node prediction cannot be converted to graph prediction since we typically only have 1 graph

    # ogbn-proteins is for node prediction but is missing node features (x) so we need to add them
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)
    
    print_dataset_info('ogbn-proteins', dataset)

    return dataset

# examples:
# load_planetoid('PubMed')
def load_planetoid(name):       # node prediction
    dataset = Planetoid(root="../data", name=name)[0]
    dataset.num_classes = dataset.y.unique().size(0)
    dataset.y = one_hot(dataset.y, num_classes=dataset.num_classes).to(float32)
    
    print_dataset_info(name, dataset)

    return dataset

# examples:
# load_dimacs('coPapersCiteseer')
def load_dimacs(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root="../data", name=name, group='DIMACS10')[0]

    # add node features (x)
    degrees = dataset.edge_index[0].bincount(minlength=dataset.num_nodes)
    max_degree = degrees.max().item()
    transform = OneHotDegree(max_degree=max_degree)
    dataset = transform(dataset)

    # add node prediction (y)
    dataset.num_classes = 2
    y = zeros((dataset.num_nodes, dataset.num_classes), dtype=float32)
    y[arange(dataset.num_nodes), randint(0, dataset.num_classes, (dataset.num_nodes,))] = 1
    dataset.y = y
    
    print_dataset_info(name, dataset)

    return dataset


def load_dataset(name):
    if name == "ca-HepPh":
        return load_snap("ca-HepPh"), 4
        #return load_snap("ca-HepPh"), 1
    elif name == "ca-AstroPh":
        return load_snap("ca-AstroPh"), 2
        #return load_snap("ca-AstroPh"), 8
    elif name == "ogbn-proteins-ignore":
        return load_ogbn_proteins('all', None), #16
    elif name == "ogbn-proteins-raw":
        return load_ogbn_proteins(None, None), 8 #16
        #return load_ogbn_proteins(None, None), 1 #16
    elif name == "PubMed":
        return load_planetoid("PubMed"), 4
        #return load_planetoid("PubMed"), 16 #32
    elif name == "Cora":
        return load_planetoid("Cora"), 2 #32
        #return load_planetoid("Cora"), 4 #32
    elif name == "coPapersCiteseer":
        return load_dimacs("coPapersCiteseer"), 4
        #return load_dimacs("coPapersCiteseer"), 32
    elif name == "coPapersDBLP":
        return load_dimacs("coPapersDBLP"), 4 #16
        #return load_dimacs("coPapersDBLP"), 32 #16
    elif name == "COLLAB":
        return load_tudataset("COLLAB", "node"), 4
        #return load_tudataset("COLLAB", "node"), 16
    else:
        raise NotImplementedError(f"Dataset {name} is not valid")