import os
import numpy as np 
import torch
import torch_geometric
import os.path as osp
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.data import Data


##define a function to read file
def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    print(path)
    return read_txt_array(path, sep=',', dtype=dtype)


##define a function to combine items into sequences
def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

##define a funtion to split data into batches
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    print("----row----")
    print(row)
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    ##define a slices
    slices = {'edge_index': edge_slice}
        
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
        
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
            
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    ##for second-order index
    if data.edge_index2 is not None:
        row2, _ = data.edge_index2
        edge_slice2 = torch.cumsum(torch.from_numpy(np.bincount(batch[row2])), 0)
        edge_slice2 = torch.cat([torch.tensor([0]), edge_slice2])
        
        # Edge indices should start at zero for every graph.
        data.edge_index2 -= node_slice[batch[row2]].unsqueeze(0)
        
        ##define a slices
        slices['edge_index2'] = edge_slice2
        slices['edge_attr2'] = edge_slice2

        
    return data, slices

def inspect_raw_file(folder, prefix, name):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()[:10]
            print(f"First 10 lines of {path}:")
            for line in lines:
                print(line.strip())
    else:
        print(f"File not found: {path}")

##IMPORTANT function 1: define a function to read data from text files
def read_tu_data(folder, prefix):
    
    # =============================================================================
    # read edge index from adj matrix
    # =============================================================================
    ##first order adj matrix
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1 
    
    ##second order adj matrix
    edge_index2 = read_file(folder, prefix, 'A2', torch.long).t() - 1 

    print(f"Original edge_index shape: {edge_index.shape}")
    print(f"Min value: {edge_index.min().item()}, Max value: {edge_index.max().item()}")

    if torch.any(edge_index < 0):
        neg_values, counts = torch.unique(edge_index[edge_index < 0], return_counts=True)
        print(f"Negative values found in edge_index: {neg_values.tolist()}")
        print(f"Counts of each negative value: {counts.tolist()}")
        
        neg_edges_mask = (edge_index[0] < 0) | (edge_index[1] < 0)
        neg_edges = edge_index[:, neg_edges_mask]
        print(f"Sample of edges with negative indices (showing up to 10):")
        print(neg_edges[:, :min(10, neg_edges.shape[1])])

    # =============================================================================
    # read graph index
    # =============================================================================    
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    print(batch)
    print(batch.size(0))
    inspect_raw_file(folder, prefix, 'A')

    # =============================================================================
    # read node attributes
    # =============================================================================
    node_attributes = torch.empty((batch.size(0), 0))

    node_attributes = read_file(folder, prefix, 'node_attributes', torch.float32)
    
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    # =============================================================================
    # read edge attributes
    # =============================================================================
    ##first-order edge attributes
    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)
        
    ##second-order edge attributes
    edge_attributes2 = torch.empty((edge_index2.size(1), 0))
    edge_attributes2 = read_file(folder, prefix, 'edge_attributes2')
    if edge_attributes2.dim() == 1:
        edge_attributes2 = edge_attributes2.unsqueeze(-1)

    # =============================================================================
    # concategate node attributes
    # =============================================================================
    x = cat([node_attributes])
    
    print("-------------x------------")
    print(x)
    # =============================================================================
    # concategate edge attributes and edge lables
    # =============================================================================
    ##first-order edge attributes
    edge_attr = cat([edge_attributes])

    ##second-order edge attributes
    edge_attr2 = cat([edge_attributes2])
  
    # =============================================================================
    # read graph attributes or graph labels
    # =============================================================================
    y = read_file(folder, prefix, 'graph_labels', torch.long)
    
    # =============================================================================
    # get total number of nodes for all graphs
    # =============================================================================
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    # Check for negative indices and filter them out
    if torch.any(edge_index < 0):
        print(f"Warning: Found negative indices in edge_index. Filtering them out.")
        valid_edges = edge_index.min(dim=0)[0] >= 0
        edge_index = edge_index[:, valid_edges]
        if edge_attr is not None:
            edge_attr = edge_attr[valid_edges]

    # Similarly check and filter second-order indices
    if torch.any(edge_index2 < 0):
        print(f"Warning: Found negative indices in edge_index2. Filtering them out.")
        valid_edges2 = edge_index2.min(dim=0)[0] >= 0
        edge_index2 = edge_index2[:, valid_edges2]
        if edge_attr2 is not None:
            edge_attr2 = edge_attr2[valid_edges2]

    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    
    # print(edge_index)
    # print(edge_index.size(1))

        
    ##second-order   
    # edge_index2, edge_attr2 = remove_self_loops(edge_index2, edge_attr2) 
    edge_index2, edge_attr2 = coalesce(edge_index2, edge_attr2, num_nodes)

    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2
        
    # print("_______________important info 0_______________")
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_index2)
    # print(data.edge_attr)
    # print(data.edge_attr2)
    # print("____________________________________________")
    
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_attributes2': edge_attributes2.size(-1)
    }

    
    # print("_______________important info 1_______________")
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_index2)
    # print(data.edge_attr)
    # print(data.edge_attr2)
    # print("____________________________________________")
    
    return data, slices, sizes


class ParseDataset(InMemoryDataset):
    
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        load_data = torch.load(self.processed_paths[0])
              
        self.data, self.slices, self.sizes = load_data
        
        num_node_attributes = self.num_node_attributes
        self.data.x = self.data.x[:, :num_node_attributes]
        
        num_edge_attrs = self.num_edge_attributes
        self.data.edge_attr = self.data.edge_attr[:, :num_edge_attrs]
        
        num_edge_attrs2 = self.num_edge_attributes2
        self.data.edge_attr2 = self.data.edge_attr2[:, :num_edge_attrs2]
        
        # print("_______________important info 3_______________")
        # print(self.data.x)
        # print(self.data.edge_index)
        # print(self.data.edge_index2)
        # print(self.data.edge_attr)
        # print(self.data.edge_attr2)
        # print("____________________________________________")

    @property
    def raw_dir(self) -> str:
        name = f'Raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']
    
    @property
    def num_edge_attributes2(self) -> int:
        return self.sizes['num_edge_attributes2']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    ##renove ~/processed/ directory to run this 
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
 
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])
        # print("_______________important info 2_______________")
        # print(self.data.x)
        # print(self.data.edge_index)
        # print(self.data.edge_index2)
        # print(self.data.edge_attr)
        # print(self.data.edge_attr2)
        # print("____________________________________________")

        
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
   

# =============================================================================
# Step 2: define a function to create data loader based on ParseDataset class
# =============================================================================

from torch_geometric.data import DataLoader, DenseDataLoader

DATA_PATH = '/home/rcp_user/suzhang/graph/logade/Data/'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)


##IMPORTANT function: define a function to load dataset
def load_data(data_name, 
              dense=False, 
              seed=1213, 
              save_indices_to_disk=True):
    
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    ##get raw dataset if it already exists
    # print(DATA_PATH + "/" + data_name + "/Raw/")
    # if os.path.exists(DATA_PATH + "/" + data_name + "/Raw/"):
    raw_path = os.path.join(DATA_PATH, data_name, "Raw")
    print(raw_path)
    if os.path.exists(raw_path):        
        print("++++++++find dataset++++++++++++")
        
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)
    else:
        raise NotImplementedError

    dataset = dataset_raw
    
    dataset_list = [data for data in dataset]
    
    print(f"Total dataset size: {len(dataset_list)}")
    print(f"Normal samples: {sum(1 for data in dataset_list if data.y.item()==0)}")
    print(f"Anomaly samples: {sum(1 for data in dataset_list if data.y.item()==1)}")
    
    normal_indices = [i for i, data in enumerate(dataset_list) if data.y.item()==0]
    np.random.shuffle(normal_indices)  
    
    train_size = int(len(normal_indices) * 0.8)
    train_indices = normal_indices[:train_size]
    test_normal_indices = normal_indices[train_size:]
    
    anomaly_indices = [i for i, data in enumerate(dataset_list) if data.y.item()==1]
    
    test_indices = test_normal_indices + anomaly_indices
    
    train_dataset = [dataset_list[idx] for idx in train_indices]
    test_dataset = [dataset_list[idx] for idx in test_indices]
    
 
    return train_dataset, test_dataset, dataset_raw


##define a function as dataloader
def create_loaders(data_name, 
                   batch_size=64, 
                   dense=False, 
                   data_seed=1213):

    ##generate training dataset and testing dataset using predefined function 
    train_dataset, test_dataset, dataset_raw= load_data(data_name, 
                                                        dense=dense, 
                                                        seed=data_seed)



    print("After downsampling and test-train splitting, distribution of classes:")
    labels = np.array([data.y.item() for data in train_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TRAIN: Number of graphs: %d, Class distribution %s"%(len(train_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TEST: Number of graphs: %d, Class distribution %s"%(len(test_dataset), label_dist))

    Loader = DenseDataLoader if dense else DataLoader
    
    num_workers = 0
    
    ##----create a batch-based training dataset loader----##
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    ##----create a batch-based testing dataset loader----##
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, train_dataset[0].num_features, train_dataset, test_dataset, dataset_raw

