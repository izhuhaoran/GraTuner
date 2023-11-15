import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import dgl

direction_map = {
    'SparsePush': 0, 
    'DensePull' : 1, 
    'DensePull-SparsePush' : 2, 
    'SparsePush-DensePull' : 3, 
    'DensePush-SparsePush' : 4
}
parallelization_map = {
    'serial' : 0, 
    'dynamic-vertex-parallel' : 1, 
    'static-vertex-parallel' : 2, 
    'edge-aware-dynamic-vertex-parallel' : 3
}
DenseVertexSet_map = {
    'boolean-array' : 0, 
    'bitvector' : 1
}

SSGNum_map = {
    '0': 0,
    '5': 1,
    '10': 2,
    '15': 3,
    '20': 4
}

direction_map_onehot = {
    'SparsePush': [1, 0, 0, 0, 0], 
    'DensePull' : [0, 1, 0, 0, 0], 
    'DensePull-SparsePush' : [0, 0, 1, 0, 0], 
    'SparsePush-DensePull' : [0, 0, 0, 1, 0], 
    'DensePush-SparsePush' : [0, 0, 0, 0, 1]
}
parallelization_map_onehot = {
    'serial' : [1, 0, 0, 0], 
    'dynamic-vertex-parallel' : [0, 1, 0, 0], 
    'static-vertex-parallel' : [0, 0, 1, 0], 
    'edge-aware-dynamic-vertex-parallel' : [0, 0, 0, 1]
}
DenseVertexSet_map_onehot = {
    'boolean-array' : [1, 0], 
    'bitvector' : [0, 1]
}

SSGNum_map_onehot = {
    '0': [1, 0, 0, 0, 0],
    '5': [0, 1, 0, 0, 0],
    '10': [0, 0, 1, 0, 0],
    '15': [0, 0, 0, 1, 0],
    '20': [0, 0, 0, 0, 1]
}

# config_bucket_update_strategy = ['eager_priority_update', 'eager_priority_update_with_merge', 'lazy_priority_update']
# config_NUMA = ['false', 'static-parallel', 'dynamic-parallel']

graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
              'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
              'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
               'youtube-groupmemberships', 
              ]


def create_dgl_graph(file_name):
    # file_name = file_name[0]
    file_path = f'/home/zhuhaoran/AutoGraph/graphs/{file_name}/{file_name}.el'
    origin_graph = pd.read_csv(file_path, comment='#', sep=' ', header=None)

    # 提取源节点列（第一列）
    src_list = origin_graph[0].to_numpy()

    # 提取目标节点列（第二列）
    dst_list = origin_graph[1].to_numpy()

    # 生成dgl graph数据
    g = dgl.graph((src_list, dst_list))
    return g

def schedule_preprocess(file_path):
    origin_schedule_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    origin_schedule_data = origin_schedule_data.values
    return origin_schedule_data


class GraphsDataset(Dataset):
    def __init__(self):
        self.graphs_list = graph_list
        
    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, index):
        return self.graphs_list[index]


class ScheduleDataset(Dataset):
    def __init__(self, graph_name, algo_name='pagerank'):
        file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/test/grouped/{graph_name}_{algo_name}.gt_output.csv'
        
        schedules = []
        runtimes =[]
        
        # origin_data = schedule_preprocess(file_path)
        origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
        origin_data = origin_data.values
        
        for data in origin_data:
            # graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[1]]
            sche_tmp[1] = parallelization_map[data[2]]
            sche_tmp[2] = DenseVertexSet_map[data[3]]
            sche_tmp[3] = SSGNum_map[str(data[5])]
            schedules.append(sche_tmp)
    
            runtimes.append(data[6])
        
        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        
        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)
        
        # # Normalize
        # self.runtimes = self.runtimes / 1000.0

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)
        # print(1)
        

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.schedules[index], self.runtimes[index]


class ScheduleDataset_Onehot(Dataset):
    def __init__(self, origin_data):
        schedules = []
        runtimes =[]
        
        for data in origin_data:
            # graphs.append(data[0])
            
            sche_tmp_0 = direction_map_onehot[data[1]]
            sche_tmp_1 = parallelization_map_onehot[data[2]]
            sche_tmp_2 = DenseVertexSet_map_onehot[data[3]]
            sche_tmp_3 = SSGNum_map_onehot[str(data[5])]
            
            sche_tmp = np.concatenate((sche_tmp_0, sche_tmp_1, sche_tmp_2, sche_tmp_3))
            schedules.append(sche_tmp)
    
            runtimes.append(data[6])
        
        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        
        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)
        
        # # Normalize
        # self.runtimes = self.runtimes / 1000.0

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.schedules[index], self.runtimes[index]


class ScheduleDataset_v2(Dataset):
    def __init__(self, origin_data):
        schedules = []
        runtimes =[]
        
        for data in origin_data:
            # graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[1]]
            sche_tmp[1] = parallelization_map[data[2]]
            sche_tmp[2] = DenseVertexSet_map[data[3]]
            sche_tmp[3] = SSGNum_map[str(data[5])]
            schedules.append(sche_tmp)
    
            runtimes.append(data[6])
        
        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        
        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)
        
        # # Normalize
        # self.runtimes = self.runtimes / 1000.0

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.schedules[index], self.runtimes[index]


class ScheduleGraphDataset(Dataset):
    def __init__(self, file_path):
        graphs = []
        schedules = []
        runtimes =[]
        
        origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
        origin_data = origin_data.values
        
        for data in origin_data:
            graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[1]]
            sche_tmp[1] = parallelization_map[data[2]]
            sche_tmp[2] = DenseVertexSet_map[data[3]]
            sche_tmp[3] = SSGNum_map[str(data[5])]
            schedules.append(sche_tmp)
    
            runtimes.append(data[6])
        
        # graphs = np.stack(graphs, axis=0)
        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        
        self.graphs = graphs
        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)
        
        # # Normalize
        # self.runtimes = self.runtimes / 1000.0

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)
        

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        g = create_dgl_graph(self.graphs[index])
        return g, self.schedules[index], self.runtimes[index]
    
    