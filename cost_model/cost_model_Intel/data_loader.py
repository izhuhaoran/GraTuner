import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


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



class ScheduleDataset(Dataset):
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
    
            # runtimes.append(data[6])
            runtimes.append(data[-1])

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


class ScheduleDataset_mkl(Dataset):
    def __init__(self, origin_data):
        schedules = []
        runtimes_cpu =[]
        runtimes_yitian =[]
        
        for data in origin_data:
            # graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[1]]
            sche_tmp[1] = parallelization_map[data[2]]
            sche_tmp[2] = DenseVertexSet_map[data[3]]
            sche_tmp[3] = SSGNum_map[str(data[5])]
            schedules.append(sche_tmp)
    
            runtimes_cpu.append(data[-3])
            runtimes_yitian.append(data[-1])
        
        schedules = np.stack(schedules, axis=0)
        runtimes_cpu = np.stack(runtimes_cpu, axis=0)
        runtimes_yitian = np.stack(runtimes_yitian, axis=0)

        self.schedules = schedules.astype(np.float32)
        self.runtimes_intel = runtimes_cpu.astype(np.float32)
        self.runtimes_yitian = runtimes_yitian.astype(np.float32)

        # # Normalize
        # self.runtimes = self.runtimes / 1000.0

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes_intel = torch.from_numpy(self.runtimes_intel)
        self.runtimes_yitian = torch.from_numpy(self.runtimes_yitian)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.schedules[index], self.runtimes_intel[index], self.runtimes_yitian[index]

class ScheduleDataset_all_algo(Dataset):
    def __init__(self, origin_data):
        schedules = []
        runtimes =[]
        algos = []

        algo_op_dict = {
            'bfs': {'msg_create': 0, 'msg_reduce': 0, 'compute_mode': 0},   # first, =, 部分激活
            'pagerank': {'msg_create': 0, 'msg_reduce': 2, 'compute_mode': 1}, # first, +=, 全图计算
            'sssp': {'msg_create': 1, 'msg_reduce': 1, 'compute_mode': 0},  # +, min=, 部分激活
            'cc': {'msg_create': 0, 'msg_reduce': 1, 'compute_mode': 0},  # first, min=, 部分激活
        }
        
        for data in origin_data:
            # algo
            msg_create, msg_reduce, compute_mode = algo_op_dict[data[0]].values()
            algos.append([msg_create, msg_reduce, compute_mode])

            # graph data
            # graphs.append(data[1])
            
            # schedule
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[2]]
            sche_tmp[1] = parallelization_map[data[3]]
            sche_tmp[2] = DenseVertexSet_map[data[4]]
            sche_tmp[3] = SSGNum_map[str(data[6])]
            schedules.append(sche_tmp)
    
            runtimes.append(data[7])
        
        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        algos = np.stack(algos, axis=0)

        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)
        self.algos = algos.astype(np.float32)

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)
        self.algos = torch.from_numpy(self.algos)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.algos[index], self.schedules[index], self.runtimes[index]

