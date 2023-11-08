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

# config_bucket_update_strategy = ['eager_priority_update', 'eager_priority_update_with_merge', 'lazy_priority_update']
# config_NUMA = ['false', 'static-parallel', 'dynamic-parallel']

graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
              'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
              'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
               'youtube-groupmemberships', 
              ]


def schedule_preprocess(file_path):
    origin_schedule_data = pd.read_csv(file_path, comment='#', sep=', ', header=None)
    origin_schedule_data = origin_schedule_data.values
    return origin_schedule_data


class CustomDataset(Dataset):
    def __init__(self, file_path):
        graphs = []
        schedules = []
        runtimes =[]
        
        origin_data = schedule_preprocess(file_path)
        
        for data in origin_data:
            graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[data[1]]
            sche_tmp[1] = parallelization_map[data[2]]
            sche_tmp[2] = DenseVertexSet_map[data[3]]
            sche_tmp[3] = data[5]
            schedules.append(sche_tmp)
    
            runtimes.append(data[6])
        
        graphs = np.stack(graphs, axis=0)
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
        
        print(self.schedules)
        print(self.runtimes)
        

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, index):
        return self.graphs[index], self.schedules[index], self.runtimes[index]

testdata = CustomDataset('/data/zhr_data/AutoGraph/test/pagerank.gt_output.csv')