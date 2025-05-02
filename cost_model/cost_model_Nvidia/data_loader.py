import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


kernel_fusion_map = {
    'DISABLED': 0,
    'ENABLED': 1
}

LB_map = {
    'VERTEX_BASED' : 0,
    'TWC' : 1,
    'TWCE' : 2,
    'WM' : 3,
    'CM' : 4,
}

EB_map = {
    'None' : 0,
    'ENABLED' : 1,
    'DISABLED' : 2,
}

direction_map = {
    'PUSH' : 0,
    'PULL' : 1,
}

dedup_map = {
    'ENABLED' : 0,
    'DISABLED' : 1,
}

frontier_output_map = {
    'FUSED' : 0,
    'UNFUSED_BITMAP' : 1,
    'UNFUSED_BOOLMAP' : 2,
}

pull_rep_map = {
    'None' : 0,
    'BITMAP' : 1,
    'BOOLMAP' : 2,
}


class ScheduleDataset_v2(Dataset):
    def __init__(self, origin_data):
        schedules = []
        runtimes =[]
        
        for data in origin_data:
            # graphs.append(data[0])
            
            sche_tmp = [0, 0, 0, 0, 0, 0]
            sche_tmp[0] = kernel_fusion_map[data[1]]
            sche_tmp[1] = LB_map[data[2]]
            # sche_tmp[2] = EB_map[data[3]]
            # sche_tmp[3] = int(data[4])
            sche_tmp[2] = direction_map[data[5]]
            sche_tmp[3] = dedup_map[data[6]]
            sche_tmp[4] = frontier_output_map[data[7]]
            sche_tmp[5] = pull_rep_map[data[8]]

            schedules.append(sche_tmp)
    
            # runtimes.append(data[9])
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

