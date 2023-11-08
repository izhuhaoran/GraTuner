import os
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import random_split, DataLoader

import dgl

from data_loader import CustomDataset
from model import AutoGraphModel

lr = 1e-4
batch_size = 32
epoches = 100


def create_dgl_graph(file_path):
    origin_graph = pd.read_csv(file_path, comment='#', sep=' ', header=None)
    
    # 提取源节点列（第一列）
    src_list = origin_graph[0].to_numpy()

    # 提取目标节点列（第二列）
    dst_list = origin_graph[1].to_numpy()
    
    # 生成dgl graph数据
    g = dgl.graph((src_list, dst_list))
    return g

def train():
    f = open("./trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoGraphModel()
    model = model.to(device)
    #model.load_state_dict(torch.load('./resnet.pth'))
  
    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)    
    
    Schedules_Dataset = CustomDataset('/data/zhr_data/AutoGraph/test/pagerank.gt_output.csv')
    
    # 定义训练集和测试集的划分比例
    train_ratio = 0.8
    val_ratio = 0.2

    train_size = int(train_ratio * len(Schedules_Dataset))
    val_size = int(val_ratio * len(Schedules_Dataset))
    # test_size = len(Schedules_Dataset) - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset = random_split(Schedules_Dataset, [train_size, val_size])
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

    for epoch in range(epoches) :
      # Train
      model.train()
      train_loss = 0
      train_loss_cnt = 0 
      for train_batchidx, (graph_name, schedule, runtime) in enumerate(train_data) :
        
        g = create_dgl_graph(graph_name)
        g = g.to(device)
        
        schedule = schedule.to(device)
        runtime = runtime.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        predict = model(g, schedule)
        
        #HingeRankingLoss
        iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
        pred1, pred2 = predict[iu[0]], predict[iu[1]]
        true1, true2 = runtime[iu[0]], runtime[iu[1]]
        sign = (true1-true2).sign()
        
        loss = criterion(pred1, pred2, sign)
        
        train_loss += loss.item()
        train_loss_cnt += 1

        loss.backward()
        optimizer.step()
        
        # print("TrainEpoch: ", epoch, ", Graph: ", graph_name , ", Schedule: ", train_batchidx, ", TrainLoss: ", loss.item())
      
      #Validation
      model.eval()
      with torch.no_grad() :
        valid_loss = 0
        valid_loss_cnt = 0
        for val_batchidx, (graph_name, schedule, runtime) in enumerate(val_data) :
            
            g = create_dgl_graph(graph_name)
            g = g.to(device)
            
            schedule = schedule.to(device)
            runtime = runtime.to(device)
            
            # 前向传播
            predict = model(g, schedule)

            #HingeRankingLoss
            iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
            pred1, pred2 = predict[iu[0]], predict[iu[1]]
            true1, true2 = runtime[iu[0]], runtime[iu[1]]
            sign = (true1-true2).sign()
            loss = criterion(pred1, pred2, sign)
            valid_loss += loss.item()
            valid_loss_cnt += 1
           
            # print("ValidEpoch: ", epoch, ", Graphs: ", graph_name, ", Schedule : ", val_batchidx, ", ValidLoss: ", loss.item())

      
      print ("--- Epoch {} : Train {} Valid {} ---".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      
      f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      f.flush()