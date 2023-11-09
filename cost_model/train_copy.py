import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import random_split, DataLoader

import dgl

from data_loader import ScheduleDataset, GraphsDataset, ScheduleGraphDataset
from model import AutoGraphModel

lr = 1e-4
batch_size = 8
epoches = 100

graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
              'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
              'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
               'youtube-groupmemberships', 
              ]



def create_dgl_graph(file_name):
    file_name = file_name[0]
    file_path = f'/home/zhuhaoran/AutoGraph/graphs/{file_name}/{file_name}.el'
    origin_graph = pd.read_csv(file_path, comment='#', sep=' ', header=None)

    # 提取源节点列（第一列）
    src_list = origin_graph[0].to_numpy()

    # 提取目标节点列（第二列）
    dst_list = origin_graph[1].to_numpy()

    # 生成dgl graph数据
    g = dgl.graph((src_list, dst_list))
    return g

def create_dgl_graph_batch(file_name_batch):
    if file_name_batch.size() > 0:
        pass
    file_name = file_name[0]
    file_path = f'/home/zhuhaoran/AutoGraph/graphs/{file_name}/{file_name}.el'
    origin_graph = pd.read_csv(file_path, comment='#', sep=' ', header=None)

    # 提取源节点列（第一列）
    src_list = origin_graph[0].to_numpy()

    # 提取目标节点列（第二列）
    dst_list = origin_graph[1].to_numpy()

    # 生成dgl graph数据
    g = dgl.graph((src_list, dst_list))
    return g


def train():
    f = open("./trainlog.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    Graphs_Dataset = GraphsDataset()
    train_data_graph = DataLoader(Graphs_Dataset, batch_size=1, shuffle=True)

    # Schedules_Dataset = ScheduleDataset(
    #     '/home/zhuhaoran/AutoGraph/AutoGraph/test/pagerank.gt_output.csv')

    # # 定义训练集和测试集的划分比例
    # train_ratio = 0.8
    # val_ratio = 0.2

    # train_size = int(train_ratio * len(Schedules_Dataset))
    # # val_size = int(val_ratio * len(Schedules_Dataset))
    # val_size = len(Schedules_Dataset) - train_size

    # # test_size = len(Schedules_Dataset) - train_size - val_size

    # # 划分数据集
    # train_dataset, val_dataset = random_split(
    #     Schedules_Dataset, [train_size, val_size])

    # train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        for graph_batchidx, graph_name in enumerate(train_data_graph):
            torch.cuda.empty_cache()

            Schedules_Dataset = ScheduleDataset(graph_name[0])

            # 定义训练集和测试集的划分比例
            train_ratio = 0.8
            val_ratio = 0.2

            train_size = int(train_ratio * len(Schedules_Dataset))
            # val_size = int(val_ratio * len(Schedules_Dataset))
            val_size = len(Schedules_Dataset) - train_size

            # 划分数据集
            train_dataset, val_dataset = random_split(
                Schedules_Dataset, [train_size, val_size])

            train_data_schedule = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True)
            
            g = create_dgl_graph(graph_name)
            g = g.to(device)
            
            # Train
            model.train()

            for train_batchidx, (schedule, runtime) in enumerate(train_data_schedule):
                schedule = schedule.to(device)
                runtime = runtime.to(device)

                optimizer.zero_grad()
                
                graph_feature = model.embed_sparse_matrix(g)
                graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                # 前向传播
                # predict = model(g, schedule)
                predict = model.forward_after_query(graph_feature, schedule)
                # print(predict)

                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()

                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()

                print("TrainEpoch: ", epoch, ", Graph: ", graph_name , ", Schedule: ", train_batchidx, ", TrainLoss: ", loss.item())

            # Validation
            model.eval()
            with torch.no_grad():

                for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)
                    
                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    # predict = model(g, schedule)
                    predict = model.forward_after_query(graph_feature, schedule)

                    # HingeRankingLoss
                    iu = torch.triu_indices(
                        predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    
                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1

                    print("ValidEpoch: ", epoch, ", Graph: ", graph_name, ", Schedule : ", val_batchidx, ", ValidLoss: ", loss.item())

        print("--- Epoch {} : Train {} Valid {} ---".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))

        f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
        f.flush()


def train_v2():
    f = open("./trainlog.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        for graph_batchidx, graph_name in enumerate(train_data_graph):
            torch.cuda.empty_cache()

            Schedules_Dataset = ScheduleDataset(graph_name[0])

            # 定义训练集和测试集的划分比例
            train_ratio = 0.8
            val_ratio = 0.2

            train_size = int(train_ratio * len(Schedules_Dataset))
            # val_size = int(val_ratio * len(Schedules_Dataset))
            val_size = len(Schedules_Dataset) - train_size

            # 划分数据集
            train_dataset, val_dataset = random_split(
                Schedules_Dataset, [train_size, val_size])

            train_data_schedule = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True)
            
            g = create_dgl_graph(graph_name)
            g = g.to(device)
            
            # Train
            model.train()

            for train_batchidx, (schedule, runtime) in enumerate(train_data_schedule):
                schedule = schedule.to(device)
                runtime = runtime.to(device)

                optimizer.zero_grad()
                
                graph_feature = model.embed_sparse_matrix(g)
                graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                # 前向传播
                # predict = model(g, schedule)
                predict = model.forward_after_query(graph_feature, schedule)
                # print(predict)

                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()

                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()

                print("TrainEpoch: ", epoch, ", Graph: ", graph_name , ", Schedule: ", train_batchidx, ", TrainLoss: ", loss.item())

            # Validation
            model.eval()
            with torch.no_grad():

                for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)
                    
                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    # predict = model(g, schedule)
                    predict = model.forward_after_query(graph_feature, schedule)

                    # HingeRankingLoss
                    iu = torch.triu_indices(
                        predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    
                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1

                    print("ValidEpoch: ", epoch, ", Graph: ", graph_name, ", Schedule : ", val_batchidx, ", ValidLoss: ", loss.item())

        print("--- Epoch {} : Train {} Valid {} ---".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))

        f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
        f.flush()


def collate_fn(batch):
    # 对一个批次的数据进行处理，组合成一个批次的张量
    graphs, schedules, runtimes = zip(*batch)
    
    # 使用 DGL 提供的 `batch` 函数将图列表转换为一个大图（批图）
    batched_graph = dgl.batch(graphs)
    
    # 返回批次的张量
    return batched_graph, torch.stack(schedules), torch.stack(runtimes)

# 这个训练方法不行，out of memory
def train_new():
    f = open("./trainlog.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    # Graphs_Dataset = GraphsDataset()
    # train_data_graph = DataLoader(Graphs_Dataset, batch_size=1, shuffle=True)

    Schedules_Dataset = ScheduleGraphDataset(
        '/home/zhuhaoran/AutoGraph/AutoGraph/test/pagerank.gt_output.csv')

    # 定义训练集和测试集的划分比例
    train_ratio = 0.8
    val_ratio = 0.2

    train_size = int(train_ratio * len(Schedules_Dataset))
    # val_size = int(val_ratio * len(Schedules_Dataset))
    val_size = len(Schedules_Dataset) - train_size

    # test_size = len(Schedules_Dataset) - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset = random_split(
        Schedules_Dataset, [train_size, val_size])

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        

        torch.cuda.empty_cache()
        
        # Train
        model.train()

        for train_batchidx, (graphs, schedule, runtime) in enumerate(train_data):
            graphs = graphs.to(device)
            schedule = schedule.to(device)
            runtime = runtime.to(device)

            optimizer.zero_grad()
            
            # graph_feature = model.embed_sparse_matrix(graphs)
            # graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

            # 前向传播
            predict = model(graphs, schedule)
            # predict = model.forward_after_query(graph_feature, schedule)
            print(predict)

            # HingeRankingLoss
            iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
            
            pred1, pred2 = predict[iu[0]], predict[iu[1]]
            true1, true2 = runtime[iu[0]], runtime[iu[1]]
            sign = (true1-true2).sign()

            loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

            train_loss += loss.item()
            train_loss_cnt += 1

            loss.backward()
            optimizer.step()

            print("TrainEpoch: ", epoch , ", Schedule: ", train_batchidx, ", TrainLoss: ", loss.item())

        # Validation
        model.eval()
        with torch.no_grad():

            for val_batchidx, (graphs, schedule, runtime) in enumerate(val_data):

                schedule = schedule.to(device)
                runtime = runtime.to(device)
                
                # graph_feature = model.embed_sparse_matrix(graphs)
                # graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                # 前向传播
                predict = model(graphs, schedule)
                # predict = model.forward_after_query(graph_feature, schedule)

                # HingeRankingLoss
                iu = torch.triu_indices(
                    predict.shape[0], predict.shape[0], 1)
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                
                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                valid_loss += loss.item()
                valid_loss_cnt += 1

                print("ValidEpoch: ", epoch, ", Schedule : ", val_batchidx, ", ValidLoss: ", loss.item())

        print("--- Epoch {} : Train {} Valid {} ---".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))

        f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
        f.flush()


if __name__ == '__main__':
    train_new()
