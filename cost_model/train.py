import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import random_split, DataLoader

import dgl

from data_loader import ScheduleDataset, GraphsDataset, ScheduleGraphDataset, ScheduleDataset_v2, ScheduleDataset_Onehot
from model import AutoGraphModel, AutoGraphModel_GAT, AutoGraphModel_Onehot


lr = 1e-4
batch_size = 32
epoches = 200


graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
              'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
              'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
               'youtube-groupmemberships', 
              ]


def eval_rank_quality(pred, true_y):
    if len(pred) != len(true_y):
        return None
    
    iu = torch.triu_indices(len(pred), len(pred), 1)

    pred1, pred2 = pred[iu[0]], pred[iu[1]]
    t1, t2 = true_y[iu[0]], true_y[iu[1]]

    y1 = torch.sign(pred1-pred2)
    y2 = torch.sign(t1-t2)

    # 计算相等元素的数目
    equal_count = (y1 == y2).sum().item()

    # 计算不等元素的数目
    unequal_count = (y1 != y2).sum().item()
    return equal_count, unequal_count


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

def load_data(file_path):
    data = {}
    # origin_data = schedule_preprocess(file_path)
    origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    
    # 根据第一列的不同值进行分组
    origin_data = origin_data.groupby(0)
    
    # 遍历分组
    for name, group in origin_data:
        data[name] = group.values
    
    return data


def train():

    f = open("./trainlog.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    
    checkpoint = torch.load("/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel.pth")
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    Graphs_Dataset = GraphsDataset()
    train_data_graph = DataLoader(Graphs_Dataset, batch_size=1, shuffle=True)


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
            
            g = create_dgl_graph(graph_name[0])
            g = g.to(device)
            
            # Train
            model.train()

            for train_batchidx, (schedule, runtime) in enumerate(train_data_schedule):
                schedule = schedule.to(device)
                runtime = runtime.to(device)

                optimizer.zero_grad()
                
                graph_feature = model.embed_sparse_matrix(g)    # graph_feature [1, 128]
                # 将graph_feature 复制扩充到 [batch, 128]
                graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                # 前向传播
                predict = model.forward_after_query(graph_feature, schedule)

                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                
                sign = (true1-true2).sign()
                sign[sign==0] = 1

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
                    # 8， 8*1
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
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
        
        torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel.pth")
        print("save over.")

def train_v2():
    algo_name = 'pagerank'
    train_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo_name}.csv'
    val_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/val/{algo_name}.csv'
    
    train_all_data = load_data(train_file_path)
    val_all_data = load_data(val_file_path)
    
    f = open("./trainlog_v2_new.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    
    # checkpoint = torch.load("/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_v2.pth")
    # model.load_state_dict(checkpoint)
    
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num = 0
        rank_all_num = 0
        
        valid_accuracy_best = 0.0
        
        for graph_name in graph_list:
            # torch.cuda.empty_cache()

            Train_Schedules_Dataset = ScheduleDataset_v2(train_all_data[graph_name])
            Val_Schedules_Dataset = ScheduleDataset_v2(val_all_data[graph_name])

            train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            
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
                predict = model.forward_after_query(graph_feature, schedule)
                
                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                sign[sign==0] = 1

                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()
                
                # 计算排序情况
                pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                equal_count = (sign == pred_sign).sum().item()
                # unequal_count = (sign != pred_sign).sum().item()
                
                accuracy = equal_count / len(sign)

                print("-------- Epoch", epoch, "Train Batch: ", train_batchidx, ", Graph: ", graph_name , ", TrainLoss: ", loss.item(), ", Accuracy: ", accuracy)
                
            rank_equal_num_graph = 0
            rank_all_num_graph = 0

            # Validation
            model.eval()
            with torch.no_grad():
                for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)

                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    predict = model.forward_after_query(graph_feature, schedule)
                    if predict.shape[0] < 2:
                        continue
                    # HingeRankingLoss
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    sign[sign==0] = 1

                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    # 计算排序情况
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    # unequal_count = (sign != pred_sign).sum().item()
                    
                    accuracy = equal_count / len(sign)
                    
                    rank_equal_num += equal_count
                    rank_all_num += len(sign)
                    
                    rank_equal_num_graph += equal_count
                    rank_all_num_graph += len(sign)
                    print("-------- Epoch", epoch, "Valid Batch: ", val_batchidx, ", Graph: ", graph_name, ", ValidLoss: ", loss.item(), ", Accuracy : ", accuracy)
                    
            valid_accuracy_graph = rank_equal_num_graph / rank_all_num_graph
            print("--- Epoch {} : graph {} Valid Acc {}---".format(epoch, graph_name, valid_accuracy_graph))

        valid_accuracy = rank_equal_num / rank_all_num
        
        print("--- Epoch {} : Train {} Valid {} Acc {}---\n".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))

        f.write("--- Epoch {} : Train {} Valid {} Acc {}---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))
        f.flush()
        
        if valid_accuracy_best < valid_accuracy:
            torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_v2_new_best.pth")
            valid_accuracy_best = valid_accuracy
            
        torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_v2_new.pth")
        # print("save over.")

def train_onehot():
    algo_name = 'pagerank'
    train_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo_name}.csv'
    val_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/val/{algo_name}.csv'
    
    train_all_data = load_data(train_file_path)
    val_all_data = load_data(val_file_path)
    
    f = open("./trainlog_onehot.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel_Onehot()
    
    # checkpoint = torch.load("/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_onehot.pth")
    # model.load_state_dict(checkpoint)
    
    model = model.to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num = 0
        rank_all_num = 0
        
        valid_accuracy_best = 0.0
        
        for graph_name in graph_list:
            # torch.cuda.empty_cache()

            Train_Schedules_Dataset = ScheduleDataset_Onehot(train_all_data[graph_name])
            Val_Schedules_Dataset = ScheduleDataset_Onehot(val_all_data[graph_name])

            train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            
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
                predict = model.forward_after_query(graph_feature, schedule)
                
                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                sign[sign==0] = 1

                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()
                
                # 计算排序情况
                pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                equal_count = (sign == pred_sign).sum().item()
                # unequal_count = (sign != pred_sign).sum().item()
                
                accuracy = equal_count / len(sign)

                print("-------- Epoch", epoch, "Train Batch: ", train_batchidx, ", Graph: ", graph_name , ", TrainLoss: ", loss.item(), ", Accuracy: ", accuracy)
                
            rank_equal_num_graph = 0
            rank_all_num_graph = 0

            # Validation
            model.eval()
            with torch.no_grad():
                for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)

                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    predict = model.forward_after_query(graph_feature, schedule)
                    if predict.shape[0] < 2:
                        continue
                    # HingeRankingLoss
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    sign[sign==0] = 1

                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    # 计算排序情况
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    # unequal_count = (sign != pred_sign).sum().item()
                    
                    accuracy = equal_count / len(sign)
                    
                    rank_equal_num += equal_count
                    rank_all_num += len(sign)
                    
                    rank_equal_num_graph += equal_count
                    rank_all_num_graph += len(sign)
                    print("-------- Epoch", epoch, "Valid Batch: ", val_batchidx, ", Graph: ", graph_name, ", ValidLoss: ", loss.item(), ", Accuracy : ", accuracy)
                    
            valid_accuracy_graph = rank_equal_num_graph / rank_all_num_graph
            print("--- Epoch {} : graph {} Valid Acc {}---".format(epoch, graph_name, valid_accuracy_graph))

        valid_accuracy = rank_equal_num / rank_all_num
        
        print("--- Epoch {} : Train {} Valid {} Acc {}---\n".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))

        f.write("--- Epoch {} : Train {} Valid {} Acc {}---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))
        f.flush()
        
        if valid_accuracy_best < valid_accuracy:
            torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_onehot_best.pth")
            valid_accuracy_best = valid_accuracy
            
        torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_onehot.pth")
        # print("save over.")

def train_GAT():
    algo_name = 'pagerank'
    train_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo_name}.csv'
    val_file_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/val/{algo_name}.csv'
    
    train_all_data = load_data(train_file_path)
    val_all_data = load_data(val_file_path)
    
    f = open("./trainlog_gat.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel_GAT()
    
    # checkpoint = torch.load("/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_gat.pth")
    # model.load_state_dict(checkpoint)
    
    model = model.to(device)
    # model.load_state_dict(torch.load('./resnet.pth'))

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num = 0
        rank_all_num = 0
        
        valid_accuracy_best = 0.0
        
        for graph_name in graph_list:
            # torch.cuda.empty_cache()

            Train_Schedules_Dataset = ScheduleDataset_v2(train_all_data[graph_name])
            Val_Schedules_Dataset = ScheduleDataset_v2(val_all_data[graph_name])

            train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            
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
                predict = model.forward_after_query(graph_feature, schedule)
                
                # HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                sign[sign==0] = 1

                loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()
                
                # 计算排序情况
                pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                equal_count = (sign == pred_sign).sum().item()
                # unequal_count = (sign != pred_sign).sum().item()
                
                accuracy = equal_count / len(sign)

                print("TrainBatch: ", epoch, ", Graph: ", graph_name , ", TrainLoss: ", loss.item(), ", Accuracy: ", accuracy)

            # Validation
            model.eval()
            with torch.no_grad():
                for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)
                    
                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    predict = model.forward_after_query(graph_feature, schedule)

                    # HingeRankingLoss
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    sign[sign==0] = 1

                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    # 计算排序情况
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    # unequal_count = (sign != pred_sign).sum().item()
                    
                    rank_equal_num += equal_count
                    rank_all_num += len(sign)
                    accuracy = equal_count / len(sign)
                    
                    print("ValidBatch: ", epoch, ", Graph: ", graph_name, ", ValidLoss: ", loss.item(), ", Accuracy : ", accuracy)

        valid_accuracy = rank_equal_num / rank_all_num
        
        print("--- Epoch {} : Train {} Valid {} Acc {}---".format(epoch,
              train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))

        f.write("--- Epoch {} : Train {} Valid {} Acc {}---\n".format(epoch,
                train_loss/train_loss_cnt, valid_loss/valid_loss_cnt, valid_accuracy))
        f.flush()
        
        if valid_accuracy_best < valid_accuracy:
            torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_gat_best.pth")
            valid_accuracy_best = valid_accuracy
            
        torch.save(model.state_dict(), "/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/costmodel_gat.pth")
        print("save over.")



if __name__ == '__main__':
    # train_v2()
    train_onehot()
    # train_GAT()
