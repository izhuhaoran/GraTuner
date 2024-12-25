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



def list_mle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps=1e-10, padded_value_indicator=-1, k=None):
    """
    Improved ListMLE Loss Implementation.
    :param y_pred: Tensor of shape [batch_size, slate_length], predicted scores
    :param y_true: Tensor of shape [batch_size, slate_length], ground truth labels
    :param eps: Small value to ensure numerical stability
    :param padded_value_indicator: Value in y_true indicating padding
    :param k: Optional, top-k items to consider in loss computation
    :return: Loss value (scalar)
    """
    if k is not None:
        # Sample top-k indices if specified
        indices = torch.randperm(y_pred.shape[1])[:k]
        y_pred = y_pred[:, indices]
        y_true = y_true[:, indices]

    # Sort y_true in descending order and reorder y_pred accordingly
    y_true_sorted, indices = y_true.sort(dim=-1, descending=True)
    preds_sorted_by_true = y_pred.gather(dim=1, index=indices)

    # Mask for padded values
    mask = y_true_sorted == padded_value_indicator
    preds_sorted_by_true[mask] = float('-inf')  # Exclude padded values

    # Normalize for numerical stability
    max_pred = preds_sorted_by_true.max(dim=1, keepdim=True).values
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred

    # Compute cumulative sums and ListMLE loss
    cumsums = preds_sorted_by_true_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    observation_loss[mask] = 0.0  # Ignore padded values

    return observation_loss.sum(dim=1).mean()


def eval_rank_quality(pred, true_y):
    if len(pred) != len(true_y):
        return None
    
    # 获取上三角矩阵的索引
    iu = torch.triu_indices(len(pred), len(pred), 1)

    # 根据索引获取预测值和真实值的配对
    pred1, pred2 = pred[iu[0]], pred[iu[1]]
    t1, t2 = true_y[iu[0]], true_y[iu[1]]

    # 计算预测值和真实值的符号差异
    y1 = torch.sign(pred1-pred2)
    y2 = torch.sign(t1-t2)

    # 计算相等元素的数目
    equal_count = (y1 == y2).sum().item()

    # 计算不等元素的数目
    unequal_count = (y1 != y2).sum().item()
    return equal_count, unequal_count


def kendall_tau(pred, true_y):
    equal_count, unequal_count = eval_rank_quality(pred, true_y)

    # n = len(pred)
    # total_pairs = n * (n - 1) / 2
    total_pairs = equal_count + unequal_count

    if total_pairs == 0:
        return 0  # Avoid division by zero

    tau = (equal_count - unequal_count) / total_pairs
    return tau


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

                graph_feature = model.embed_graph(g)
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

                    graph_feature = model.embed_graph(g)
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


# 自定义训练函数
def train_model(model, data_loader, num_epochs, optimizer, device):
    """
    Train the model with ListMLE loss.
    :param model: PyTorch model
    :param data_loader: DataLoader object for training data
    :param num_epochs: Number of epochs for training
    :param optimizer: Optimizer for training
    :param device: Device (CPU or GPU)
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in data_loader:
            configs, runtimes = batch
            configs, runtimes = configs.to(device), runtimes.to(device)

            # Forward pass: predict scores
            scores = model(configs).squeeze(-1)  # Shape [batch_size, n_configs]

            # Compute ListMLE loss
            loss = list_mle_loss(scores, runtimes)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")


def test_model(model, data_loader, device):
    """
    Evaluate the model with ListMLE loss on the test set.
    :param model: Trained PyTorch model
    :param data_loader: DataLoader object for test data
    :param device: Device (CPU or GPU)
    :return: Average loss on the test set
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in data_loader:
            configs, runtimes = batch
            configs, runtimes = configs.to(device), runtimes.to(device)

            # Forward pass: predict scores
            scores = model(configs).squeeze(-1)  # Shape [batch_size, n_configs]

            # Compute ListMLE loss
            loss = list_mle_loss(scores, runtimes)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


# 数据准备
def prepare_data(dataset, batch_size=32):
    """
    Prepare data for training.
    :param dataset: Input dataset as a 2D numpy array (configs and runtime)
    :param batch_size: Batch size for DataLoader
    :return: DataLoader for training
    """
    configs = dataset[:, :-1]  # 配置项
    runtimes = dataset[:, -1]  # 运行时间

    # 转换为 PyTorch 张量
    configs_tensor = torch.tensor(configs, dtype=torch.float32)
    runtimes_tensor = torch.tensor(runtimes, dtype=torch.float32)

    # 构建 DataLoader
    data = TensorDataset(configs_tensor, runtimes_tensor)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    # train_v2()
    train_onehot()
    # train_GAT()
