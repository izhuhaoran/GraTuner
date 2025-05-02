import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import random_split, DataLoader
import gc
import dgl

from data_loader import ScheduleDataset, GraphsDataset, ScheduleGraphDataset, ScheduleDataset_v2, ScheduleDataset_Onehot, ScheduleDataset_all_algo, ScheduleDataset_mkl
from model import AutoGraphModel_MKL, AutoGraphModel_algo, AutoGraphModel, AutoGraphModel_NoGraph
import logging
from common.common import PROJECT_ROOT

# 设置cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Configure logging
# logging.basicConfig(filename=f'{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_gcn_transformer_algo.log', level=logging.INFO, format='%(asctime)s %(message)s')

lr = 1e-4
batch_size = 32
epoches = 41


# graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#                'youtube-groupmemberships', 
#               ]

graph_list = ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA', ]

algo_op_dict = {
    'bfs': {'msg_create': 0, 'msg_reduce': 0, 'compute_mode': 0},   # first, =, 部分激活
    'pagerank': {'msg_create': 0, 'msg_reduce': 2, 'compute_mode': 1}, # first, +=, 全图计算
    'sssp': {'msg_create': 1, 'msg_reduce': 1, 'compute_mode': 0},  # +, min=, 部分激活
    'cc': {'msg_create': 0, 'msg_reduce': 1, 'compute_mode': 0},  # first, min=, 部分激活
}

def get_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 Handler
    if not logger.handlers:
        # 输出到文件
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # # 输出到终端
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        # ch.setFormatter(formatter)
        # logger.addHandler(ch)

    return logger


# def list_mle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps=1e-10, padded_value_indicator=-1, k=None):
#     """
#     Improved ListMLE Loss Implementation.
#     :param y_pred: 1d Tensor of shape [batch_size], predicted scores
#     :param y_true: 1d Tensor of shape [batch_size], ground truth labels
#     :param eps: Small value to ensure numerical stability
#     :param padded_value_indicator: Value in y_true indicating padding
#     :param k: Optional, top-k items to consider in loss computation
#     :return: Loss value (scalar)
#     """
#     if k is not None:
#         # Sample top-k indices if specified
#         indices = torch.randperm(y_pred.shape[0])[:k]
#         y_pred = y_pred[indices]
#         y_true = y_true[indices]

#     # Sort y_true in descending order and reorder y_pred accordingly
#     y_true_sorted, indices = y_true.sort(descending=True)
#     preds_sorted_by_true = y_pred[indices]

#     # Mask for padded values
#     mask = y_true_sorted == padded_value_indicator
#     preds_sorted_by_true[mask] = float('-inf')  # Exclude padded values

#     # Normalize for numerical stability
#     max_pred = preds_sorted_by_true.max()
#     preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred

#     # Compute cumulative sums and ListMLE loss
#     cumsums = preds_sorted_by_true_minus_max.exp().flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

#     observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
#     observation_loss[mask] = 0.0  # Ignore padded values

#     return observation_loss.sum().mean()


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


def eval_rank_quality(pred: torch.Tensor, true_y: torch.Tensor):
    """
    Evaluate the rank quality of the predicted scores.
    :param pred: 1d Tensor of shape [slate_length], predicted scores
    :param true_y: 1d Tensor of shape [slate_length], ground truth labels
    :return: Tuple of (equal_count, unequal_count)
    """
    # if len(pred) != len(true_y):
    #     return None
    
    # # 获取上三角矩阵的索引
    # iu = torch.triu_indices(len(pred), len(pred), 1)

    if pred.shape != true_y.shape:
        return None
    
    data_num = pred.shape[0]

    # 获取上三角矩阵的索引
    iu = torch.triu_indices(data_num, data_num, 1)

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


def create_dgl_graph(file_name) -> dgl.DGLGraph:
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

def create_and_cache_graphs():
    """预先创建所有图但保存在CPU上"""
    graph_dict = {}
    print("Creating DGL graphs...")
    for graph_name in graph_list:
        print(f"Creating DGL graph for {graph_name}")
        g = create_dgl_graph(graph_name)
        graph_dict[graph_name] = g
        print(f"Graph {graph_name} created")
    print("All DGL graphs created and cached on CPU")
    return graph_dict

def load_data(file_path, graph_col_id=0):
    data = {}
    # origin_data = schedule_preprocess(file_path)
    origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    
    # 根据第一列的不同值进行分组
    origin_data = origin_data.groupby(graph_col_id)
    
    # 遍历分组
    for name, group in origin_data:
        data[name] = group.values
    
    return data


def train_algo(algo_name='pagerank'):
    logger = get_logger(f"train_{algo_name}_logger", f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_{algo_name}.log")
    
    train_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/train/{algo_name}.csv'
    val_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/val/{algo_name}.csv'
    
    train_all_data = load_data(train_file_path)
    val_all_data = load_data(val_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel()
    
    checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result/costmodel_best_transformer.pth")
    model.load_state_dict(checkpoint)
    
    model = model.to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start Training")

    for epoch in range(epoches):
        print(f"Epoch {epoch} start")
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num_train = 0
        rank_unequal_num_train = 0
        rank_all_num_train = 0
        rank_equal_num_test = 0
        rank_unequal_num_test = 0
        rank_all_num_test = 0
        
        test_accuracy_best = 0.0
        test_best_epoch = 0
        
        for graph_name in graph_list:
            print(f"Train: Graph {graph_name} start")
            # torch.cuda.empty_cache()

            # Train_Schedules_Dataset = ScheduleDataset_Onehot(train_all_data[graph_name])
            # Val_Schedules_Dataset = ScheduleDataset_Onehot(val_all_data[graph_name])
            Train_Schedules_Dataset = ScheduleDataset_v2(train_all_data[graph_name])
            Val_Schedules_Dataset = ScheduleDataset_v2(val_all_data[graph_name])

            train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            
            g = create_dgl_graph(graph_name)
            g = g.to(device)

            rank_equal_num_graph_train = 0
            rank_unqual_num_graph_train = 0
            rank_all_num_graph_train = 0
            rank_equal_num_graph_test = 0
            rank_unequal_num_graph_test = 0
            rank_all_num_graph_test = 0

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
                if predict.shape[0] < 2:
                    continue

                # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

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
                # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                equal_count = (sign == pred_sign).sum().item()
                unequal_count = len(sign) - equal_count
                
                # accuracy = equal_count / len(sign)
                
                accuracy_train = equal_count / (equal_count + unequal_count)
                kendall_tau_train = (equal_count - unequal_count) / (equal_count + unequal_count)
                logger.info("--- Train batch --- Epoch {} : graph {} Train Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, graph_name, train_batchidx, loss.item(), accuracy_train, kendall_tau_train))
                
                rank_equal_num_graph_train += equal_count
                rank_unqual_num_graph_train += unequal_count
                rank_all_num_graph_train += equal_count + unequal_count

                rank_equal_num_train += equal_count
                rank_unequal_num_train += unequal_count
                rank_all_num_train += equal_count + unequal_count

            accuracy_graph_train = rank_equal_num_graph_train / rank_all_num_graph_train
            kendall_tau_graph_train = (rank_equal_num_graph_train - rank_unqual_num_graph_train) / rank_all_num_graph_train
            logger.info("--- Train graph --- Epoch {} : graph {} Train Acc {} Kendall {} ---".format(epoch, graph_name, accuracy_graph_train, kendall_tau_graph_train))

            # Test
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
                    
                    # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

                    # HingeRankingLoss
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                    
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    sign[sign==0] = 1

                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    # 计算排序情况
                    # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    unequal_count = len(sign) - equal_count
                
                    accuracy_test = equal_count / (equal_count + unequal_count)
                    kendall_tau_test = (equal_count - unequal_count) / (equal_count + unequal_count)
                    logger.info("--- Test batch --- Epoch {} : graph {} Test Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, graph_name, val_batchidx, loss.item(), accuracy_test, kendall_tau_test))
                    
                    rank_equal_num_graph_test += equal_count
                    rank_unequal_num_graph_test += unequal_count
                    rank_all_num_graph_test += equal_count + unequal_count
                    
                    rank_equal_num_test += equal_count
                    rank_unequal_num_test += unequal_count
                    rank_all_num_test += equal_count + unequal_count
            
            accuracy_graph_test = rank_equal_num_graph_test / rank_all_num_graph_test
            kendall_tau_graph_test = (rank_equal_num_graph_test - rank_unequal_num_graph_test) / rank_all_num_graph_test
            logger.info("--- Test graph --- Epoch {} : graph {} Test Acc {} Kendall {} ---".format(epoch, graph_name, accuracy_graph_test, kendall_tau_graph_test))

        train_accuracy = rank_equal_num_train / rank_all_num_train
        test_accuracy = rank_equal_num_test / rank_all_num_test
        
        train_kendall = (rank_equal_num_train - rank_unequal_num_train) / rank_all_num_train
        test_kendall = (rank_equal_num_test - rank_unequal_num_test) / rank_all_num_test

        logger.info("--- After Epoch {} : Train {} Valid {} Train Acc {} Test Acc {} Train Kendall {} Test Kendall {} ---".format(epoch, train_loss / train_loss_cnt, valid_loss / valid_loss_cnt, train_accuracy, test_accuracy, train_kendall, test_kendall))

        if test_accuracy > test_accuracy_best:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
            test_accuracy_best = test_accuracy
            test_best_epoch = epoch
            logger.info("--- Best Model Updated --- Epoch {} Test Acc {} ---".format(test_best_epoch, test_accuracy_best))
        
        torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_{epoch+300}.pth")



def train_all_algo_bak():
    logger = get_logger(f"train_all_algo_logger", f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_all_algo.log")
    train_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/train/all_algo.csv'
    val_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/val/all_algo.csv'
    
    train_all_data = load_data(train_file_path, graph_col_id=1)
    val_all_data = load_data(val_file_path, graph_col_id=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel_algo()
    
    # checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result/costmodel_best_transformer.pth")
    # model.load_state_dict(checkpoint)
    
    model = model.to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start Training")

    for epoch in range(epoches):
        print(f"Epoch {epoch} start")
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num_train = 0
        rank_unequal_num_train = 0
        rank_all_num_train = 0
        rank_equal_num_test = 0
        rank_unequal_num_test = 0
        rank_all_num_test = 0
        
        test_accuracy_best = 0.0
        test_best_epoch = 0
        
        for graph_name in graph_list:
            print(f"Train: Graph {graph_name} start")
            # torch.cuda.empty_cache()

            Train_Schedules_Dataset = ScheduleDataset_all_algo(train_all_data[graph_name])
            Val_Schedules_Dataset = ScheduleDataset_all_algo(val_all_data[graph_name])

            train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)
            
            g = create_dgl_graph(graph_name)
            g = g.to(device)

            rank_equal_num_graph_train = 0
            rank_unqual_num_graph_train = 0
            rank_all_num_graph_train = 0
            rank_equal_num_graph_test = 0
            rank_unequal_num_graph_test = 0
            rank_all_num_graph_test = 0

            # Train
            model.train()
            for train_batchidx, (algo_op, schedule, runtime) in enumerate(train_data_schedule):
                algo_op = algo_op.to(device)
                schedule = schedule.to(device)
                runtime = runtime.to(device)

                optimizer.zero_grad()

                graph_feature = model.embed_sparse_matrix(g)
                graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                # 前向传播
                # predict = model.forward_after_query(graph_feature, schedule)
                predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule)
                if predict.shape[0] < 2:
                    continue

                # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

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
                # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                equal_count = (sign == pred_sign).sum().item()
                unequal_count = len(sign) - equal_count
                
                # accuracy = equal_count / len(sign)
                
                accuracy_train = equal_count / (equal_count + unequal_count)
                kendall_tau_train = (equal_count - unequal_count) / (equal_count + unequal_count)
                logger.info("--- Train batch --- Epoch {} : graph {} Train Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, graph_name, train_batchidx, loss.item(), accuracy_train, kendall_tau_train))
                
                rank_equal_num_graph_train += equal_count
                rank_unqual_num_graph_train += unequal_count
                rank_all_num_graph_train += equal_count + unequal_count

                rank_equal_num_train += equal_count
                rank_unequal_num_train += unequal_count
                rank_all_num_train += equal_count + unequal_count

            accuracy_graph_train = rank_equal_num_graph_train / rank_all_num_graph_train
            kendall_tau_graph_train = (rank_equal_num_graph_train - rank_unqual_num_graph_train) / rank_all_num_graph_train
            logger.info("--- Train graph --- Epoch {} : graph {} Train Acc {} Kendall {} ---".format(epoch, graph_name, accuracy_graph_train, kendall_tau_graph_train))

            # Test
            model.eval()
            with torch.no_grad():
                for val_batchidx, (algo_op, schedule, runtime) in enumerate(val_data_schedule):

                    schedule = schedule.to(device)
                    runtime = runtime.to(device)

                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

                    # 前向传播
                    # predict = model.forward_after_query(graph_feature, schedule)
                    predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule)
                    
                    if predict.shape[0] < 2:
                        continue
                    
                    # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

                    # HingeRankingLoss
                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                    
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1-true2).sign()
                    sign[sign==0] = 1

                    loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    # 计算排序情况
                    # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    unequal_count = len(sign) - equal_count
                
                    accuracy_test = equal_count / (equal_count + unequal_count)
                    kendall_tau_test = (equal_count - unequal_count) / (equal_count + unequal_count)
                    logger.info("--- Test batch --- Epoch {} : graph {} Test Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, graph_name, val_batchidx, loss.item(), accuracy_test, kendall_tau_test))
                    
                    rank_equal_num_graph_test += equal_count
                    rank_unequal_num_graph_test += unequal_count
                    rank_all_num_graph_test += equal_count + unequal_count
                    
                    rank_equal_num_test += equal_count
                    rank_unequal_num_test += unequal_count
                    rank_all_num_test += equal_count + unequal_count
            
            accuracy_graph_test = rank_equal_num_graph_test / rank_all_num_graph_test
            kendall_tau_graph_test = (rank_equal_num_graph_test - rank_unequal_num_graph_test) / rank_all_num_graph_test
            logger.info("--- Test graph --- Epoch {} : graph {} Test Acc {} Kendall {} ---".format(epoch, graph_name, accuracy_graph_test, kendall_tau_graph_test))

        train_accuracy = rank_equal_num_train / rank_all_num_train
        test_accuracy = rank_equal_num_test / rank_all_num_test
        
        train_kendall = (rank_equal_num_train - rank_unequal_num_train) / rank_all_num_train
        test_kendall = (rank_equal_num_test - rank_unequal_num_test) / rank_all_num_test

        logger.info("--- After Epoch {} : Train {} Valid {} Train Acc {} Test Acc {} Train Kendall {} Test Kendall {} ---".format(epoch, train_loss / train_loss_cnt, valid_loss / valid_loss_cnt, train_accuracy, test_accuracy, train_kendall, test_kendall))

        if test_accuracy > test_accuracy_best:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
            test_accuracy_best = test_accuracy
            test_best_epoch = epoch
            logger.info("--- Best Model Updated --- Epoch {} Test Acc {} ---".format(test_best_epoch, test_accuracy_best))
        
        torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_{epoch+300}.pth")


def train_all_algo_v2():
    logger = get_logger("train_all_algo_v2_logger", f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_all_algo_v2.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel_algo()
    
    # checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result/costmodel_best_transformer.pth")
    # model.load_state_dict(checkpoint)
    
    model = model.to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start Training")

    graph_dict = create_and_cache_graphs()

    for epoch in range(epoches):
        print(f"Epoch {epoch} start")
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num_train = 0
        rank_unequal_num_train = 0
        rank_all_num_train = 0
        rank_equal_num_test = 0
        rank_unequal_num_test = 0
        rank_all_num_test = 0
        
        test_accuracy_best = 0.0
        test_best_epoch = 0
    
    
        for algo in ["pagerank", "sssp", "bfs", "cc",]:

            train_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/train/{algo}.csv'
            val_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/val/{algo}.csv'
            
            train_all_data = load_data(train_file_path, graph_col_id=0)
            val_all_data = load_data(val_file_path, graph_col_id=0)

            rank_equal_num_algo_train = 0
            rank_unequal_num_algo_train = 0
            rank_all_num_algo_train = 0
            rank_equal_num_algo_test = 0
            rank_unequal_num_algo_test = 0
            rank_all_num_algo_test = 0

            for graph_name in graph_list:
                print(f"Train: Epoch {epoch} Algo {algo} Graph {graph_name} start")
                torch.cuda.empty_cache()

                Train_Schedules_Dataset = ScheduleDataset_v2(train_all_data[graph_name])
                Val_Schedules_Dataset = ScheduleDataset_v2(val_all_data[graph_name])

                train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
                val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)

                # g_cpu = create_dgl_graph(graph_name)
                g_cpu = graph_dict[graph_name]
                g = g_cpu.to(device)

                rank_equal_num_graph_train = 0
                rank_unqual_num_graph_train = 0
                rank_all_num_graph_train = 0
                rank_equal_num_graph_test = 0
                rank_unequal_num_graph_test = 0
                rank_all_num_graph_test = 0

                # Train
                model.train()
                for train_batchidx, (schedule, runtime) in enumerate(train_data_schedule):
                    if len(schedule) <= 2:
                        continue
                    algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                    schedule = schedule.to(device)
                    runtime = runtime.to(device)

                    optimizer.zero_grad()

                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                    algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                    # 前向传播
                    # predict = model.forward_after_query(graph_feature, schedule)
                    predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule)
                    if predict.shape[0] <= 2:
                        continue

                    # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

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
                    # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                    pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                    equal_count = (sign == pred_sign).sum().item()
                    unequal_count = len(sign) - equal_count
                    
                    # accuracy = equal_count / len(sign)
                    
                    accuracy_train = equal_count / (equal_count + unequal_count)
                    kendall_tau_train = (equal_count - unequal_count) / (equal_count + unequal_count)
                    logger.info("--- Train batch --- Epoch {} : algo {} graph {} Train Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, algo, graph_name, train_batchidx, loss.item(), accuracy_train, kendall_tau_train))
                    
                    rank_equal_num_graph_train += equal_count
                    rank_unqual_num_graph_train += unequal_count
                    rank_all_num_graph_train += equal_count + unequal_count
                
                    rank_equal_num_algo_train += equal_count
                    rank_unequal_num_algo_train += unequal_count
                    rank_all_num_algo_train += equal_count + unequal_count

                    rank_equal_num_train += equal_count
                    rank_unequal_num_train += unequal_count
                    rank_all_num_train += equal_count + unequal_count

                accuracy_graph_train = rank_equal_num_graph_train / rank_all_num_graph_train
                kendall_tau_graph_train = (rank_equal_num_graph_train - rank_unqual_num_graph_train) / rank_all_num_graph_train
                logger.info("--- Train graph --- Epoch {} : algo {} graph {} Train Acc {} Kendall {} ---".format(epoch, algo, graph_name, accuracy_graph_train, kendall_tau_graph_train))

                # Test
                model.eval()
                with torch.no_grad():
                    for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):
                        if len(schedule) <= 2:
                            continue
                        algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                        schedule = schedule.to(device)
                        runtime = runtime.to(device)

                        graph_feature = model.embed_sparse_matrix(g)
                        graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                        algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                        # 前向传播
                        # predict = model.forward_after_query(graph_feature, schedule)
                        predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule)

                        if predict.shape[0] <= 2:
                            continue

                        # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

                        # HingeRankingLoss
                        iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                        
                        pred1, pred2 = predict[iu[0]], predict[iu[1]]
                        true1, true2 = runtime[iu[0]], runtime[iu[1]]
                        sign = (true1-true2).sign()
                        sign[sign==0] = 1

                        loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                        valid_loss += loss.item()
                        valid_loss_cnt += 1
                        
                        # 计算排序情况
                        # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                        pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                        equal_count = (sign == pred_sign).sum().item()
                        unequal_count = len(sign) - equal_count
                    
                        accuracy_test = equal_count / (equal_count + unequal_count)
                        kendall_tau_test = (equal_count - unequal_count) / (equal_count + unequal_count)
                        logger.info("--- Test batch --- Epoch {} : algo {} graph {} Test Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, algo, graph_name, val_batchidx, loss.item(), accuracy_test, kendall_tau_test))
                        
                        rank_equal_num_graph_test += equal_count
                        rank_unequal_num_graph_test += unequal_count
                        rank_all_num_graph_test += equal_count + unequal_count
                        
                        rank_equal_num_algo_test += equal_count
                        rank_unequal_num_algo_test += unequal_count
                        rank_all_num_algo_test += equal_count + unequal_count
                        
                        rank_equal_num_test += equal_count
                        rank_unequal_num_test += unequal_count
                        rank_all_num_test += equal_count + unequal_count
                
                accuracy_graph_test = rank_equal_num_graph_test / rank_all_num_graph_test
                kendall_tau_graph_test = (rank_equal_num_graph_test - rank_unequal_num_graph_test) / rank_all_num_graph_test
                logger.info("--- Test graph --- Epoch {} : algo {} graph {} Test Acc {} Kendall {} ---".format(epoch, algo, graph_name, accuracy_graph_test, kendall_tau_graph_test))
                
                # 每个图训练完后清理
                del g
                gc.collect()
                torch.cuda.empty_cache()
            
            accuracy_algo_train = rank_equal_num_algo_train / rank_all_num_algo_train
            kendall_tau_algo_train = (rank_equal_num_algo_train - rank_unequal_num_algo_train) / rank_all_num_algo_train
            logger.info("--- Train algo --- Epoch {} : algo {} Train Acc {} Kendall {} ---".format(epoch, algo, accuracy_algo_train, kendall_tau_algo_train))
            
            accuracy_algo_test = rank_equal_num_algo_test / rank_all_num_algo_test
            kendall_tau_algo_test = (rank_equal_num_algo_test - rank_unequal_num_algo_test) / rank_all_num_algo_test
            logger.info("--- Test algo --- Epoch {} : algo {} Test Acc {} Kendall {} ---".format(epoch, algo, accuracy_algo_test, kendall_tau_algo_test))
        
        train_accuracy = rank_equal_num_train / rank_all_num_train
        train_kendall = (rank_equal_num_train - rank_unequal_num_train) / rank_all_num_train

        test_accuracy = rank_equal_num_test / rank_all_num_test
        test_kendall = (rank_equal_num_test - rank_unequal_num_test) / rank_all_num_test

        logger.info("--- After Epoch {} : Train {} Valid {} Train Acc {} Test Acc {} Train Kendall {} Test Kendall {} ---".format(epoch, train_loss / train_loss_cnt, valid_loss / valid_loss_cnt, train_accuracy, test_accuracy, train_kendall, test_kendall))

        # logger.info("--- After Epoch {} : Train {} Train Acc {} Train Kendall {} ---".format(epoch, train_loss / train_loss_cnt, train_accuracy, train_kendall))
        if test_accuracy > test_accuracy_best:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
            test_accuracy_best = test_accuracy
            test_best_epoch = epoch
            logger.info("--- Best Model Updated --- Epoch {} Test Acc {} ---".format(test_best_epoch, test_accuracy_best))
        
        # if train_accuracy > test_accuracy_best:
        #     torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
        #     test_accuracy_best = train_accuracy
        #     test_best_epoch = epoch
        #     logger.info("--- Best Model Updated --- Epoch {} Test Acc {} ---".format(test_best_epoch, test_accuracy_best))

        torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_latest.pth")


def train_all_algo_mkl():
    logger = get_logger("train_all_algo_mkl_logger", f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_gcn_transformer_algo_mkl.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoGraphModel_MKL().to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start Training")

    graph_dict = create_and_cache_graphs()

    for epoch in range(epoches):
        epoch += 160
        print(f"Epoch {epoch} start")
        train_loss = 0.0
        train_loss_cnt = 0
        
        valid_loss = 0.0
        valid_loss_cnt = 0
        
        rank_equal_num_train = 0
        rank_unequal_num_train = 0
        rank_all_num_train = 0
        rank_equal_num_test = 0
        rank_unequal_num_test = 0
        rank_all_num_test = 0
        
        test_accuracy_best = 0.0
        test_best_epoch = 0

        # 遍历不同硬件
        for hardware in ["intel", "yitian"]:
            rank_all_num_hardware_train = 0
            rank_equal_num_hardware_train = 0
            rank_unequal_num_hardware_train = 0

            rank_all_num_hardware_test = 0
            rank_equal_num_hardware_test = 0
            rank_unequal_num_hardware_test = 0
    
            
            for algo in ["pagerank", "sssp", "bfs", "cc"]:

                train_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/train/{algo}_{hardware}.csv'
                val_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/val/{algo}_{hardware}.csv'
                
                train_all_data = load_data(train_file_path, graph_col_id=0)
                val_all_data = load_data(val_file_path, graph_col_id=0)

                rank_equal_num_algo_train = 0
                rank_unequal_num_algo_train = 0
                rank_all_num_algo_train = 0
                rank_equal_num_algo_test = 0
                rank_unequal_num_algo_test = 0
                rank_all_num_algo_test = 0

                for graph_name in graph_list:
                    print(f"Train: Epoch {epoch} Hardware {hardware} Algo {algo} Graph {graph_name} start")
                    torch.cuda.empty_cache()

                    Train_Schedules_Dataset = ScheduleDataset_v2(train_all_data[graph_name])
                    Val_Schedules_Dataset = ScheduleDataset_v2(val_all_data[graph_name])

                    train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
                    val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)

                    g_cpu = graph_dict[graph_name]
                    g = g_cpu.to(device)

                    rank_equal_num_graph_train = 0
                    rank_unqual_num_graph_train = 0
                    rank_all_num_graph_train = 0
                    rank_equal_num_graph_test = 0
                    rank_unequal_num_graph_test = 0
                    rank_all_num_graph_test = 0

                    # Train
                    model.train()
                    for train_batchidx, (schedule, runtime) in enumerate(train_data_schedule):
                        if len(schedule) <= 2:
                            continue
                        algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                        schedule = schedule.to(device)
                        runtime = runtime.to(device)

                        optimizer.zero_grad()

                        graph_feature = model.embed_sparse_matrix(g)
                        graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                        algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                        # 前向传播
                        predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule, hardware)
                        if predict.shape[0] <= 2:
                            continue

                        # HingeRankingLoss
                        iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                        pred1, pred2 = predict[iu[0]], predict[iu[1]]
                        true1, true2 = runtime[iu[0]], runtime[iu[1]]
                        sign = (true1 - true2).sign()
                        sign[sign == 0] = 1

                        loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)
                    
                        train_loss += loss.item()
                        train_loss_cnt += 1

                        loss.backward()
                        optimizer.step()
                        
                        # 计算排序情况
                        pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                        equal_count = (sign == pred_sign).sum().item()
                        unequal_count = len(sign) - equal_count
                        
                        accuracy_train = equal_count / (equal_count + unequal_count)
                        kendall_tau_train = (equal_count - unequal_count) / (equal_count + unequal_count)
                        logger.info(f"--- Train batch --- Epoch {epoch} : Hardware {hardware} Algo {algo} Graph {graph_name} Train Batch {train_batchidx} Loss {loss.item()} Accuracy {accuracy_train} Kendall {kendall_tau_train} ---")
                        
                        rank_equal_num_graph_train += equal_count
                        rank_unqual_num_graph_train += unequal_count
                        rank_all_num_graph_train += equal_count + unequal_count
                    
                        rank_equal_num_algo_train += equal_count
                        rank_unequal_num_algo_train += unequal_count
                        rank_all_num_algo_train += equal_count + unequal_count
                        
                        rank_equal_num_hardware_train += equal_count
                        rank_unequal_num_hardware_train += unequal_count
                        rank_all_num_hardware_train += equal_count + unequal_count

                        rank_equal_num_train += equal_count
                        rank_unequal_num_train += unequal_count
                        rank_all_num_train += equal_count + unequal_count

                    accuracy_graph_train = rank_equal_num_graph_train / rank_all_num_graph_train
                    kendall_tau_graph_train = (rank_equal_num_graph_train - rank_unqual_num_graph_train) / rank_all_num_graph_train
                    logger.info(f"--- Train graph --- Epoch {epoch} : Hardware {hardware} Algo {algo} Graph {graph_name} Train Acc {accuracy_graph_train} Kendall {kendall_tau_graph_train} ---")

                    # Test
                    model.eval()
                    with torch.no_grad():
                        for val_batchidx, (schedule, runtime) in enumerate(val_data_schedule):
                            if len(schedule) <= 2:
                                continue
                            algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                            schedule = schedule.to(device)
                            runtime = runtime.to(device)

                            graph_feature = model.embed_sparse_matrix(g)
                            graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                            algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                            # 前向传播
                            # predict = model.forward_after_query(graph_feature, schedule) 
                            predict = model.forward_with_graph_feature(graph_feature, algo_op, schedule, hardware)

                            if predict.shape[0] <= 2:
                                continue

                            # loss = list_mle_loss(predict.permute(1, 0), runtime.unsqueeze(0))

                            # HingeRankingLoss
                            iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)  # 使用上三角索引获得任意两个预测结果的组合
                            
                            pred1, pred2 = predict[iu[0]], predict[iu[1]]
                            true1, true2 = runtime[iu[0]], runtime[iu[1]]
                            sign = (true1-true2).sign()
                            sign[sign==0] = 1

                            loss = criterion(pred1.squeeze(), pred2.squeeze(), sign)

                            valid_loss += loss.item()
                            valid_loss_cnt += 1
                            
                            # 计算排序情况
                            # equal_count, unequal_count = eval_rank_quality(predict.squeeze(1), runtime)
                            pred_sign = torch.sign(pred1.squeeze() - pred2.squeeze())
                            equal_count = (sign == pred_sign).sum().item()
                            unequal_count = len(sign) - equal_count
                        
                            accuracy_test = equal_count / (equal_count + unequal_count)
                            kendall_tau_test = (equal_count - unequal_count) / (equal_count + unequal_count)
                            logger.info("--- Test batch --- Epoch {} : algo {} graph {} Test Batch {} Loss {} Accuracy {} Kendall {} ---".format(epoch, algo, graph_name, val_batchidx, loss.item(), accuracy_test, kendall_tau_test))
                            
                            rank_equal_num_graph_test += equal_count
                            rank_unequal_num_graph_test += unequal_count
                            rank_all_num_graph_test += equal_count + unequal_count
                            
                            rank_equal_num_algo_test += equal_count
                            rank_unequal_num_algo_test += unequal_count
                            rank_all_num_algo_test += equal_count + unequal_count
                            
                            rank_equal_num_hardware_test += equal_count
                            rank_unequal_num_hardware_test += unequal_count
                            rank_all_num_hardware_test += equal_count + unequal_count
                            
                            rank_equal_num_test += equal_count
                            rank_unequal_num_test += unequal_count
                            rank_all_num_test += equal_count + unequal_count
                    
                    accuracy_graph_test = rank_equal_num_graph_test / rank_all_num_graph_test
                    kendall_tau_graph_test = (rank_equal_num_graph_test - rank_unequal_num_graph_test) / rank_all_num_graph_test
                    logger.info("--- Test graph --- Epoch {} : algo {} graph {} Test Acc {} Kendall {} ---".format(epoch, algo, graph_name, accuracy_graph_test, kendall_tau_graph_test))
                    
                    # 每个图训练完后清理
                    del g
                    gc.collect()
                    torch.cuda.empty_cache()
                
                accuracy_algo_train = rank_equal_num_algo_train / rank_all_num_algo_train
                kendall_tau_algo_train = (rank_equal_num_algo_train - rank_unequal_num_algo_train) / rank_all_num_algo_train
                logger.info(f"--- Train algo --- Epoch {epoch} : Hardware {hardware} Algo {algo} Train Acc {accuracy_algo_train} Kendall {kendall_tau_algo_train} ---")

                accuracy_algo_test = rank_equal_num_algo_test / rank_all_num_algo_test
                kendall_tau_algo_test = (rank_equal_num_algo_test - rank_unequal_num_algo_test) / rank_all_num_algo_test
                logger.info("--- Test algo --- Epoch {} : algo {} Test Acc {} Kendall {} ---".format(epoch, algo, accuracy_algo_test, kendall_tau_algo_test))
        
            accuracy_hardware_train = rank_equal_num_hardware_train / rank_all_num_hardware_train
            kendall_tau_hardware_train = (rank_equal_num_hardware_train - rank_unequal_num_hardware_train) / rank_all_num_hardware_train
            logger.info(f"--- Train hardware --- Epoch {epoch} : Hardware {hardware} Train Acc {accuracy_hardware_train} Kendall {kendall_tau_hardware_train} ---")
            
            accuracy_hardware_test = rank_equal_num_hardware_test / rank_all_num_hardware_test
            kendall_tau_hardware_test = (rank_equal_num_hardware_test - rank_unequal_num_hardware_test) / rank_all_num_hardware_test
            logger.info("--- Test hardware --- Epoch {} : Hardware {} Test Acc {} Kendall {} ---".format(epoch, hardware, accuracy_hardware_test, kendall_tau_hardware_test))    
        
        train_accuracy = rank_equal_num_train / rank_all_num_train
        train_kendall = (rank_equal_num_train - rank_unequal_num_train) / rank_all_num_train

        test_accuracy = rank_equal_num_test / rank_all_num_test
        test_kendall = (rank_equal_num_test - rank_unequal_num_test) / rank_all_num_test

        logger.info("--- After Epoch {} : Train {} Valid {} Train Acc {} Test Acc {} Train Kendall {} Test Kendall {} ---".format(epoch, train_loss / train_loss_cnt, valid_loss / valid_loss_cnt, train_accuracy, test_accuracy, train_kendall, test_kendall))
        # logger.info("--- After Epoch {} : Train {} Train Acc {} Train Kendall {} ---".format(epoch, train_loss / train_loss_cnt, train_accuracy, train_kendall))
        
        if test_accuracy > test_accuracy_best:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
            test_accuracy_best = test_accuracy
            test_best_epoch = epoch
            logger.info("--- Best Model Updated --- Epoch {} Test Acc {} ---".format(test_best_epoch, test_accuracy_best))
    
        # if train_accuracy > test_accuracy_best:
        #     torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best.pth")
        #     test_accuracy_best = train_accuracy
        #     test_best_epoch = epoch
        #     logger.info(f"--- Best Model Updated --- Epoch {test_best_epoch} Test Acc {test_accuracy_best} ---")

        torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_latest.pth")

def train_all_algo_mkl_v2():
    logger = get_logger("train_all_algo_mkl_v2_logger", f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_gcn_transformer_algo_mkl_v2.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = AutoGraphModel_MKL().to(device)

    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start Training")

    graph_dict = create_and_cache_graphs()

    for epoch in range(epoches):
        print(f"Epoch {epoch} start")

        train_loss = {"intel": 0.0, "yitian": 0.0}
        train_loss_cnt = {"intel": 0, "yitian": 0}
        
        test_loss = {"intel": 0.0, "yitian": 0.0}
        test_loss_cnt = {"intel": 0, "yitian": 0}
        
        rank_equal_num_train = {"intel": 0, "yitian": 0}
        rank_unequal_num_train = {"intel": 0, "yitian": 0}
        rank_all_num_train = {"intel": 0, "yitian": 0}
        
        rank_equal_num_test = {"intel": 0, "yitian": 0}
        rank_unequal_num_test = {"intel": 0, "yitian": 0}
        rank_all_num_test = {"intel": 0, "yitian": 0}
        
        test_accuracy_best = {"intel": 0.0, "yitian": 0.0}
        test_best_epoch = {"intel": 0, "yitian": 0}

            
        for algo in ["pagerank", "sssp", "bfs", "cc"]:
            train_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/train/{algo}.csv'
            val_file_path = f'{PROJECT_ROOT}/datset/dataset_MKL/finetune/val/{algo}.csv'

            train_all_data = load_data(train_file_path, graph_col_id=0)
            val_all_data = load_data(val_file_path, graph_col_id=0)

            rank_equal_num_algo_train = {"intel": 0, "yitian": 0}
            rank_unequal_num_algo_train = {"intel": 0, "yitian": 0}
            rank_all_num_algo_train = {"intel": 0, "yitian": 0}
            rank_equal_num_algo_test = {"intel": 0, "yitian": 0}
            rank_unequal_num_algo_test = {"intel": 0, "yitian": 0}
            rank_all_num_algo_test = {"intel": 0, "yitian": 0}

            for graph_name in graph_list:
                print(f"Train: Epoch {epoch} Algo {algo} Graph {graph_name} start")
                torch.cuda.empty_cache()

                Train_Schedules_Dataset = ScheduleDataset_mkl(train_all_data[graph_name])
                Val_Schedules_Dataset = ScheduleDataset_mkl(val_all_data[graph_name])

                train_data_schedule = DataLoader(Train_Schedules_Dataset, batch_size=batch_size, shuffle=True)
                val_data_schedule = DataLoader(Val_Schedules_Dataset, batch_size=batch_size, shuffle=True)

                g_cpu = graph_dict[graph_name]
                g = g_cpu.to(device)

                rank_equal_num_graph_train = {"intel": 0, "yitian": 0}
                rank_unqual_num_graph_train = {"intel": 0, "yitian": 0}
                rank_all_num_graph_train = {"intel": 0, "yitian": 0}
                rank_equal_num_graph_test = {"intel": 0, "yitian": 0}
                rank_unequal_num_graph_test = {"intel": 0, "yitian": 0}
                rank_all_num_graph_test = {"intel": 0, "yitian": 0}

                # Train
                model.train()
                for train_batchidx, (schedule, runtime_intel, runtime_yitian) in enumerate(train_data_schedule):
                    if len(schedule) <= 2:
                        continue

                    optimizer.zero_grad()
                
                    algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                    schedule = schedule.to(device)
                    runtime_intel = runtime_intel.to(device)
                    runtime_yitian = runtime_yitian.to(device)

                    graph_feature = model.embed_sparse_matrix(g)
                    graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                    algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                    # 前向传播
                    predict_intel, predict_yitian = model.forward_with_graph_feature_v2(graph_feature, algo_op, schedule)
                    if predict_intel.shape[0] <= 2:
                        continue

                    # HingeRankingLoss
                    iu = torch.triu_indices(predict_intel.shape[0], predict_intel.shape[0], 1)
                    pred1_intel, pred2_intel = predict_intel[iu[0]], predict_intel[iu[1]]
                    true1_intel, true2_intel = runtime_intel[iu[0]], runtime_intel[iu[1]]
                    sign_intel = (true1_intel - true2_intel).sign()
                    sign_intel[sign_intel == 0] = 1

                    loss_intel = criterion(pred1_intel.squeeze(), pred2_intel.squeeze(), sign_intel)
                    
                    pred1_yitian, pred2_yitian = predict_yitian[iu[0]], predict_yitian[iu[1]]
                    true1_yitian, true2_yitian = runtime_yitian[iu[0]], runtime_yitian[iu[1]]
                    sign_yitian = (true1_yitian - true2_yitian).sign()
                    sign_yitian[sign_yitian == 0] = 1
                    
                    loss_yitian = criterion(pred1_yitian.squeeze(), pred2_yitian.squeeze(), sign_yitian)
                    
                    loss = loss_intel + loss_yitian
                
                    train_loss['intel'] += loss_intel.item()
                    train_loss['yitian'] += loss_yitian.item()
                    train_loss_cnt['intel'] += 1
                    train_loss_cnt['yitian'] += 1

                    loss.backward()
                    optimizer.step()
                    
                    # 计算排序情况
                    pre_sign_intel = torch.sign(pred1_intel.squeeze() - pred2_intel.squeeze())
                    equal_count_intel = (sign_intel == pre_sign_intel).sum().item()
                    unequal_count_intel = len(sign_intel) - equal_count_intel
                    
                    pre_sign_yitian = torch.sign(pred1_yitian.squeeze() - pred2_yitian.squeeze())
                    equal_count_yitian = (sign_yitian == pre_sign_yitian).sum().item()
                    unequal_count_yitian = len(sign_yitian) - equal_count_yitian
                    
                    accuracy_intel = equal_count_intel / (equal_count_intel + unequal_count_intel)
                    kendall_tau_intel = (equal_count_intel - unequal_count_intel) / (equal_count_intel + unequal_count_intel)

                    accuracy_yitian = equal_count_yitian / (equal_count_yitian + unequal_count_yitian)
                    kendall_tau_yitian = (equal_count_yitian - unequal_count_yitian) / (equal_count_yitian + unequal_count_yitian)
                    
                    logger.info(f"--- Train batch --- Epoch {epoch} : Algo {algo} Graph {graph_name} Train Batch {train_batchidx} Loss Intel {loss_intel.item()} Yitian {loss_yitian.item()} Accuracy Intel {accuracy_intel} Kendall Intel {kendall_tau_intel} Accuracy Yitian {accuracy_yitian} Kendall Yitian {kendall_tau_yitian} ---")
                    
                    rank_equal_num_graph_train['intel'] += equal_count_intel
                    rank_unqual_num_graph_train['intel'] += unequal_count_intel
                    rank_all_num_graph_train['intel'] += equal_count_intel + unequal_count_intel
                    
                    rank_equal_num_graph_train['yitian'] += equal_count_yitian
                    rank_unqual_num_graph_train['yitian'] += unequal_count_yitian
                    rank_all_num_graph_train['yitian'] += equal_count_yitian + unequal_count_yitian
                    
                    rank_equal_num_algo_train['intel'] += equal_count_intel
                    rank_unequal_num_algo_train['intel'] += unequal_count_intel
                    rank_all_num_algo_train['intel'] += equal_count_intel + unequal_count_intel
                    
                    rank_equal_num_algo_train['yitian'] += equal_count_yitian
                    rank_unequal_num_algo_train['yitian'] += unequal_count_yitian
                    rank_all_num_algo_train['yitian'] += equal_count_yitian + unequal_count_yitian
                    
                    rank_equal_num_train['intel'] += equal_count_intel
                    rank_unequal_num_train['intel'] += unequal_count_intel
                    rank_all_num_train['intel'] += equal_count_intel + unequal_count_intel
                    
                    rank_equal_num_train['yitian'] += equal_count_yitian
                    rank_unequal_num_train['yitian'] += unequal_count_yitian
                    rank_all_num_train['yitian'] += equal_count_yitian + unequal_count_yitian
                    
                    
                accuracy_graph_train_intel = rank_equal_num_graph_train['intel'] / rank_all_num_graph_train['intel']
                kendall_tau_graph_train_intel = (rank_equal_num_graph_train['intel'] - rank_unqual_num_graph_train['intel']) / rank_all_num_graph_train['intel']

                accuracy_graph_train_yitian = rank_equal_num_graph_train['yitian'] / rank_all_num_graph_train['yitian']
                kendall_tau_graph_train_yitian = (rank_equal_num_graph_train['yitian'] - rank_unqual_num_graph_train['yitian']) / rank_all_num_graph_train['yitian']
                logger.info(f"--- Train graph --- Epoch {epoch} : Algo {algo} Graph {graph_name} Train Acc Intel {accuracy_graph_train_intel} Kendall Intel {kendall_tau_graph_train_intel} Train Acc Yitian {accuracy_graph_train_yitian} Kendall Yitian {kendall_tau_graph_train_yitian} ---")
                
                # Test
                model.eval()
                with torch.no_grad():
                    for val_batchidx, (schedule, runtime_intel, runtime_yitian) in enumerate(val_data_schedule):
                        if len(schedule) <= 2:
                            continue
                        algo_op = torch.tensor([algo_op_dict[algo]['msg_create'], algo_op_dict[algo]['msg_reduce'], algo_op_dict[algo]['compute_mode']]).to(device)
                        schedule = schedule.to(device)
                        runtime_intel = runtime_intel.to(device)
                        runtime_yitian = runtime_yitian.to(device)

                        graph_feature = model.embed_sparse_matrix(g)
                        graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))
                        algo_op = algo_op.expand((schedule.shape[0], algo_op.shape[0]))

                        # 前向传播
                        predict_intel, predict_yitian = model.forward_with_graph_feature_v2(graph_feature, algo_op, schedule)
                        if predict_intel.shape[0] <= 2:
                            continue

                        # HingeRankingLoss
                        iu = torch.triu_indices(predict_intel.shape[0], predict_intel.shape[0], 1)
                        pred1_intel, pred2_intel = predict_intel[iu[0]], predict_intel[iu[1]]
                        true1_intel, true2_intel = runtime_intel[iu[0]], runtime_intel[iu[1]]
                        sign_intel = (true1_intel - true2_intel).sign()
                        sign_intel[sign_intel == 0] = 1

                        loss_intel = criterion(pred1_intel.squeeze(), pred2_intel.squeeze(), sign_intel)
                        
                        pred1_yitian, pred2_yitian = predict_yitian[iu[0]], predict_yitian[iu[1]]
                        true1_yitian, true2_yitian = runtime_yitian[iu[0]], runtime_yitian[iu[1]]
                        sign_yitian = (true1_yitian - true2_yitian).sign()
                        sign_yitian[sign_yitian == 0] = 1
                        
                        loss_yitian = criterion(pred1_yitian.squeeze(), pred2_yitian.squeeze(), sign_yitian)
                        
                        loss = loss_intel + loss_yitian

                        test_loss['intel'] += loss_intel.item()
                        test_loss['yitian'] += loss_yitian.item()
                        test_loss_cnt['intel'] += 1
                        test_loss_cnt['yitian'] += 1
                        
                        # 计算排序情况
                        pre_sign_intel = torch.sign(pred1_intel.squeeze() - pred2_intel.squeeze())
                        equal_count_intel = (sign_intel == pre_sign_intel).sum().item()
                        unequal_count_intel = len(sign_intel) - equal_count_intel
                        
                        pre_sign_yitian = torch.sign(pred1_yitian.squeeze() - pred2_yitian.squeeze())
                        equal_count_yitian = (sign_yitian == pre_sign_yitian).sum().item()
                        unequal_count_yitian = len(sign_yitian) - equal_count_yitian
                        
                        accuracy_intel = equal_count_intel / (equal_count_intel + unequal_count_intel)
                        kendall_tau_intel = (equal_count_intel - unequal_count_intel) / (equal_count_intel + unequal_count_intel)
                        
                        accuracy_yitian = equal_count_yitian / (equal_count_yitian + unequal_count_yitian)
                        kendall_tau_yitian = (equal_count_yitian - unequal_count_yitian) / (equal_count_yitian + unequal_count_yitian)
                        
                        logger.info(f"--- Test batch --- Epoch {epoch} : Algo {algo} Graph {graph_name} Test Batch {val_batchidx} Loss Intel {loss_intel.item()} Yitian {loss_yitian.item()} Accuracy Intel {accuracy_intel} Kendall Intel {kendall_tau_intel} Accuracy Yitian {accuracy_yitian} Kendall Yitian {kendall_tau_yitian} ---")
                        
                        rank_equal_num_graph_test['intel'] += equal_count_intel
                        rank_unequal_num_graph_test['intel'] += unequal_count_intel
                        
                        rank_equal_num_graph_test['yitian'] += equal_count_yitian
                        rank_unequal_num_graph_test['yitian'] += unequal_count_yitian
                        
                        rank_all_num_graph_test['intel'] += equal_count_intel + unequal_count_intel
                        rank_all_num_graph_test['yitian'] += equal_count_yitian + unequal_count_yitian
                        
                        rank_equal_num_algo_test['intel'] += equal_count_intel
                        rank_unequal_num_algo_test['intel'] += unequal_count_intel
                        
                        rank_equal_num_algo_test['yitian'] += equal_count_yitian
                        rank_unequal_num_algo_test['yitian'] += unequal_count_yitian
                        
                        rank_all_num_algo_test['intel'] += equal_count_intel + unequal_count_intel
                        rank_all_num_algo_test['yitian'] += equal_count_yitian + unequal_count_yitian
                        
                        rank_equal_num_test['intel'] += equal_count_intel
                        rank_unequal_num_test['intel'] += unequal_count_intel
                        
                        rank_equal_num_test['yitian'] += equal_count_yitian
                        rank_unequal_num_test['yitian'] += unequal_count_yitian
                        
                        rank_all_num_test['intel'] += equal_count_intel + unequal_count_intel
                        rank_all_num_test['yitian'] += equal_count_yitian + unequal_count_yitian
                        
                accuracy_graph_test_intel = rank_equal_num_graph_test['intel'] / rank_all_num_graph_test['intel']
                kendall_tau_graph_test_intel = (rank_equal_num_graph_test['intel'] - rank_unequal_num_graph_test['intel']) / rank_all_num_graph_test['intel']
                
                accuracy_graph_test_yitian = rank_equal_num_graph_test['yitian'] / rank_all_num_graph_test['yitian']
                kendall_tau_graph_test_yitian = (rank_equal_num_graph_test['yitian'] - rank_unequal_num_graph_test['yitian']) / rank_all_num_graph_test['yitian']
                
                logger.info(f"--- Test graph --- Epoch {epoch} : Algo {algo} Graph {graph_name} Test Acc Intel {accuracy_graph_test_intel} Kendall Intel {kendall_tau_graph_test_intel} Test Acc Yitian {accuracy_graph_test_yitian} Kendall Yitian {kendall_tau_graph_test_yitian} ---")
                        
                # 每个图训练完后清理
                del g
                gc.collect()
                torch.cuda.empty_cache()
            
            accuracy_algo_train_intel = rank_equal_num_algo_train['intel'] / rank_all_num_algo_train['intel']
            kendall_tau_algo_train_intel = (rank_equal_num_algo_train['intel'] - rank_unequal_num_algo_train['intel']) / rank_all_num_algo_train['intel']
            
            accuracy_algo_train_yitian = rank_equal_num_algo_train['yitian'] / rank_all_num_algo_train['yitian']
            kendall_tau_algo_train_yitian = (rank_equal_num_algo_train['yitian'] - rank_unequal_num_algo_train['yitian']) / rank_all_num_algo_train['yitian']
            
            logger.info(f"--- Train algo --- Epoch {epoch} : Algo {algo} Train Acc Intel {accuracy_algo_train_intel} Kendall Intel {kendall_tau_algo_train_intel} Train Acc Yitian {accuracy_algo_train_yitian} Kendall Yitian {kendall_tau_algo_train_yitian} ---")
            
            accuracy_algo_test_intel = rank_equal_num_algo_test['intel'] / rank_all_num_algo_test['intel']
            kendall_tau_algo_test_intel = (rank_equal_num_algo_test['intel'] - rank_unequal_num_algo_test['intel']) / rank_all_num_algo_test['intel']
            
            accuracy_algo_test_yitian = rank_equal_num_algo_test['yitian'] / rank_all_num_algo_test['yitian']
            kendall_tau_algo_test_yitian = (rank_equal_num_algo_test['yitian'] - rank_unequal_num_algo_test['yitian']) / rank_all_num_algo_test['yitian']
            
            logger.info(f"--- Test algo --- Epoch {epoch} : Algo {algo} Test Acc Intel {accuracy_algo_test_intel} Kendall Intel {kendall_tau_algo_test_intel} Test Acc Yitian {accuracy_algo_test_yitian} Kendall Yitian {kendall_tau_algo_test_yitian} ---")
            
        accuracy_hardware_train_intel = rank_equal_num_train['intel'] / rank_all_num_train['intel']
        kendall_tau_hardware_train_intel = (rank_equal_num_train['intel'] - rank_unequal_num_train['intel']) / rank_all_num_train['intel']
        
        accuracy_hardware_train_yitian = rank_equal_num_train['yitian'] / rank_all_num_train['yitian']
        kendall_tau_hardware_train_yitian = (rank_equal_num_train['yitian'] - rank_unequal_num_train['yitian']) / rank_all_num_train['yitian']
        
        logger.info(f"--- Train hardware --- Epoch {epoch} : Hardware Intel Train Acc {accuracy_hardware_train_intel} Kendall Intel {kendall_tau_hardware_train_intel} Train Acc Yitian {accuracy_hardware_train_yitian} Kendall Yitian {kendall_tau_hardware_train_yitian} ---")
        
        accuracy_hardware_test_intel = rank_equal_num_test['intel'] / rank_all_num_test['intel']
        kendall_tau_hardware_test_intel = (rank_equal_num_test['intel'] - rank_unequal_num_test['intel']) / rank_all_num_test['intel']
        
        accuracy_hardware_test_yitian = rank_equal_num_test['yitian'] / rank_all_num_test['yitian']
        kendall_tau_hardware_test_yitian = (rank_equal_num_test['yitian'] - rank_unequal_num_test['yitian']) / rank_all_num_test['yitian']
        
        logger.info(f"--- Test hardware --- Epoch {epoch} : Hardware Intel Test Acc {accuracy_hardware_test_intel} Kendall Intel {kendall_tau_hardware_test_intel} Test Acc Yitian {accuracy_hardware_test_yitian} Kendall Yitian {kendall_tau_hardware_test_yitian} ---")

        if accuracy_hardware_test_intel > test_accuracy_best['intel']:
        # if accuracy_hardware_train_intel > test_accuracy_best['intel']:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best_intel.pth")
            test_accuracy_best['intel'] = accuracy_hardware_test_intel
            # test_accuracy_best['intel'] = accuracy_hardware_train_intel
            test_best_epoch['intel'] = epoch
            logger.info(f"--- Best Model Updated --- Epoch {test_best_epoch['intel']} Test Acc {test_accuracy_best['intel']} ---")
        
        if accuracy_hardware_test_yitian > test_accuracy_best['yitian']:
        # if accuracy_hardware_train_yitian > test_accuracy_best['yitian']:
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_best_yitian.pth")
            test_accuracy_best['yitian'] = accuracy_hardware_test_yitian
            # test_accuracy_best['yitian'] = accuracy_hardware_train_yitian
            test_best_epoch['yitian'] = epoch
            logger.info(f"--- Best Model Updated --- Epoch {test_best_epoch['yitian']} Test Acc {test_accuracy_best['yitian']} ---")
        
        torch.save(model.state_dict(), f"{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/costmodel_latest_v2.pth")


if __name__ == '__main__':
    # train_onehot()
    # train_all_algo_mkl_v2()
    train_all_algo_mkl()
