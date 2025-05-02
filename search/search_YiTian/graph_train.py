import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import heapq
import pickle
import logging
import sys
from numpy.random import RandomState

from common.common import PROJECT_ROOT
from cost_model.cost_model_YiTian.model import AutoGraphModel
from graph_create import save_knn_graph, load_knn_graph, read_all_schedules
from graph_create import graph_num, graph_list, algo_list, algo_op_dict, device

output_dir = f'{PROJECT_ROOT}/search/search_YiTian/trained_graph'

runtimes_dict = {}

def read_runtime_dict():
    runtime_dict = {}
    for algo in ["bfs", "pagerank", "sssp", 'cc']:
        file_path = f"{PROJECT_ROOT}/dataset_YiTian/finetune/train/{algo}.csv"
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                
                # 提取数据
                graph_name = parts[0]
                configs = parts[1:4] + [parts[5]]  # 跳过false列(parts[4])
                runtime = float(parts[-1])
                
                # 构建key
                key = algo +'_' + graph_name + '_' + '_'.join(configs)
                
                # 存储到dict
                runtime_dict[key] = runtime 
    return runtime_dict


runtimes_dict = read_runtime_dict()


def runtime_get_cost_model(model, graph_feature, schedule, algo_name):
    algo_op = torch.tensor([algo_op_dict[algo_name]['msg_create'], 
                            algo_op_dict[algo_name]['msg_reduce'], 
                            algo_op_dict[algo_name]['compute_mode']]).to(device)

    return model.forward_with_graph_feature(graph_feature, algo_op, schedule)


def runtime_get(graph_name, schedule, algo_name):
    key = algo_name + '_' + graph_name + '_' + '_'.join(schedule)
    if key not in runtimes_dict:
        print(f"Key {key} not found in runtime dict")
        return 10000
    return runtimes_dict[key]


def evaluate_episode_by_search(knn_graph, k, graph_id, alog_name, schedules_origin, schedules_data=None, model=None):
    n = len(knn_graph)
    graph_name = graph_list[graph_id]
    eval_random = RandomState(11)
    final_score = float('-inf')

    # for _ in range(10):
    # 初始化激活集 (随机选择 k 个初始点)
    active_set = eval_random.choice(range(n), k, replace=False)
    active_set = list(active_set)

    # 初始化结果队列，使用最小堆来维护得分最大的 k 个点
    result_queue = []

    # 初始化访问集合
    visited = set()

    # # 对每个节点的邻居按照边权重降序排序
    # for neighbors in knn_graph:
    #     neighbors.sort(key=lambda x: x[1], reverse=True)

    # 对初始激活集中的点进行处理
    for node in active_set:
        visited.add(node)
        score = runtime_get(graph_name, schedules_origin[node], alog_name)
        # score = runtime_get_cost_model(model, graph_feature, schedules_data[node], alog_name)
        heapq.heappush(result_queue, (-score, node))
        if len(result_queue) > k:
            heapq.heappop(result_queue)

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(current_nodes):
        new_active_set = []
        for current_node in current_nodes:
            neighbors = knn_graph[current_node]
            if neighbors:  # 如果当前节点有邻居
                # 选择最大权重的那k个邻居
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for next_node_idx in [0]:
                    next_node = neighbors[next_node_idx][0]
                    if next_node not in visited:
                        visited.add(next_node)
                        score = runtime_get(graph_name, schedules_origin[next_node], alog_name)
                        # score = runtime_get_cost_model(model, graph_feature, schedules_data[next_node], alog_name)
                        heapq.heappush(result_queue, (-score, next_node))
                        if len(result_queue) > k:
                            heapq.heappop(result_queue)
                        new_active_set.append(next_node)
        return new_active_set

    # 多轮遍历
    while active_set:
        active_set = max_weight_priority_traversal(active_set)
        if not active_set:
            break

    # 返回堆中最大的k个值
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    result.sort(reverse=True)  # 按照得分从大到小排序
    final_score = result[0][0]
    return final_score


def q_learning_train(knn_graph, algo_name, graphid, model=None):
    # Configure logging
    logging.basicConfig(filename=f'{output_dir}/q_learning_train.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f"KNN graph for {algo_name} {graph_list[graphid]} start training...")

    schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/search_YiTian/all_true_schedule_{algo_name}_YiTian.csv")
    n = len(schedules_data)

    # Q-learning 参数
    num_episodes = 3000  # 训练轮数
    max_steps = 50  # 每轮最大步数
    learning_rate = 0.1 # 学习率
    discount_factor = 0.9   # 折扣因子

    epsilon_start = 0.5  # 初始探索率
    epsilon_end = 0.1   # 最终探索率
    epsilon_decay = 0.995  # 衰减率

    epsilon = epsilon_start # 当前探索率

    best_episode = 0
    best_search_result = float('-inf')
    
    graphname = graph_list[graphid]
    
    # Q-learning 训练过程
    for episode in range(num_episodes):
        # 随机选择一个起始点
        current_state = random.choice(range(n))

        for step in range(max_steps):
            # 获取当前状态的所有邻居
            neighbors = knn_graph[current_state]

            if not neighbors:
                break  # 如果没有邻居，跳出循环

            if random.random() > epsilon:
                # 以一定概率选择最优动作
                next_state_idx = max(range(len(neighbors)), key=lambda x: neighbors[x][1])
            else:
                # 以一定概率随机选择动作
                next_state_idx = random.randrange(len(neighbors))
    
            next_state = neighbors[next_state_idx][0]
            
            # 获取当前状态到邻居的边权重
            old_weight = neighbors[next_state_idx][1]
            current_cost = runtime_get(graphname, schedules_origin[current_state], algo_name)
            next_cost = runtime_get(graphname, schedules_origin[next_state], algo_name)
            # current_cost = runtime_get_cost_model(model, graph_feature, schedules_data[current_state], algo_name)
            # next_cost = runtime_get_cost_model(model, graph_feature, schedules_data[next_state], algo_name)
            reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1

            # 使用贝尔曼方程直接在knn_graph上更新q值<=>权重
            max_future_weight = max(weight for _, weight in knn_graph[next_state])
            new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
            
            # 更新边权重
            knn_graph[current_state][next_state_idx] = (next_state, new_weight)


            # logging.info(f"Episode {episode}, Epsilon: {epsilon}, Step {step}, Current state: {current_state}, Next state: {next_state}, Reward: {reward}")
            current_state = next_state

        search_result = evaluate_episode_by_search(knn_graph, 10, graphid, algo_name, schedules_origin, schedules_data, model)
        logging.info(f"Episode {episode}, Search result: {search_result}")
        if search_result > best_search_result:
            best_search_result = search_result
            best_episode = episode
            save_knn_graph(knn_graph, f"{output_dir}/one_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
            logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {best_episode}, evale search result: {search_result}")

        epsilon = max(epsilon_end, epsilon_decay * epsilon)


def q_learning_train_all_graph(knn_graph, algo_name, model=None):
    # Configure logging
    logging.basicConfig(filename=f'{output_dir}/q_learning_train_all_graph.log', level=logging.INFO, format='%(asctime)s %(message)s')

    schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/search_YiTian/all_true_schedule_{algo_name}_YiTian.csv")
    n = len(schedules_data)

    # Q-learning 参数
    num_episodes = 2000  # 训练轮数
    max_steps = 50  # 每轮最大步数
    learning_rate = 0.1 # 学习率
    discount_factor = 0.9   # 折扣因子
    
    epsilon_start = 0.5  # 初始探索率
    epsilon_end = 0.1   # 最终探索率
    epsilon_decay = 0.99  # 衰减率
    
    epsilon = epsilon_start # 当前探索率
    
    best_avg_reward = float('-inf')
    best_episode = [0] * graph_num
    best_search_result = [float('-inf')] * graph_num
    
    # Q-learning 训练过程
    for episode in range(num_episodes):
        # 随机选择一个起始点
        current_state = random.choice(range(n))

        for step in range(max_steps):
            # 获取当前状态的所有邻居
            neighbors = knn_graph[current_state]

            if not neighbors:
                break  # 如果没有邻居，跳出循环

            for graphid, graphname in enumerate(graph_list):

                if random.random() > epsilon:
                    # 以一定概率选择最优动作
                    next_state_idx = max(range(len(neighbors)), key=lambda x: neighbors[x][1][graphid])
                else:
                    # 以一定概率随机选择动作
                    next_state_idx = random.randrange(len(neighbors))
        
                next_state = neighbors[next_state_idx][0]
                
                # 获取当前状态到邻居的边权重
                old_weight = neighbors[next_state_idx][1][graphid]
                current_cost = runtime_get(graphname, schedules_origin[current_state], algo_name)
                next_cost = runtime_get(graphname, schedules_origin[next_state], algo_name)
                # current_cost = runtime_get_cost_model(model, graph_feature, schedules_data[current_state], algo_name)
                # next_cost = runtime_get_cost_model(model, graph_feature, schedules_data[next_state], algo_name)
                reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1

                # 使用贝尔曼方程直接在knn_graph上更新q值<=>权重
                max_future_weight = max(weight[graphid] for _, weight in knn_graph[next_state])
                new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
                
                # 更新边权重
                knn_graph[current_state][next_state_idx][1][graphid] = new_weight

                logging.info(f"Episode {episode}, Step {step}, Current state: {current_state}, Next state: {next_state}, Reward: {reward}, Epsilon: {epsilon}")
            current_state = next_state

        for graphid, graphname in enumerate(graph_list):
            search_result = evaluate_episode_by_search(knn_graph, 10, 0, algo_name, schedules_origin, schedules_data, model)
            logging.info(f"Episode {episode}, Search result: {search_result}")
            if search_result > best_search_result[graphid]:
                best_search_result[graphid] = search_result
                best_episode[graphid] = episode
                save_knn_graph(knn_graph, f"{output_dir}/all_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
                logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {episode}, evale search result: {search_result}")

        # if (episode + 1) % 100 == 0:
        #     save_knn_graph(knn_graph, f"{output_dir}/all_graph/knn_graph_train_{algo_name}_{episode}.pkl")
        #     logging.info(f"KNN graph for {algo_name} saved at episode {episode}")
        
        epsilon = max(epsilon_end, epsilon_decay * epsilon)


def q_learning_train_all_path(knn_graph, algo_name, graphid, model=None):

    # Configure logging
    logging.basicConfig(filename=f'{output_dir}/q_learning_train_all_path.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f"KNN graph for {algo} {graph_list[graphid]} start training...")

    schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/search_YiTian/all_true_schedule_{algo_name}_YiTian.csv")
    n = len(schedules_data)

    # Q-learning 参数
    num_episodes = 100  # 训练轮数
    learning_rate = 0.1 # 学习率
    discount_factor = 0.9   # 折扣因子

    best_episode = 0
    best_search_result = float('-inf')
    
    graphname = graph_list[graphid]
    
    # Q-learning 训练过程
    for episode in range(num_episodes):
        # 随机打乱状态顺序
        states = list(range(n))
        random.shuffle(states)
        for current_state in states:
            neighbors = knn_graph[current_state]
            if not neighbors:
                continue

            # 遍历该状态的所有可能动作
            for next_state_idx in range(len(neighbors)):
                next_state = neighbors[next_state_idx][0]
                old_weight = neighbors[next_state_idx][1]

                # 计算reward
                current_cost = runtime_get(graphname, schedules_origin[current_state], algo_name)
                next_cost = runtime_get(graphname, schedules_origin[next_state], algo_name)
                # current_cost = runtime_get_cost_model(model, graph_feature, schedules_data[current_state], algo_name)
                # next_cost = runtime_get_cost_model(model, graph_feature, schedules_data[next_state], algo_name)
                reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1
                
                # 更新Q值
                max_future_weight = max(weight for _, weight in knn_graph[next_state])
                new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
                knn_graph[current_state][next_state_idx] = (next_state, new_weight)
        
        # 评估和保存
        search_result = evaluate_episode_by_search(knn_graph, 10, graphid, algo_name, schedules_origin, schedules_data, model)
        logging.info(f"Episode {episode}, algo: {algo_name} graph: {graphname} Search result: {search_result}")

        if search_result >= best_search_result:
            best_search_result = search_result
            best_episode = episode
            save_knn_graph(knn_graph, f"{output_dir}/one_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
            logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {best_episode}, evale search result: {search_result}")

        save_knn_graph(knn_graph, f"{output_dir}/one_graph/knn_graph_train_{algo_name}_{graphname}_after{num_episodes}.pkl")


def q_learning_train_all_path_all_graph(knn_graph, algo_name, model=None):
    
    # Configure logging
    logging.basicConfig(filename=f'{output_dir}/q_learning_train_all_path_all_graph.log', level=logging.INFO, format='%(asctime)s %(message)s')

    schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/search_YiTian/all_true_schedule_{algo_name}_YiTian.csv")
    n = len(schedules_data)

    # Q-learning 参数
    num_episodes = 100  # 训练轮数
    learning_rate = 0.1 # 学习率
    discount_factor = 0.9   # 折扣因子

    best_episode = 0
    best_search_result = float('-inf')
    
    # Q-learning 训练过程
    for episode in range(num_episodes):
        # 随机打乱状态顺序
        states = list(range(n))
        random.shuffle(states)
        for current_state in states:
            neighbors = knn_graph[current_state]
            if not neighbors:
                continue

            for graphid, graphname in enumerate(graph_list):
                # 遍历该状态的所有可能动作
                for next_state_idx in range(len(neighbors)):
                    next_state = neighbors[next_state_idx][0]
                    old_weight = neighbors[next_state_idx][1][graphid]

                    # 计算reward
                    current_cost = runtime_get(graphname, schedules_origin[current_state], algo_name)
                    next_cost = runtime_get(graphname, schedules_origin[next_state], algo_name)
                    # current_cost = runtime_get_cost_model(model, graph_feature, schedules_data[current_state], algo_name)
                    # next_cost = runtime_get_cost_model(model, graph_feature, schedules_data[next_state], algo_name)
                    reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1
                    
                    # 更新Q值
                    max_future_weight = max(weight[graphid] for _, weight in knn_graph[next_state])
                    new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
                    knn_graph[current_state][next_state_idx][1][graphid] = new_weight
        
        # 评估和保存
        for graphid, graphname in enumerate(graph_list):
            search_result = evaluate_episode_by_search(knn_graph, 10, graphid, algo_name, schedules_origin, schedules_data, model)
            logging.info(f"Episode {episode}, graph: {graphname} Search result: {search_result}")

            if search_result > best_search_result[graphid]:
                best_search_result[graphid] = search_result
                best_episode[graphid] = episode
                save_knn_graph(knn_graph, f"{output_dir}/all_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
                logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {episode}, evale search result: {search_result}")
                
        save_knn_graph(knn_graph, f"{output_dir}/all_graph/knn_graph_train_{algo_name}_after{num_episodes}.pkl")


if __name__ == "__main__":

    model = AutoGraphModel()
    checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_YiTian/finetune_result/costmodel_best.pth")
    model.load_state_dict(checkpoint)
    model = model.to(device)

    for algo in algo_list:
        for graphid in range(graph_num):
            knn_graph = load_knn_graph(f"{PROJECT_ROOT}/search/search_YiTian/init_graph/knn_graph_init_{algo}.pkl")
            q_learning_train_all_path(knn_graph, algo, graphid, model=model)
            # q_learning_train(knn_graph, algo, graphid)

