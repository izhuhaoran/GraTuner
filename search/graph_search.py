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

from cost_model.model import AutoGraphModel
from cost_model.data_loader import direction_map, parallelization_map, DenseVertexSet_map, SSGNum_map, graph_list


algo_list = ["pagerank", "bfs", "sssp", 'cc']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoGraphModel()
checkpoint = torch.load("/home/zhuhaoran/AutoGraph/AutoGraph/cost_model/train_result/costmodel_best_transformer.pth")
model.load_state_dict(checkpoint)

model = model.to(device)

runtimes_dict = {}

def read_runtime_dict():
    runtime_dict = {}
    for algo in ["bfs", "pagerank", "sssp", 'cc']:
        file_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo}.csv"
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


def read_all_schedules(path):
    schedules_origin = []
    schedules_data = []
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # processed = parts[1:-1]
            processed = parts[1:4] + [parts[5]]  # 跳过false列(parts[4])
            schedules_origin.append(processed)

            sche_tmp = [0, 0, 0, 0]
            sche_tmp[0] = direction_map[processed[0]]
            sche_tmp[1] = parallelization_map[processed[1]]
            sche_tmp[2] = DenseVertexSet_map[processed[2]]
            sche_tmp[3] = SSGNum_map[str(processed[3])]
            schedules_data.append(sche_tmp)
    
    # schedules_data = np.stack(schedules_data, axis=0)
    # schedules_data = schedules_data.astype(np.float32)
    # schedules_data = torch.from_numpy(schedules_data)

    return schedules_origin, schedules_data

# 生成嵌入向量
def get_embedding(sche: torch.Tensor) -> torch.Tensor:
    sche = sche.to(device)
    return model.embed_schedule(sche)

def cost_model_1(graph_embed, schedule, algo=None):
    return model.forward_after_query(graph_embed, schedule, algo)

def cost_model(graph_name, schedule, algo_name):
    key = algo_name + '_' + graph_name + '_' + '_'.join(schedule)
    return runtimes_dict[key]


# 存储 KNN 图到文件
def save_knn_graph(knn_graph, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(knn_graph, f)

# 从文件读取 KNN 图
def load_knn_graph(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


runtimes_dict = read_runtime_dict()
schedules_origin, schedules_data = read_all_schedules("/home/zhuhaoran/AutoGraph/AutoGraph/dataset/all_true_schedule.csv")
n = len(schedules_data)
graph_num = len(graph_list)


def knn_graph_create(m=20, random_init=False, all_graph_stack=False):
    # # 生成所有嵌入向量
    # embeddings = torch.stack([get_embedding(schedule) for schedule in schedules])   # shape (n, 128)
    embeddings = get_embedding(torch.tensor(schedules_data).float())

    # 计算余弦距离
    distances = 1 - F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # L2欧式距离
    # distances = torch.cdist(embeddings, embeddings, p=2)

    # 生成 KNN 图
    knn_graph = [[] for _ in range(n)]  # 使用二维列表表示 KNN 图
    # 每个元素为(邻居节点索引, 边权重列表), 边权重列表长度为 graph_num，表示不同图数据下的q值

    for i in range(n):
        # 获取与节点 i 距离最近的 m 个节点索引（不包括自己）
        nearest_indices = torch.argsort(distances[i])[:m + 1].tolist()  # 包括自己，所以取 m + 1 个
        for j in nearest_indices:
            if i != j:  # 排除自己
                # 初始化边权重为 0
                if all_graph_stack:
                    if not random_init:
                        knn_graph[i].append((j, [0.0] * graph_num))
                    else:
                        knn_graph[i].append((j, [random.uniform(-0.001, 0.001) for _ in range(graph_num)]))
                else:
                    if not random_init:
                        knn_graph[i].append((j, 0.0))
                    else:
                        knn_graph[i].append((j, random.uniform(-0.001, 0.001)))

    if all_graph_stack:
        if random_init:
            save_knn_graph(knn_graph, "/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init_all_graph_random.pkl")
        else:
            save_knn_graph(knn_graph, "/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init_all_graph.pkl")
    else:
        if random_init:
            save_knn_graph(knn_graph, "/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init_random.pkl")
        else:
            save_knn_graph(knn_graph, "/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init.pkl")    

    return knn_graph


def evaluate_episode(knn_graph, algo_name, schedules_origin):
    """评估一个episode的性能
    Returns:
        float: 平均reward [-1,1]
        float: 正向提升比例 [0,1]
    """
    total_reward = 0
    positive_count = 0
    count = 0
    
    for start_state in range(len(knn_graph)):
        current_state = start_state
        for step in range(10):
            neighbors = knn_graph[current_state]
            if not neighbors:
                break
                
            for graphid, graphname in enumerate(graph_list):
                next_state_idx = max(range(len(neighbors)), 
                                   key=lambda x: neighbors[x][1][graphid])
                next_state = neighbors[next_state_idx][0]
                
                current_cost = cost_model(graphname, schedules_origin[current_state], algo_name)
                next_cost = cost_model(graphname, schedules_origin[next_state], algo_name)
                reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1
                
                total_reward += reward
                if reward > 0:
                    positive_count += 1
                count += 1
                
            current_state = next_state
            
    avg_reward = total_reward / count if count > 0 else float('-inf')
    positive_ratio = positive_count / count if count > 0 else 0
    
    return avg_reward, positive_ratio


def evaluate_episode_by_search(knn_graph, k, graph_id, alog_name):
    graph_name = graph_list[graph_id]
    eval_random = RandomState(11)
    final_score = float('-inf')

    # for _ in range(10):
    # 初始化激活集 (随机选择 k 个初始点)
    active_set = eval_random.choice(range(n), 2 * k, replace=False)
    active_set = list(active_set)

    # 初始化结果队列，使用最小堆来维护得分最大的 k 个点
    result_queue = []

    # 初始化访问集合
    visited = set()

    # # 对每个节点的邻居按照边权重降序排序
    # for neighbors in knn_graph:
    #     neighbors.sort(key=lambda x: x[1][graph_id], reverse=True)

    # 对初始激活集中的点进行处理
    for node in active_set:
        visited.add(node)
        score = cost_model(graph_name, schedules_origin[node], alog_name)
        heapq.heappush(result_queue, (-score, node))
        if len(result_queue) > k:
            heapq.heappop(result_queue)

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(current_nodes):
        new_active_set = []
        for current_node in current_nodes:
            neighbors = knn_graph[current_node]
            if neighbors:  # 如果当前节点有邻居
                # 选择最大权重的那个邻居
                # next_node, weight = neighbors[0]
                next_node_idx = max(range(len(neighbors)), 
                    key=lambda x: neighbors[x][1])
                next_node = neighbors[next_node_idx][0]

                if next_node not in visited:
                    visited.add(next_node)
                    score = cost_model(graph_name, schedules_origin[next_node], alog_name)
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

    # 返回堆顶元素
    final_score = max(result_queue[0][0], final_score)

    return final_score


def q_learning_train_all_path(knn_graph, algo_name, graphid):

    # Configure logging
    logging.basicConfig(filename='/home/zhuhaoran/AutoGraph/AutoGraph/search/one_graph/q_learning_train.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f"KNN graph for {algo} {graph_list[graphid]} start training...")

    # Q-learning 参数
    num_episodes = 5000  # 训练轮数
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
                current_cost = cost_model(graphname, schedules_origin[current_state], algo_name)
                next_cost = cost_model(graphname, schedules_origin[next_state], algo_name)
                reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1
                
                # 更新Q值
                max_future_weight = max(weight for _, weight in knn_graph[next_state])
                new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
                knn_graph[current_state][next_state_idx] = (next_state, new_weight)
        
        # 评估和保存
        search_result = evaluate_episode_by_search(knn_graph, 10, graphid, algo_name)
        logging.info(f"Episode {episode}, graph: {graphname} Search result: {search_result:.4f}")
        
        if search_result > best_search_result:
            best_search_result = search_result
            best_episode = episode
            save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/one_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
            logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {best_episode}, evale search result: {search_result:.4f}")

        save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/one_graph/knn_graph_train_{algo_name}_{graphname}_latest.pkl")


def q_learning_train(knn_graph, algo_name, graphid):

    # Configure logging
    logging.basicConfig(filename='/home/zhuhaoran/AutoGraph/AutoGraph/search/one_graph/q_learning_train.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f"KNN graph for {algo} {graph_list[graphid]} start training...")

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
            current_cost = cost_model(graphname, schedules_origin[current_state], algo_name)
            next_cost = cost_model(graphname, schedules_origin[next_state], algo_name)
            reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1

            # 使用贝尔曼方程直接在knn_graph上更新q值<=>权重
            max_future_weight = max(weight for _, weight in knn_graph[next_state])
            new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
            
            # 更新边权重
            knn_graph[current_state][next_state_idx] = (next_state, new_weight)


            # logging.info(f"Episode {episode}, Epsilon: {epsilon:.4f}, Step {step}, Current state: {current_state}, Next state: {next_state}, Reward: {reward:.4f}")
            current_state = next_state

        search_result = evaluate_episode_by_search(knn_graph, 10, graphid, algo_name)
        logging.info(f"Episode {episode}, Search result: {search_result:.4f}")
        if search_result > best_search_result:
            best_search_result = search_result
            best_episode = episode
            save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/one_graph/knn_graph_best_{algo_name}_{graphname}.pkl")
            logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {best_episode}, evale search result: {search_result:.4f}")

        # if (episode + 1) % 100 == 0:
        #     save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_train_{algo_name}_{episode}.pkl")
        #     logging.info(f"KNN graph for {algo_name} saved at episode {episode}")
        
        epsilon = max(epsilon_end, epsilon_decay * epsilon)


def q_learning_train_all_graph(knn_graph, algo_name):
    # Configure logging
    logging.basicConfig(filename='/home/zhuhaoran/AutoGraph/AutoGraph/search/all_graph/q_learning_train.log', level=logging.INFO, format='%(asctime)s %(message)s')

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
                current_cost = cost_model(graphname, schedules_origin[current_state], algo_name)
                next_cost = cost_model(graphname, schedules_origin[next_state], algo_name)
                reward = (current_cost - next_cost) / current_cost if current_cost != 0 else -1

                # 使用贝尔曼方程直接在knn_graph上更新q值<=>权重
                max_future_weight = max(weight[graphid] for _, weight in knn_graph[next_state])
                new_weight = (1 - learning_rate) * old_weight + learning_rate * (reward + discount_factor * max_future_weight)
                
                # 更新边权重
                knn_graph[current_state][next_state_idx][1][graphid] = new_weight

                logging.info(f"Episode {episode}, Step {step}, Current state: {current_state}, Next state: {next_state}, Reward: {reward:.4f}, Epsilon: {epsilon:.4f}")
            current_state = next_state

        for graphid, graphname in enumerate(graph_list):
            search_result = evaluate_episode_by_search(knn_graph, 10, 0, algo_name)
            logging.info(f"Episode {episode}, Search result: {search_result:.4f}")
            if search_result > best_search_result[graphid]:
                best_search_result[graphid] = search_result
                best_episode[graphid] = episode
                save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_best_{algo_name}_{graphname}.pkl")
                logging.info(f"Best KNN graph for {algo_name} on {graphname} saved at episode {episode}, evale search result: {search_result:.4f}")

        # if (episode + 1) % 100 == 0:
        #     save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_train_{algo_name}_{episode}.pkl")
        #     logging.info(f"KNN graph for {algo_name} saved at episode {episode}")
        
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    save_knn_graph(knn_graph, f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_updated_{algo_name}.pkl")

# for r in [True, False]:
#     for g in [True, False]:
#         knn_graph = knn_graph_create(random_init=r, all_graph_stack=g)

# knn_graph = knn_graph_create(random_init=False, all_graph_stack=True)
# knn_graph = load_knn_graph(f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init.pkl")

# for algo in algo_list:
#     knn_graph = load_knn_graph(f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init.pkl")
#     logging.info(f"KNN graph for {algo} loaded, start training...")
#     q_learning_train_all_graph(knn_graph, algo)

for algo in algo_list:
    for graphid in range(graph_num):
        knn_graph = load_knn_graph(f"/home/zhuhaoran/AutoGraph/AutoGraph/search/knn_graph_init.pkl")
        q_learning_train(knn_graph, algo, graphid)




def search(knn_graph, k, graph_id, alog_name):
    graph_name = graph_list[graph_id]

    # 初始化激活集 (随机选择 k 个初始点)
    eval_random = RandomState(11)
    active_set = eval_random.choice(range(n), k, replace=False)
    active_set = list(active_set)
    
    # 初始化结果队列，使用最小堆来维护得分最大的 k 个点
    result_queue = []
    
    # 初始化访问集合
    visited = set()

    # 对每个节点的邻居按照边权重降序排序
    for neighbors in knn_graph:
        neighbors.sort(key=lambda x: x[1][graph_id], reverse=True)

    # 对初始激活集中的点进行处理
    for node in active_set:
        visited.add(node)
        score = cost_model(graph_name, schedules_origin[node], alog_name)
        heapq.heappush(result_queue, (score, node))
        if len(result_queue) > k:
            heapq.heappop(result_queue)

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(current_nodes):
        new_active_set = []
        for current_node in current_nodes:
            neighbors = knn_graph[current_node]
            if neighbors:  # 如果当前节点有邻居
                # 选择最大权重的那个邻居
                next_node, weight = neighbors[0]
                # next_node_idx = max(range(len(neighbors)), 
                #     key=lambda x: neighbors[x][1][graph_id])
                # next_node = neighbors[next_node_idx][0]

                if next_node not in visited:
                    visited.add(next_node)
                    score = cost_model(graph_name, schedules_origin[next_node], alog_name)
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

    # 获取得分最大的 k 个点
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    # result.sort(reverse=True)  # 按照得分从大到小排序
    return result


def anns_search(knn_graph, query_node, k):
    n = len(knn_graph)
    
    # 初始化优先队列，使用最大堆来维护最近的 k 个节点
    result_queue = []
    
    # 初始化访问集合
    visited = set()
    visited.add(query_node)

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(start_node):
        max_heap = [(-cost_model(start_node), start_node)]
        heapq.heapify(max_heap)
        
        while max_heap and len(result_queue) < k:
            current_score, current_node = heapq.heappop(max_heap)
            current_score = -current_score  # 还原原来的分数（因为最大堆）
            
            if current_node not in visited:
                visited.add(current_node)
                heapq.heappush(result_queue, (current_score, current_node))
                if len(result_queue) > k:
                    heapq.heappop(result_queue)
            
            # 遍历当前节点的邻居，基于边权重进行优先遍历
            for neighbor, weight in knn_graph[current_node]:
                if neighbor not in visited:
                    neighbor_score = cost_model(neighbor)
                    heapq.heappush(max_heap, (-neighbor_score, neighbor))

    # 开始最大权重优先遍历，从查询节点开始
    max_weight_priority_traversal(query_node)

    # 获取得分最高的 k 个节点
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    result.sort(reverse=True)  # 按照得分从大到小排序
    return result



# # 示例使用
# # 创建一个示例二维列表表示的 KNN 图
# knn_graph = [
#     [(1, 0.9), (2, 0.5), (3, 0.2)],
#     [(0, 0.9), (2, 0.8), (4, 0.3)],
#     [(0, 0.5), (1, 0.8), (3, 0.7)],
#     [(0, 0.2), (2, 0.7), (4, 0.6)],
#     [(1, 0.3), (3, 0.6)]
# ]

# # 搜索得分最大的 k 个点
# k = 3
# result = search(knn_graph, k)
# print("Top k nodes with highest scores:")
# for score, node in result:
#     print(f"Node: {node}, Score: {score:.4f}")




# def compare():
#     path1 = "/home/zhuhaoran/AutoGraph/AutoGraph/dataset/all_true_schedule.csv"
    
#     path2 = "/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/pagerank.csv"

#     schedules1 = set()
#     with open(path1, 'r') as f:
#         for line in f:
#             # 分割并去除首尾空白
#             parts = line.strip().split(',')
#             # 去除第一列(none)和最后一列(0.0)
#             processed = parts[1:-2]
#             # 将列表转换为元组以便能加入集合
#             schedules1.add(tuple(processed))

#     schedules2 = set()
#     with open(path2, 'r') as f:
#         for line in f:
#             # 分割并去除首尾空白
#             parts = line.strip().split(',')
#             # 去除第一列(none)和最后一列(0.0)
#             processed = parts[1:-2]
#             # 将列表转换为元组以便能加入集合
#             schedules2.add(tuple(processed))
    
#     # 判断schedules1中的元素是否都在schedules2中
#     for schedule in schedules1:
#         if schedule not in schedules2:
#             print(f"schedule {schedule} not in schedules2")
#             break
#     print("schedules1 is a subset of schedules2")

# # a_sch = read_all_schedules("/home/zhuhaoran/AutoGraph/AutoGraph/dataset/all_true_schedule.csv")
# # b_sch = read_all_schedules("/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/pagerank.csv")
# # print(f"len(a_sch): {len(a_sch)}, len(b_sch): {len(b_sch)}")

# compare()