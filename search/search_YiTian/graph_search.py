import random
import heapq
import torch
import logging
import sys
from numpy.random import RandomState
import time

from common.common import PROJECT_ROOT
from common.common import create_dgl_graph, get_logger
from cost_model.cost_model_YiTian.model import AutoGraphModel

from graph_create import read_all_schedules, load_knn_graph, algo_list, graph_list, algo_op_dict, device
from graph_train import runtime_get, runtime_get_cost_model


def search(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model):
    graph_name = graph_list[graph_id]
    logger = get_logger(f"search_{algo_name}_{graph_name}", f"{PROJECT_ROOT}/search/search_YiTian/search_{algo_name}_{graph_name}.log")
    logger.info(f"Start search for algorithm {algo_name}, graph {graph_name}")
    
    # 初始化激活集 (随机选择 k 个初始点)
    eval_random = RandomState()
    active_set = eval_random.choice(range(n), k, replace=False)
    active_set = list(active_set)
    
    # 初始化结果队列，使用最小堆来维护得分最大的 k 个点
    result_queue = []
    
    # 初始化访问集合
    visited = set()

    # 对每个节点的邻居按照边权重降序排序, 可以预处理保存下来，不用每次都sort
    for neighbors in knn_graph:
        neighbors.sort(key=lambda x: x[1], reverse=True)

    t1 = time.perf_counter()
    # 对初始激活集中的点进行处理
    for node in active_set:
        visited.add(node)
        score = runtime_get(graph_name, schedules_origin[node], algo_name)
        # score = runtime_get_cost_model(model, graph_name, schedules_data[node], algo_name)
        heapq.heappush(result_queue, (-score, node))
        if len(result_queue) > k:
            heapq.heappop(result_queue)
    t2 = time.perf_counter()
    logger.info(f"Initialization active set time: {t2 - t1}")

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(current_nodes):
        new_active_set = []
        for current_node in current_nodes:
            neighbors = knn_graph[current_node]
            if neighbors:  # 如果当前节点有邻居
                # 选择最大权重的那个邻居
                next_node, _ = neighbors[0]

                if next_node not in visited:
                    visited.add(next_node)
                    score = runtime_get(graph_name, schedules_origin[next_node], algo_name)
                    # score = runtime_get_cost_model(model, graph_name, schedules_data[next_node], algo_name)
                    heapq.heappush(result_queue, (-score, next_node))
                    if len(result_queue) > k:
                        heapq.heappop(result_queue)
                    new_active_set.append(next_node)
        return new_active_set

    # 多轮遍历
    search_round = 0
    while active_set:
        # 复制一个result_queue
        result_queue_copy = result_queue.copy()
        cur_result = [heapq.heappop(result_queue_copy) for _ in range(len(result_queue_copy))]
        logger.info(f"Round {search_round}, active_set_num: {len(active_set)}, best_score: {cur_result}")
        active_set = max_weight_priority_traversal(active_set)
        search_round += 1
        if not active_set:
            break

    # 获取得分最大的 k 个点
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    result.sort(reverse=True)  # 按照得分从大到小排序
    return result


@torch.no_grad()
def search_v1(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model):
    graph_name = graph_list[graph_id]

    g = create_dgl_graph(graph_name).to(device)
    # 准备一次性的图特征
    graph_feature = model.embed_sparse_matrix(g)  # [1, out_dim=128]

    # 算法Embedding（单一样本同样可batch化）
    algo_op = torch.tensor([algo_op_dict[algo_name]['msg_create'],
                            algo_op_dict[algo_name]['msg_reduce'],
                            algo_op_dict[algo_name]['compute_mode']], device=device).unsqueeze(0)
    algo_feature = model.embed_algo(algo_op)  # [1, 128]

    # 初始化激活集
    eval_random = RandomState()
    active_set = eval_random.choice(range(n), k, replace=False).tolist()

    # 结果最小堆 + 访问集合
    result_queue = []
    visited = set()

    # 对每个节点的邻居按边权重降序排序
    for neighbors in knn_graph:
        neighbors.sort(key=lambda x: x[1], reverse=True)

    # 初始化阶段：对 active_set 中每个节点 individually forward
    for node in active_set:
        visited.add(node)
        schedules_batch = [schedules_data[node]]
        schedule_tensor = torch.tensor(schedules_batch, device=device)  # shape: [1, 128]

        predict = model.forward_after_query(algo_feature, graph_feature, schedule_tensor).squeeze(-1)
        # predict shape: [1]

        score = predict.item()
        heapq.heappush(result_queue, (-score, node))
        if len(result_queue) > k:
            heapq.heappop(result_queue)

    # 定义最大权重优先遍历函数：逐个 forward
    def max_weight_priority_traversal(cur_nodes):
        new_active = []
        for cur_node in cur_nodes:
            nbrs = knn_graph[cur_node]
            if nbrs:
                next_node, _ = nbrs[0]
                if next_node not in visited:
                    visited.add(next_node)
                    schedules_batch = [schedules_data[node]]
                    schedule_tensor = torch.tensor(schedules_batch, device=device)  # shape: [1, 128]

                    pred = model.forward_after_query(algo_feature, graph_feature, schedule_tensor)
                    score = pred.item()

                    heapq.heappush(result_queue, (-score, next_node))
                    if len(result_queue) > k:
                        heapq.heappop(result_queue)

                    new_active.append(next_node)
        return new_active

    # 多轮遍历
    while active_set:
        active_set = max_weight_priority_traversal(active_set)
        if not active_set:
            break

    # 提取得分最高的k个
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    result.sort(reverse=True)  # 按照得分从大到小排序
    return result


@torch.no_grad()
def search_v2(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model):
    graph_name = graph_list[graph_id]
    logger = get_logger(f"search_v2_{algo_name}_{graph_name}", f"{PROJECT_ROOT}/search/search_YiTian/search_v2_{algo_name}_{graph_name}.log")
    
    t0 = time.perf_counter()
    g = create_dgl_graph(graph_name).to(device)
    t1 = time.perf_counter()
    logger.info(f"Create graph time: {t1 - t0}")
    
    # 准备图特征
    graph_feature = model.embed_sparse_matrix(g)  # [1, out_dim=128]
    t2 = time.perf_counter()
    logger.info(f"Embed graph time: {t2 - t1}")

    # 算法Embedding（单一样本同样可batch化）
    algo_op = torch.tensor([algo_op_dict[algo_name]['msg_create'],
                            algo_op_dict[algo_name]['msg_reduce'],
                            algo_op_dict[algo_name]['compute_mode']], device=device).unsqueeze(0)
    algo_feature = model.embed_algo(algo_op)  # [1, 128]
    t3 = time.perf_counter()
    logger.info(f"Embed algo time: {t3 - t2}")

    # 初始化激活集
    eval_random = RandomState()
    active_set = eval_random.choice(range(n), k, replace=False).tolist()

    # 结果最小堆 + 访问集合
    result_queue = []
    visited = set()

    # 对每个节点的邻居按边权重降序排序
    for neighbors in knn_graph:
        neighbors.sort(key=lambda x: x[1], reverse=True)

    # -------------------------
    # 将初始active_set构造成一个batch通过模型预测
    # -------------------------
    schedules_batch = []
    for node in active_set:
        visited.add(node)
        schedules_batch.append(schedules_data[node])  # 收集schedule

    # 如果batch不为空，批量转tensor后一次forward
    if schedules_batch:
        schedule_tensor = torch.tensor(schedules_batch, device=device)
        # 令 batch_size = len(active_set)
        # graph_feature: [1, 128] -> expand到 [batch_size, 128]
        expanded_graph_feat = graph_feature.expand(schedule_tensor.size(0), graph_feature.size(1))
        expanded_algo_feat  = algo_feature.expand(schedule_tensor.size(0), algo_feature.size(1))

        predict = model.forward_after_query(expanded_algo_feat, expanded_graph_feat, schedule_tensor).squeeze(-1)
        # predict shape: [batch_size]

        # 将预测结果放入堆
        for node_id, score_pred in zip(active_set, predict):
            score_val = score_pred.item()
            heapq.heappush(result_queue, (-score_val, node_id))
            if len(result_queue) > k:
                heapq.heappop(result_queue)
    t4 = time.perf_counter()
    logger.info(f"Initialization active set time: {t4 - t3}")

    # 定义最大权重优先遍历
    def max_weight_priority_traversal(current_nodes):
        new_active = []
        neighbors_sche_batch = []

        for cur_node in current_nodes:
            nbrs = knn_graph[cur_node]
            if nbrs:
                next_node, _ = nbrs[0]
                if next_node not in visited:
                    visited.add(next_node)
                    new_active.append(next_node)
                    neighbors_sche_batch.append(schedules_data[next_node])
        # 批量调用模型计算，再放入result_queue
        if neighbors_sche_batch:
            schedule_tensor = torch.tensor(neighbors_sche_batch, device=device)
            expanded_graph_feat = graph_feature.expand(schedule_tensor.size(0), graph_feature.size(1))
            expanded_algo_feat  = algo_feature.expand(schedule_tensor.size(0), algo_feature.size(1))

            predict = model.forward_after_query(expanded_algo_feat, expanded_graph_feat, schedule_tensor).squeeze(-1)

            for nd, score_pred in zip(new_active, predict):
                val = score_pred.item()
                heapq.heappush(result_queue, (-val, nd))
                if len(result_queue) > k:
                    heapq.heappop(result_queue)

        return new_active

    # 多轮遍历
    while active_set:
        active_set = max_weight_priority_traversal(active_set)
        if not active_set:
            break
    
    t5 = time.perf_counter()
    logger.info(f"Traversal search time: {t5 - t4}")

    # 提取得分最高的k个
    # result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    # result.sort(reverse=True)  # 按照得分从大到小排序
    
    # 使用runtime最后验证结果
    final_result = []
    for _, node in result_queue:
        score = runtime_get(graph_name, schedules_origin[node], algo_name)
        final_result.append((score, node))
    final_result.sort()
    t6 = time.perf_counter()
    logger.info(f"Final result time: {t6 - t5}")
    logger.info(f"All time {t6 - t0}, Top k nodes for {algo_name} on {graph_name}: {final_result}")
    return final_result


@torch.no_grad()
def search_v2_print_iter(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model):
    graph_name = graph_list[graph_id]
    logger = get_logger(f"search_v2_print_iter_{algo_name}_{graph_name}", f"{PROJECT_ROOT}/search/search_YiTian/search_print_iter_{algo_name}_{graph_name}.log")
    
    t0 = time.perf_counter()
    g = create_dgl_graph(graph_name).to(device)
    t1 = time.perf_counter()
    logger.info(f"Create graph time: {t1 - t0}")
    
    # 准备图特征
    graph_feature = model.embed_sparse_matrix(g)  # [1, out_dim=128]
    t2 = time.perf_counter()
    logger.info(f"Embed graph time: {t2 - t1}")

    # 算法Embedding（单一样本同样可batch化）
    algo_op = torch.tensor([algo_op_dict[algo_name]['msg_create'],
                            algo_op_dict[algo_name]['msg_reduce'],
                            algo_op_dict[algo_name]['compute_mode']], device=device).unsqueeze(0)
    algo_feature = model.embed_algo(algo_op)  # [1, 128]
    t3 = time.perf_counter()
    logger.info(f"Embed algo time: {t3 - t2}")

    # 初始化激活集
    eval_random = RandomState()
    active_set = eval_random.choice(range(n), k, replace=False).tolist()

    # 结果最小堆 + 访问集合
    result_queue = []
    visited = set()

    # 对每个节点的邻居按边权重降序排序
    for neighbors in knn_graph:
        neighbors.sort(key=lambda x: x[1], reverse=True)

    # -------------------------
    # 将初始active_set构造成一个batch通过模型预测
    # -------------------------
    schedules_batch = []
    for node in active_set:
        visited.add(node)
        schedules_batch.append(schedules_data[node])  # 收集schedule

    # 如果batch不为空，批量转tensor后一次forward
    if schedules_batch:
        schedule_tensor = torch.tensor(schedules_batch, device=device)
        # 令 batch_size = len(active_set)
        # graph_feature: [1, 128] -> expand到 [batch_size, 128]
        expanded_graph_feat = graph_feature.expand(schedule_tensor.size(0), graph_feature.size(1))
        expanded_algo_feat  = algo_feature.expand(schedule_tensor.size(0), algo_feature.size(1))

        predict = model.forward_after_query(expanded_algo_feat, expanded_graph_feat, schedule_tensor).squeeze(-1)
        # predict shape: [batch_size]

        # 将预测结果放入堆
        for node_id, score_pred in zip(active_set, predict):
            score_val = score_pred.item()
            heapq.heappush(result_queue, (-score_val, node_id))
            if len(result_queue) > k:
                heapq.heappop(result_queue)

    t4 = time.perf_counter()
    logger.info(f"Initialization active set time: {t4 - t3}")

    # 定义最大权重优先遍历
    def max_weight_priority_traversal(current_nodes):
        new_active = []
        neighbors_sche_batch = []

        for cur_node in current_nodes:
            nbrs = knn_graph[cur_node]
            if nbrs:
                next_node, _ = nbrs[0]
                if next_node not in visited:
                    visited.add(next_node)
                    new_active.append(next_node)
                    neighbors_sche_batch.append(schedules_data[next_node])
        # 批量调用模型计算，再放入result_queue
        if neighbors_sche_batch:
            schedule_tensor = torch.tensor(neighbors_sche_batch, device=device)
            expanded_graph_feat = graph_feature.expand(schedule_tensor.size(0), graph_feature.size(1))
            expanded_algo_feat  = algo_feature.expand(schedule_tensor.size(0), algo_feature.size(1))

            predict = model.forward_after_query(expanded_algo_feat, expanded_graph_feat, schedule_tensor).squeeze(-1)

            for nd, score_pred in zip(new_active, predict):
                val = score_pred.item()
                heapq.heappush(result_queue, (-val, nd))
                if len(result_queue) > k:
                    heapq.heappop(result_queue)

        return new_active

    def get_real_runtime_for_result(result_queue):
        real_runtime = []
        for _, node in result_queue:
            score = runtime_get(graph_name, schedules_origin[node], algo_name)
            real_runtime.append((score, node))
        real_runtime.sort()
        return real_runtime

    iter = 0
    cur_runtime_result = get_real_runtime_for_result(result_queue)
    logger.info(f"iter {iter}, Top k runtime: {cur_runtime_result}, active_set: {active_set}")

    # 多轮遍历
    while active_set:
        t_iter_start = time.perf_counter()
        active_set = max_weight_priority_traversal(active_set)
        t_iter_end = time.perf_counter()
        iter += 1
        cur_runtime_result = get_real_runtime_for_result(result_queue)
        logger.info(f"iter {iter}, iter time: {t_iter_end - t_iter_start}, Top k runtime: {cur_runtime_result}, active_set: {active_set}")
        if not active_set:
            break
    
    t5 = time.perf_counter()
    logger.info(f"Traversal search time: {t5 - t4}")

    # 提取得分最高的k个
    # result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    # result.sort(reverse=True)  # 按照得分从大到小排序
    
    # 使用runtime最后验证结果
    final_result = get_real_runtime_for_result(result_queue)
    t6 = time.perf_counter()
    logger.info(f"Final result time: {t6 - t5}")
    logger.info(f"All time {t6 - t0}, Top k nodes for {algo_name} on {graph_name}: {final_result}")
    return final_result


def anns_search(knn_graph, query_node, k):
    n = len(knn_graph)
    
    # 初始化优先队列，使用最大堆来维护最近的 k 个节点
    result_queue = []
    
    # 初始化访问集合
    visited = set()
    visited.add(query_node)

    # 定义最大权重优先遍历函数
    def max_weight_priority_traversal(start_node):
        max_heap = [(-runtime_get(start_node), start_node)]
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
                    neighbor_score = runtime_get(neighbor)
                    heapq.heappush(max_heap, (-neighbor_score, neighbor))

    # 开始最大权重优先遍历，从查询节点开始
    max_weight_priority_traversal(query_node)

    # 获取得分最高的 k 个节点
    result = [heapq.heappop(result_queue) for _ in range(len(result_queue))]
    result.sort(reverse=True)  # 按照得分从大到小排序
    return result


if __name__ == "__main__":

    model = AutoGraphModel()
    checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_YiTian/finetune_result/costmodel_best.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    for _ in range(10):
        for algo_name in algo_list:
            schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/search_YiTian/all_true_schedule_{algo_name}_YiTian.csv")
            n = len(schedules_data)
            for graph_id in range(len(graph_list)):
                knn_graph_path = f"{PROJECT_ROOT}/search/search_YiTian/trained_graph/one_graph/knn_graph_best_{algo_name}_{graph_list[graph_id]}.pkl"
                knn_graph = load_knn_graph(knn_graph_path)
                k = 10
                # result = search_v2(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model)
                result = search_v2_print_iter(knn_graph, k, graph_id, algo_name, n, schedules_origin, schedules_data, model)
                print(f"Top k nodes with highest scores for {algo_name} on {graph_list[graph_id]}: {result}")
                # exit()


