import torch
import random
import pickle

from common.common import PROJECT_ROOT, graph_list, algo_op_dict, graph_num, algo_list
from cost_model.cost_model_Intel.model import AutoGraphModel
from cost_model.cost_model_Intel.data_loader import direction_map, parallelization_map, DenseVertexSet_map, SSGNum_map


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

    return schedules_origin, schedules_data

# 生成嵌入向量
@torch.no_grad()
def get_embedding(model: AutoGraphModel, sche: torch.Tensor) -> torch.Tensor:
    sche = sche.to(device)
    return model.embed_schedule(sche)


# 存储 KNN 图到文件
def save_knn_graph(knn_graph, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(knn_graph, f)

# 从文件读取 KNN 图
def load_knn_graph(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def knn_graph_create(model, schedules_data, n, m=10, random_init=False, all_graph_stack=False, out_base_name='knn_graph_init'):
    # # 生成所有嵌入向量
    # embeddings = torch.stack([get_embedding(schedule) for schedule in schedules])   # shape (n, 128)
    embeddings = get_embedding(model, torch.tensor(schedules_data).float())

    # 计算余弦距离
    # distances = 1 - F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # L2欧式距离
    distances = torch.cdist(embeddings, embeddings, p=2)

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
            save_knn_graph(knn_graph, f"{PROJECT_ROOT}/search/init_graph/{out_base_name}_all_graph_random.pkl")
        else:
            save_knn_graph(knn_graph, f"{PROJECT_ROOT}/search/init_graph/{out_base_name}_all_graph.pkl")
    else:
        if random_init:
            save_knn_graph(knn_graph, f"{PROJECT_ROOT}/search/init_graph/{out_base_name}_init_random.pkl")
        else:
            save_knn_graph(knn_graph, f"{PROJECT_ROOT}/search/init_graph/{out_base_name}.pkl")    

    return knn_graph


if __name__ == "__main__":

    model = AutoGraphModel()
    checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/finetune_result/costmodel_best.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    for algo in algo_list:
        schedules_origin, schedules_data = read_all_schedules(f"{PROJECT_ROOT}/search/all_true_schedule_{algo}.csv")
        n = len(schedules_data)
        
        # 生成 KNN 图
        knn_graph = knn_graph_create(model, schedules_data, n, random_init=False, all_graph_stack=False, out_base_name=f'knn_graph_init_{algo}')

