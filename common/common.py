
import os
import pandas as pd
import numpy as np
import logging
import dgl

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = '/home/zhuhaoran/AutoGraph/GraTuner'
COST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'cost_model')
DATASET_DIRECTORY = os.path.join(PROJECT_ROOT, 'dataset')
SEARCH_SPACE_PATH = os.path.join(PROJECT_ROOT, 'search')
LIBRARY_PATH = os.path.join(PROJECT_ROOT, 'lib')
# GRAPHS_PATH = os.path.join(PROJECT_ROOT, 'graphs')
GRAPHS_PATH = "/home/zhuhaoran/AutoGraph/graphs"
GRAPHIT_PATH = os.path.join(LIBRARY_PATH, 'graphit')
GRAPHIT_GPU_PATH = os.path.join(LIBRARY_PATH, 'graphit_gpu')


# graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#                'youtube-groupmemberships', 
#               ]
graph_list = ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA', ]
graph_num = len(graph_list)

algo_list = ["pagerank", "bfs", "sssp", 'cc']

algo_op_dict = {
    'bfs': {'msg_create': 0, 'msg_reduce': 0, 'compute_mode': 0},   # first, =, 部分激活
    'pagerank': {'msg_create': 0, 'msg_reduce': 2, 'compute_mode': 1}, # first, +=, 全图计算
    'sssp': {'msg_create': 1, 'msg_reduce': 1, 'compute_mode': 0},  # +, min=, 部分激活
    'cc': {'msg_create': 0, 'msg_reduce': 1, 'compute_mode': 0},  # first, min=, 部分激活
}

def create_dgl_graph(file_name):
    # file_name = file_name[0]
    file_path = f'{GRAPHS_PATH}/{file_name}/{file_name}.el'
    origin_graph = pd.read_csv(file_path, comment='#', sep=' ', header=None)

    # 提取源节点列（第一列）
    src_list = origin_graph[0].to_numpy()

    # 提取目标节点列（第二列）
    dst_list = origin_graph[1].to_numpy()

    # 生成dgl graph数据
    g = dgl.graph((src_list, dst_list))
    return g

def schedule_preprocess(file_path):
    origin_schedule_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    origin_schedule_data = origin_schedule_data.values
    return origin_schedule_data

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
