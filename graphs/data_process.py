import os
import subprocess

# 数据集目录路径
dataset_directory = '/home/zhuhaoran/AutoGraph/graphs'

# youtube-u-growth: 3,223,589	9,375,374
# dblp_coauthor: 1,824,701	29,487,744
# soc-LiveJournal1: 4,846,609	68,475,391
# orkut-links: 3,072,441	117,185,083
# roadNet-CA: 1,965,206	2,766,607

def process_all():
    # 遍历数据集目录
    # for dataset_name in os.listdir(dataset_directory):
        for dataset_name in ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA']:
    # for dataset_name in ['moreno_blogs', 'petster-friendships-dog', 'subelj_cora']:
        dataset_path = os.path.join(dataset_directory, dataset_name)
        
        # 检查是否为目录
        if os.path.isdir(dataset_path):
            # 查找数据文本文件
            for filename in os.listdir(dataset_path):
                if filename.startswith('out.') and filename.endswith(dataset_name):
                    data_file_path = os.path.join(dataset_path, filename)
                    edges_file_path = os.path.join(dataset_path, dataset_name + '.el')
                    
                    # 处理数据文本文件
                    with open(data_file_path, 'r') as data_file, open(edges_file_path, 'w') as edges_file:
                        edge_count = 0
                        for line in data_file:
                            # 跳过注释行
                            if line.startswith('%') or line.startswith('#'):
                                continue
                            # 提取前两列形成边集数据
                            edge = line.strip().split(maxsplit=2)[:2]  # 修改这里的split()方法
                            edge_line = ' '.join(edge) + '\n'
                            edges_file.write(edge_line)
                            edge_count += 1
                    
                    print(f'Successfully processed: {data_file_path} ({edge_count} edges)')
                    
                    # 删除原始数据文本文件
                    os.remove(data_file_path)
                    
        else:
            print(f'No dataset found at: {dataset_path}')


def data_converter():

    # 遍历数据集目录
    for dataset_name in os.listdir(dataset_directory):
    # for dataset_name in ['moreno_blogs', 'petster-friendships-dog', 'subelj_cora']:
        dataset_path = os.path.join(dataset_directory, dataset_name)
        
        # 检查是否为目录
        if os.path.isdir(dataset_path):
            el_file_path = os.path.join(dataset_path, dataset_name + '.el')
            sg_file_path = os.path.join(dataset_path, dataset_name + '.sg')
            
            # 检查.el文件是否存在
            if os.path.isfile(el_file_path):
                # 构建converter命令
                command = ['/data/zhr_data/AutoGraph/gapbs/converter', '-f', el_file_path, '-b', sg_file_path]
                
                # 执行converter命令
                subprocess.run(command)
                
                print(f'Successfully converted: {el_file_path}')

process_all()
# data_converter()