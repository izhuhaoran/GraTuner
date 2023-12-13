import os
import subprocess

# 数据集目录路径
dataset_directory = '/data/zhr_data/AutoGraph/graphit/autotune/graphs'


def process_all():
    # 遍历数据集目录
    for dataset_name in os.listdir(dataset_directory):
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
                        for line in data_file:
                            # 跳过注释行
                            if line.startswith('%') or line.startswith('#'):
                                continue
                            # 提取前两列形成边集数据
                            edge = line.strip().split(maxsplit=2)[:2]  # 修改这里的split()方法
                            edge_line = ' '.join(edge) + '\n'
                            edges_file.write(edge_line)
                    
                    print(f'Successfully processed: {data_file_path}')


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

# process_all()
data_converter()