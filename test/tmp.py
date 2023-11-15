import os
import pandas as pd

import numpy as np

import torch
from torch.utils.data import random_split

cwd = '/home/zhuhaoran/AutoGraph/AutoGraph/test'

def dataHandle(file_path):
    outLines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            data[0] = data[0] + ','
            new_line = ' '.join(data) + '\n'
            outLines.append(new_line)
    
    output_path = file_path[:-4] + '.csv'
    with open(output_path, 'w') as fw:
        fw.writelines(outLines)

def grouped_data(file_name):
    file_path = f'{cwd}/{file_name}'
    origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    # origin_data.to_csv(file_path, sep=',', header=False, index=False)
    
    # origin_data = pd.read_csv(file_path, comment='#', sep=', ', header=None)
    # origin_data.to_csv(file_path, sep=',', header=False, index=False)
    
    # 根据第一列的不同值进行分组
    origin_data = origin_data.groupby(0)
    
    # 遍历分组并保存为文件
    for name, group in origin_data:
        # 构造文件名
        filename = f"{cwd}/grouped/{name}_{file_name}"
        # 保存为CSV文件
        group.to_csv(filename, sep=',', header=False, index=False)

# for dataFile in os.listdir(cwd):
#     if dataFile.endswith('.csv'):
#         # input_file = os.path.join(cwd, dataFile)
#         grouped_data(dataFile)


def data_split(algo_name):
    file_path = file_path = f'{cwd}/{algo_name}.gt_output.csv'
    origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    origin_data = origin_data.values
    
    # 定义训练集和测试集的划分比例
    train_ratio = 0.7
    val_ratio = 0.3
    # test_ratio = 0.2

    train_size = int(train_ratio * len(origin_data))
    # val_size = int(val_ratio * len(Schedules_Dataset))
    val_size = len(origin_data) - train_size
    
    # 划分数据集
    np.random.shuffle(origin_data)  # 打乱数组顺序
    
    train_dataset = origin_data[:train_size]
    val_dataset = origin_data[train_size:]
    
    train_dataframe = pd.DataFrame(train_dataset)
    val_dataframe = pd.DataFrame(val_dataset)
    
    train_output_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo_name}.csv'
    val_output_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset/val/{algo_name}.csv'
    
    train_dataframe.to_csv(train_output_path, sep=',', header=False, index=False)
    val_dataframe.to_csv(val_output_path, sep=',', header=False, index=False)
    
    # np.savetxt(train_output_path, train_dataset, delimiter=',')
    # np.savetxt(val_output_path, val_dataset, delimiter=',')

# data_split('pagerank')


def group_test():
    file_path = '/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/pagerank.csv'
    origin_data = pd.read_csv(file_path, comment='#', sep=',', header=None)
    # origin_data.to_csv(file_path, sep=',', header=False, index=False)

    # origin_data = pd.read_csv(file_path, comment='#', sep=', ', header=None)
    # origin_data.to_csv(file_path, sep=',', header=False, index=False)

    # 根据第一列的不同值进行分组
    origin_data = origin_data.groupby(0)

    data = {}

    # 遍历分组并保存为文件
    for name, group in origin_data:
        # 构造文件名
        # print(group)
        a = group.values
        # print(a)
        data[name] = group.values
        print(type(group))

    print(data['dblp-cite'])

    for da in data['dblp-cite']:
        print(da)
        print(data['dblp-cite'])


def csv_compare():

    # 读取第一个 CSV 文件
    df1 = pd.read_csv('/home/zhuhaoran/AutoGraph/AutoGraph/dataset/all_true_schedule.csv', delimiter=',', header=None)
    # 读取第二个 CSV 文件
    df2 = pd.read_csv('/home/zhuhaoran/AutoGraph/AutoGraph/test/grouped/dblp-cite_pagerank.gt_output.csv', delimiter=',', header=None)

    # 提取第2列到第5列的数据
    columns_to_compare = [1, 2, 3, 4, 5]  # 第2列到第5列对应的索引是 1, 2, 3, 4

    # 比较两个数据框的指定列
    comparison = df1[columns_to_compare].equals(df2[columns_to_compare])

    if comparison:
        print("两个 CSV 文件的指定列内容完全一致")
    else:
        print("两个 CSV 文件的指定列内容不完全一致")


a = np.array([1,2,3])
b = np.array([1,2,3])
c = np.array([1,2,3])

d = np.concatenate((a,b,c))

print(d)