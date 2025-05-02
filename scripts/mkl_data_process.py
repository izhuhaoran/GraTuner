import os
import random
from collections import defaultdict


def combined_data():
    for algo in ["bfs", "cc", "pagerank", "sssp"]:
        # 文件路径
        file1 = f"{PROJECT_ROOT}/dataset/finetune/val/{algo}.csv"
        file2 = f"{PROJECT_ROOT}/dataset_YiTian/finetune/val/{algo}.csv"
        output_file = f"{PROJECT_ROOT}/dataset_MKL/finetune/val/{algo}_combined.csv"

        # 读取 file2 并构建字典
        file2_dict = {}
        with open(file2, "r") as f2:
            for line in f2:
                parts = line.strip().split(",")
                key = tuple(parts[:6])  # 前 6 列作为匹配键
                runtime = parts[6:]     # 第 7 列和第 8 列作为 runtime
                file2_dict[key] = runtime

        # 读取 file1 并合并数据
        merged_data = []
        with open(file1, "r") as f1:
            for line in f1:
                parts = line.strip().split(",")
                key = tuple(parts[:6])  # 前 6 列作为匹配键
                if key in file2_dict:
                    # 如果匹配成功，合并两行的 runtime 列
                    merged_row = parts + file2_dict[key]
                    merged_data.append(merged_row)
                    # print(merged_row)
                    # exit()

        # 保存合并后的文件
        with open(output_file, "w") as f_out:
            for row in merged_data:
                f_out.write(",".join(row) + "\n")

        print(f"合并后的文件已保存到: {output_file}")



def convert_txt_to_csv(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # 跳过注释行
            if line.startswith('//') or line.startswith('#'):
                continue

            # 分割并处理每个字段
            parts = line.strip().split(' ')
            # 第一个元素是图名称
            graph_name = parts[0]
            # 第二个元素到倒数第二个元素是配置
            config = ''.join(parts[1:-1])
            # 最后一个元素是时间
            time = parts[-1]

            # # 去除配置中的逗号
            # config = config.replace(',', '')
            

            # 组合成CSV格式
            csv_line = f"{graph_name},{config}{time}\n"
            f_out.write(csv_line)


def convert_txt_to_csv2(input_path, train_output_path, test_output_path):
    all_lines = []
    
    # 首先读取并处理所有行
    with open(input_path, 'r') as f_in:
        for line in f_in:
            if line.startswith('//') or line.startswith('#'):
                continue
                
            parts = line.strip().split(' ')
            graph_name = parts[0]
            config = ''.join(parts[1:-1])
            time = parts[-1]
            csv_line = f"{graph_name},{config}{time}\n"
            all_lines.append(csv_line)

    # 写入原始文件
    with open(train_output_path, 'w') as f_out:
        for line in all_lines:
            f_out.write(line)

    # 计算需要选择的行数
    select_count = int(len(all_lines) * 0.3)
    
    # 随机选择30%的行
    selected_lines = random.sample(all_lines, select_count)

    # # 写入原始文件
    # with open(train_output_path, 'w') as f_out:
    #     for line in all_lines:
    #         f_out.write(line)
            
    # 写入选择的行到新文件
    with open(test_output_path, 'w') as f_selected:
        for line in selected_lines:
            f_selected.write(line)

def convert_txt_to_csv3(input_path, train_output_path, test_output_path):
    # 用于按  graph_name 分组存储数据
    grouped_data = defaultdict(list)

    # 首先读取并处理所有行
    with open(input_path, 'r') as f_in:
        for line in f_in:
            if line.startswith('//') or line.startswith('#'):
                continue
                
            parts = line.strip().split(' ')
            graph_name = parts[0]
            config = ''.join(parts[1:-1])
            time = parts[-1]
            csv_line = f"{graph_name},{config}{time}\n"

            grouped_data[graph_name].append(csv_line)

    train_lines = []
    test_lines = []
    
    for graph_name, lines in grouped_data.items():
        select_count = int(len(lines) * 0.2)
        selected_lines = random.sample(lines, select_count)
        test_lines.extend(selected_lines)
        train_lines.extend(lines)

    with open(train_output_path, 'w') as f_out:
        f_out.writelines(train_lines)
    
    with open(test_output_path, 'w') as f_out:
        f_out.writelines(test_lines)


def split_csv_to_train_val(input_path, train_output_path, test_output_path, train_output_path_intel, test_output_path_intel, train_output_path_yitian, test_output_path_yitian):
    # 用于按  graph_name 分组存储数据
    grouped_data = defaultdict(list)

    # 首先读取并处理所有行
    with open(input_path, 'r') as f_in:
        for line in f_in:
            if line.startswith('//') or line.startswith('#'):
                continue
                
            parts = line.strip().split(',')
            graph_name = parts[0]
            config = ','.join(parts[1:-4])
            intel_time = ','.join(parts[-4:-2])
            yitian_time = ','.join(parts[-2:])
            # csv_line = f"{graph_name},{config}{time}\n"

            grouped_data['1'].append([graph_name, config, intel_time, yitian_time])

    train_lines = []
    test_lines = []
    
    for graph_name, lines in grouped_data.items():
        select_count = int(len(lines) * 0.2)
        selected_lines = random.sample(lines, select_count)
        test_lines.extend(selected_lines)
        # train except selected lines
        # train_lines.extend([line for line in lines if line not in selected_lines])
        # add all lines to train
        train_lines.extend(lines)

    with open(train_output_path, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join(line) + '\n')
    
    with open(train_output_path_intel, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join(line[:-1]) + '\n')
    
    with open(train_output_path_yitian, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join([line[0], line[1], line[3]]) + '\n')
    
    with open(test_output_path, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join(line) + '\n')
    
    with open(test_output_path_intel, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join(line[:-1]) + '\n')
    
    with open(test_output_path_yitian, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join([line[0], line[1], line[3]]) + '\n')

def split_csv_to_train_val_for_all_algo(input_path, train_output_path, test_output_path, train_output_path_intel, test_output_path_intel, train_output_path_yitian, test_output_path_yitian):
    # 用于按  graph_name 分组存储数据
    grouped_data = defaultdict(list)

    # 首先读取并处理所有行
    with open(input_path, 'r') as f_in:
        for line in f_in:
            if line.startswith('//') or line.startswith('#'):
                continue

            parts = line.strip().split(',')
            algo_name = parts[0]
            graph_name = parts[1]
            config = ','.join(parts[2:-4])
            intel_time = ','.join(parts[-4:-2])
            yitian_time = ','.join(parts[-2:])
            # csv_line = f"{graph_name},{config}{time}\n"

            grouped_data['1'].append([algo_name, graph_name, config, intel_time, yitian_time])

    train_lines = []
    test_lines = []
    
    for graph_name, lines in grouped_data.items():
        select_count = int(len(lines) * 0.2)
        selected_lines = random.sample(lines, select_count)
        test_lines.extend(selected_lines)
        # train except selected lines
        # train_lines.extend([line for line in lines if line not in selected_lines])
        # add all lines to train
        train_lines.extend(lines)

    with open(train_output_path, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join(line) + '\n')
    
    with open(train_output_path_intel, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join(line[:-1]) + '\n')
    
    with open(train_output_path_yitian, 'w') as f_out:
        for line in train_lines:
            f_out.write(','.join([line[0], line[1], line[2], line[4]]) + '\n')
    
    with open(test_output_path, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join(line) + '\n')
    
    with open(test_output_path_intel, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join(line[:-1]) + '\n')
    
    with open(test_output_path_yitian, 'w') as f_out:
        for line in test_lines:
            f_out.write(','.join([line[0], line[1], line[2], line[4]]) + '\n')


def extract_unique_schedules(input_file, output_file):
    unique_schedules = set()
    
    with open(input_file, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('//'):
                continue
                
            # 分割行
            parts = line.strip().split(',')
            if len(parts) < 4:  # 确保至少有中间元素
                continue
                
            # 提取中间元素 (去除第一列和最后两列)
            middle_elements = parts[1:-2]
            # 将中间元素组合成字符串
            schedule_str = ','.join(middle_elements)
            unique_schedules.add(schedule_str)
    
    # 写入去重后的配置
    with open(output_file, 'w') as f:
        for schedule in sorted(unique_schedules):
            f.write(f"{schedule}\n")

def gather_data(input_path_intel, input_path_yitian, output_path):

    gather_data = []
    with open(input_path_intel, 'r') as f_intel:
        lines = f_intel.readlines()
        gather_data.extend(lines)
    
    with open(input_path_yitian, 'r') as f_yitian:
        lines = f_yitian.readlines()
        gather_data.extend(lines)
    
    with open(output_path, 'w') as f_out:
        f_out.writelines(gather_data)

def all_algo_gather():
    output_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/all_algo_combined.csv'
    with open(output_path, 'w') as f_out:
        for algo in ["bfs", "pagerank", "sssp", 'cc']:
            input_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/{algo}_combined.csv'
            # 打开当前数据集文件
            with open(input_path, mode='r') as f_in:
                # 读取文件的每一行
                for line in f_in:
                    # 去掉行尾的换行符
                    line = line.strip()
                    # 在行首添加算法名称
                    new_line = f"{algo},{line}\n"
                    # 写入输出文件
                    f_out.write(new_line)

# all_algo_gather()
# input_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/all_algo_combined.csv'
# train_output_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo.csv'
# test_output_path = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo.csv'
# train_output_path_intel = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo_intel.csv'
# test_output_path_intel = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo_intel.csv'
# train_output_path_yitian = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo_yitian.csv'
# test_output_path_yitian = f'/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo_yitian.csv'
# split_csv_to_train_val_for_all_algo(input_path, train_output_path, test_output_path, train_output_path_intel, test_output_path_intel, train_output_path_yitian, test_output_path_yitian)

input_path_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo_intel.csv"
input_path_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo_yitian.csv"
output_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/all_algo_gather.csv"
gather_data(input_path_intel, input_path_yitian, output_path)

input_path_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo_intel.csv"
input_path_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo_yitian.csv"
output_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/all_algo_gather.csv"
gather_data(input_path_intel, input_path_yitian, output_path)

# for algo in ["bfs", "pagerank", "sssp", 'cc']:
#     # input_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/{algo}_new.gt_runtime.txt"
#     # convert_txt_to_csv3(input_path, output_path_train, output_path_val)

#     # input_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/{algo}_combined.csv"
#     # output_path_train = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}.csv"
#     # output_path_val = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}.csv"
#     # output_path_train_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}_intel.csv"
#     # output_path_val_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}_intel.csv"
#     # output_path_train_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}_yitian.csv"
#     # output_path_val_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}_yitian.csv"
#     # split_csv_to_train_val(input_path, output_path_train, output_path_val, output_path_train_intel, output_path_val_intel, output_path_train_yitian, output_path_val_yitian)

#     input_path_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}_intel.csv"
#     input_path_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}_yitian.csv"
#     output_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}_gather.csv"
#     gather_data(input_path_intel, input_path_yitian, output_path)
    
#     input_path_intel = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}_intel.csv"
#     input_path_yitian = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}_yitian.csv"
#     output_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/val/{algo}_gather.csv"
#     gather_data(input_path_intel, input_path_yitian, output_path)



# for algo in ["bfs", "pagerank", "sssp", 'cc']:
#     input_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/finetune/train/{algo}.csv"
#     output_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset_MKL/all_true_schedule_{algo}_GPU.csv"
#     extract_unique_schedules(input_path, output_path)

