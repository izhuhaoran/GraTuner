import random
from collections import defaultdict

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
    select_count = int(len(all_lines) * 0.2)
    
    # 随机选择30%的行
    selected_lines = random.sample(all_lines, select_count)

    # 写入原始文件
    with open(train_output_path, 'w') as f_out:
        for line in all_lines:
            f_out.write(line)
            
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


def cat_all_algo_data():
    all_lines = []
    for algo in ["bfs", "pagerank", "sssp", 'cc']:
        input_path = f"{PROJECT_ROOT}/dataset/train/{algo}.csv"
        with open(input_path, 'r') as f_in:
            for line in f_in:
                csv_line = f"{algo},{line}"
                all_lines.append(csv_line)

    with open("{PROJECT_ROOT}/dataset/train/all_algo.csv", 'w') as f_out:
        for line in all_lines:
            f_out.write(line)
            


for algo in ["bfs", "pagerank", "sssp", 'cc']:
    input_path = f"{PROJECT_ROOT}/dataset/finetune/{algo}.gt_output.txt"
    output_path_train = f"{PROJECT_ROOT}/dataset/finetune_new/train/{algo}.csv"
    output_path_val = f"{PROJECT_ROOT}/dataset/finetune_new/val/{algo}.csv"
    convert_txt_to_csv3(input_path, output_path_train, output_path_val)


# cat_all_algo_data()
