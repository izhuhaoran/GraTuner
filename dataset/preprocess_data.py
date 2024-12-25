import random

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
    
    # 计算需要选择的行数
    select_count = int(len(all_lines) * 0.3)
    
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

for algo in ["bfs", "pagerank", "sssp", 'cc']:
    input_path = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset/{algo}.gt_output.txt"
    output_path_train = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset/train/{algo}.csv"
    output_path_val = f"/home/zhuhaoran/AutoGraph/AutoGraph/dataset/val/{algo}.csv"
    convert_txt_to_csv2(input_path, output_path_train, output_path_val)