import os
import csv
from collections import defaultdict

pass_compile_time_dict = {
    'CPU': 0.000423747 + 19.15371062,
    'GPU': 0.000416237 + 25.78255197,
    'YiTian': 0.000219786 + 10.83759035
}

fail_compile_time_dict = {
    'CPU': 0.000,
    'GPU': 0.000438496 + 17.21943241,
    'YiTian': 0.000
}

gratuner_csv_path_dict = {
    'CPU': '/home/zhuhaoran/AutoGraph/AutoGraph/search_results.csv',
    'GPU': '/home/zhuhaoran/AutoGraph/AutoGraph/search_results_GPU.csv',
    'YiTian': '/home/zhuhaoran/AutoGraph/AutoGraph/search_results_YiTian.csv'
}

opentuner_output_dir_path_dict = {
    'CPU': '/home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/GraTuner_tests_exectime',
    'GPU': '/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/GraTuner_tests_exectime',
    'YiTian': '/home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/GraTuner_tests_exectime_YiTian'
}



def read_gratuner_result(csv_file_path):
    # 初始化一个字典来存储结果
    result_dict = {}

    # 打开CSV文件并读取内容
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)  # 使用DictReader读取CSV文件
        for row in reader:
            # 构造字典的key：algo_name,graph_name
            key = f"{row['algo_name']},{row['graph_name']}"
            # 获取当前行的best_result_runtime
            current_runtime = float(row['best_result_runtime'])
            # 更新字典中的最小值
            if key not in result_dict:
                result_dict[key] = current_runtime
            elif current_runtime < result_dict[key]:
                result_dict[key] = current_runtime

    # # 打印结果
    # for key, value in result_dict.items():
    #     print(f"{key}: {value}")
    
    return result_dict

def process_time_log(device):
    output_file_path_dict = {
        'CPU': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results.csv',
        'GPU': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results_GPU.csv',
        'YiTian': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results_YiTian.csv'
    }
    base_dir = opentuner_output_dir_path_dict[device]
    out_file = output_file_path_dict[device]

    pass_compile_time = pass_compile_time_dict[device]
    fail_compile_time = fail_compile_time_dict[device]
    
    gratuner_result = read_gratuner_result(gratuner_csv_path_dict[device])

    headers = ['algo_name', 'graph_name', 'round', 'tune_time', 'after_1min_exec_time', 'after_5min_exec_time', 'after_10min_exec_time']
    results = []

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if os.path.isdir(dirpath) and dirname.startswith("output_autotune_"):
            parts = dirname.split("_")
            algo_name = parts[2]
            graph_name = "_".join(parts[3:-1])
            round = parts[-1]
            
            gratuner_key = f"{algo_name},{graph_name}"
            gratuner_result_runtime = gratuner_result[gratuner_key]

            input_file = os.path.join(dirpath, "time_log_iter.txt")
            output_file = os.path.join(dirpath, "time_log_iter_with_alltime.txt")

            accumulated_time = 0.0
            min_time = 1000000000.0
            
            group_data = {
                'algo_name': algo_name,
                'graph_name': graph_name,
                'round': round,
            }
            
            with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 解析当前行
                    parts = line.split(',')
                    status = parts[0].strip()
                    
                    # 提取runtime和exectime 倒数第二个和最后一个
                    runtime = 0.0
                    exectime = 0.0
                    runtime = float(parts[-2].split(':')[1])
                    exectime = float(parts[-1].split(':')[1])

                    # 计算累积时间
                    if status == 'passed':
                        accumulated_time += pass_compile_time + runtime
                        # 记录最小时间
                        min_time = min(min_time, exectime)
                    elif status == 'unpass':
                        accumulated_time += runtime + fail_compile_time
                    else:
                        accumulated_time += runtime
                    
                    # 记录tune_time
                    if min_time == gratuner_result_runtime and 'tune_time' not in group_data:
                        group_data['tune_time'] = accumulated_time

                    # 记录after_1min_exec_time, after_5min_exec_time, after_10min_exec_time
                    if accumulated_time >= 60.0 and 'after_1min_exec_time' not in group_data:
                        group_data['after_1min_exec_time'] = min_time
                    
                    if accumulated_time >= 300.0 and 'after_5min_exec_time' not in group_data:
                        group_data['after_5min_exec_time'] = min_time
                    
                    if accumulated_time >= 600.0 and 'after_10min_exec_time' not in group_data:
                        group_data['after_10min_exec_time'] = min_time
                
                    # 写入新行
                    new_line = f"{line}, all_time: {accumulated_time}, min_time: {min_time}\n"
                    fout.write(new_line)
                
                results.append(group_data)
    
    # 写入CSV
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def process_time_log_v2(device):
    output_file_path_dict = {
        'CPU': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results_v2.csv',
        'GPU': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results_GPU_v2.csv',
        'YiTian': '/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_results_YiTian_v2.csv'
    }
    base_dir = opentuner_output_dir_path_dict[device]
    out_file = output_file_path_dict[device]

    pass_compile_time = pass_compile_time_dict[device]
    fail_compile_time = fail_compile_time_dict[device]
    
    gratuner_result = read_gratuner_result(gratuner_csv_path_dict[device])


    # headers = ['algo_name', 'graph_name', 'tune_time', 'after_1min_exec_time', 'after_5min_exec_time', 'after_10min_exec_time']
    # 'tune_time', 'after_1min_exec_time', 'after_5min_exec_time', 'after_10min_exec_time'分别记录平均值 中位数 最大值 最小值
    headers = ['algo_name', 'graph_name', 'tune_time_mean', 'tune_time_median', 'tune_time_max', 'tune_time_min',
               'after_1min_exec_time_mean', 'after_1min_exec_time_median', 'after_1min_exec_time_max', 'after_1min_exec_time_min',
               'after_2min_exec_time_mean', 'after_2min_exec_time_median', 'after_2min_exec_time_max', 'after_2min_exec_time_min',
               'after_5min_exec_time_mean', 'after_5min_exec_time_median', 'after_5min_exec_time_max', 'after_5min_exec_time_min',
               'after_10min_exec_time_mean', 'after_10min_exec_time_median', 'after_10min_exec_time_max', 'after_10min_exec_time_min']
    
    # 用于存储分组数据
    grouped_data_all = defaultdict(lambda: {
        'tune_time': [],
        'after_1min_exec_time': [],
        'after_2min_exec_time': [],
        'after_5min_exec_time': [],
        'after_10min_exec_time': [],
    })

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if os.path.isdir(dirpath) and dirname.startswith("output_autotune_"):
            parts = dirname.split("_")
            algo_name = parts[2]
            graph_name = "_".join(parts[3:-1])
            tuning_round = parts[-1]
            
            gratuner_key = f"{algo_name},{graph_name}"
            gratuner_result_runtime = gratuner_result[gratuner_key]

            input_file = os.path.join(dirpath, "time_log_iter.txt")
            output_file = os.path.join(dirpath, "time_log_iter_with_alltime.txt")

            accumulated_time = 0.0
            min_time = 1000000000.0
    
            group_data = {
                'algo_name': algo_name,
                'graph_name': graph_name,
                'round': tuning_round,
            }
            
            with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 解析当前行
                    parts = line.split(',')
                    status = parts[0].strip()
                    
                    # 提取 runtime 和 exectime
                    runtime = float(parts[-2].split(':')[1])
                    exectime = float(parts[-1].split(':')[1])

                    # 计算累积时间
                    if status == 'passed':
                        accumulated_time += pass_compile_time + runtime
                        # 记录最小时间
                        min_time = min(min_time, exectime)
                    elif status == 'unpass':
                        accumulated_time += runtime + fail_compile_time
                    else:
                        accumulated_time += runtime
                    
                    # 记录tune_time
                    if min_time == gratuner_result_runtime and 'tune_time' not in group_data:
                        group_data['tune_time'] = accumulated_time

                    # 记录after_1min_exec_time, after_5min_exec_time, after_10min_exec_time
                    if accumulated_time >= 60.0 and 'after_1min_exec_time' not in group_data:
                        group_data['after_1min_exec_time'] = min_time
                    
                    if accumulated_time >= 120.0 and 'after_2min_exec_time' not in group_data:
                        group_data['after_2min_exec_time'] = min_time
                    
                    if accumulated_time >= 300.0 and 'after_5min_exec_time' not in group_data:
                        group_data['after_5min_exec_time'] = min_time
                    
                    if accumulated_time >= 600.0 and 'after_10min_exec_time' not in group_data:
                        group_data['after_10min_exec_time'] = min_time
                
                    # 写入新行
                    new_line = f"{line}, all_time: {accumulated_time}, min_time: {min_time}\n"
                    fout.write(new_line)
                
                # 将当前数据添加到分组数据中
                for key in grouped_data_all[algo_name, graph_name]:
                    if key in group_data:
                        grouped_data_all[algo_name, graph_name][key].append(group_data[key])
    
    # 计算平均值并生成最终结果
    results = []
    for (algo_name, graph_name), data in grouped_data_all.items():
        # result = {
        #     'algo_name': algo_name,
        #     'graph_name': graph_name,
        #     'tune_time': sum(data['tune_time']) / len(data['tune_time']) if data['tune_time'] else None,
        #     'after_1min_exec_time': sum(data['after_1min_exec_time']) / len(data['after_1min_exec_time']) if data['after_1min_exec_time'] else None,
        #     'after_5min_exec_time': sum(data['after_5min_exec_time']) / len(data['after_5min_exec_time']) if data['after_5min_exec_time'] else None,
        #     'after_10min_exec_time': sum(data['after_10min_exec_time']) / len(data['after_10min_exec_time']) if data['after_10min_exec_time'] else None,
        # }
        result = {
            'algo_name': algo_name,
            'graph_name': graph_name,
            'tune_time_mean': sum(data['tune_time']) / len(data['tune_time']) if data['tune_time'] else None,
            'tune_time_median': sorted(data['tune_time'])[len(data['tune_time']) // 2] if data['tune_time'] else None,
            'tune_time_max': max(data['tune_time']) if data['tune_time'] else None,
            'tune_time_min': min(data['tune_time']) if data['tune_time'] else None,
            'after_1min_exec_time_mean': sum(data['after_1min_exec_time']) / len(data['after_1min_exec_time']) if data['after_1min_exec_time'] else None,
            'after_1min_exec_time_median': sorted(data['after_1min_exec_time'])[len(data['after_1min_exec_time']) // 2] if data['after_1min_exec_time'] else None,
            'after_1min_exec_time_max': max(data['after_1min_exec_time']) if data['after_1min_exec_time'] else None,
            'after_1min_exec_time_min': min(data['after_1min_exec_time']) if data['after_1min_exec_time'] else None,
            'after_2min_exec_time_mean': sum(data['after_2min_exec_time']) / len(data['after_2min_exec_time']) if data['after_2min_exec_time'] else None,
            'after_2min_exec_time_median': sorted(data['after_2min_exec_time'])[len(data['after_2min_exec_time']) // 2] if data['after_2min_exec_time'] else None,
            'after_2min_exec_time_max': max(data['after_2min_exec_time']) if data['after_2min_exec_time'] else None,
            'after_2min_exec_time_min': min(data['after_2min_exec_time']) if data['after_2min_exec_time'] else None,
            'after_5min_exec_time_mean': sum(data['after_5min_exec_time']) / len(data['after_5min_exec_time']) if data['after_5min_exec_time'] else None,
            'after_5min_exec_time_median': sorted(data['after_5min_exec_time'])[len(data['after_5min_exec_time']) // 2] if data['after_5min_exec_time'] else None,
            'after_5min_exec_time_max': max(data['after_5min_exec_time']) if data['after_5min_exec_time'] else None,
            'after_5min_exec_time_min': min(data['after_5min_exec_time']) if data['after_5min_exec_time'] else None,
            'after_10min_exec_time_mean': sum(data['after_10min_exec_time']) / len(data['after_10min_exec_time']) if data['after_10min_exec_time'] else None,
            'after_10min_exec_time_median': sorted(data['after_10min_exec_time'])[len(data['after_10min_exec_time']) // 2] if data['after_10min_exec_time'] else None,
            'after_10min_exec_time_max': max(data['after_10min_exec_time']) if data['after_10min_exec_time'] else None,
            'after_10min_exec_time_min': min(data['after_10min_exec_time']) if data['after_10min_exec_time'] else None,
        }
        results.append(result)
    
    # 写入CSV
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def process_time_log_iter(device):
    base_dir = opentuner_output_dir_path_dict[device]

    headers = ['round', 'iter', 'min_exec_time']
    # 存各{algo_name}_{graph_name}的数据，默认值为[]
    results = defaultdict(list)

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if os.path.isdir(dirpath) and dirname.startswith("output_autotune_"):
            parts = dirname.split("_")
            algo_name = parts[2]
            graph_name = "_".join(parts[3:-1])
            round = parts[-1]
            
            out_key = f"{algo_name}_{graph_name}"
            
            input_file = os.path.join(dirpath, "time_log_iter_with_alltime.txt")
            
            iter_id = 0

            with open(input_file, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 解析当前行
                    parts = line.split(',')
                    min_exectime = float(parts[-1].split(':')[1])
                    
                    results[out_key].append({
                        'round': round,
                        'iter': iter_id,
                        'min_exec_time': min_exectime
                    })
                    
                    iter_id += 1
    
    # 写入CSV
    for key, value in results.items():
        out_file = f"/home/zhuhaoran/AutoGraph/AutoGraph/opentuner_iter/{device}/opentuner_iter_{key}.csv"
        with open(out_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(value)


if __name__ == "__main__":
    for device in ['CPU', 'GPU', 'YiTian']:
    # for device in ['CPU', 'YiTian']:
        # process_time_log(device)
        # process_time_log_v2(device)
        process_time_log_iter(device)
        print(f"{device} done ...")
    