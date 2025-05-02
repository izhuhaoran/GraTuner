import os
import re
import csv

def process_log_files(log_dir, output_file):
    # log_dir = "/home/zhuhaoran/AutoGraph/AutoGraph/search"
    # output_file = "search_results.csv"
    headers = ['algo_name', 'graph_name', 'Create graph time', 'Embed graph time', 
              'Embed algo time', 'Initialization active set time', 'Traversal search time',
              'Final result time', 'All time', 'best_result_runtime']
    
    results = []
    for filename in os.listdir(log_dir):
        if not filename.startswith("search_print_iter_") or not filename.endswith(".log"):
            continue
            
        # 解析文件名获取algo和graph
        parts = filename.replace("search_print_iter_", "").replace(".log", "").split("_")
        algo_name = parts[0]
        graph_name = "_".join(parts[1:])
        
        with open(os.path.join(log_dir, filename), 'r') as f:
            lines = f.readlines()
            
            group_data = {}
            group_count = 0
            
            for line in lines:
                if "Create graph time" in line:
                    # 新的一组开始
                    if group_data and len(group_data) > 0:
                        group_data['algo_name'] = algo_name
                        group_data['graph_name'] = graph_name
                        results.append(group_data.copy())
                    group_data = {}
                    group_count += 1
                
                # 提取时间值
                for key in headers[2:-2]:  # 除algo_name, graph_name, All time, best_result_runtime外的字段
                    if key in line:
                        try:
                            # time_value = float(re.search(f"{key}: (\d+\.?\d*)", line).group(1))
                            time_value = float(line.split(f'{key}:')[1].strip())
                            group_data[key] = time_value
                        except:
                            continue
                
                # 处理包含All time和best runtime的行
                if "All time" in line and "Top k nodes for" in line:
                    try:

                        group_data['All time'] = float(line.split('All time')[1].split(',')[0].strip())
                        # 提取best_result_runtime
                        best_result_runtime = float(line.split('Top k nodes for')[1].split(':')[1].strip().split(',')[0].strip('[]()'))
                        group_data['best_result_runtime'] = best_result_runtime
                            
                    except:
                        continue
            
            # 添加最后一组数据
            if group_data and len(group_data) > 0:
                group_data['algo_name'] = algo_name
                group_data['graph_name'] = graph_name
                results.append(group_data)
    
    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def process_log_files_get_iter(log_dir, output_file):

    headers = ['algo_name', 'graph_name', 'round', 'iter', 'best_result_runtime']
    
    results = []
    for filename in os.listdir(log_dir):
        if not filename.startswith("search_print_iter_") or not filename.endswith(".log"):
            continue
            
        # 解析文件名获取algo和graph
        parts = filename.replace("search_print_iter_", "").replace(".log", "").split("_")
        algo_name = parts[0]
        graph_name = "_".join(parts[1:])
        
        with open(os.path.join(log_dir, filename), 'r') as f:
            lines = f.readlines()
            round_id = 0
            
            for line in lines:
                if "Create graph time" in line:
                    round_id += 1
                
                # 处理包含All time和best runtime的行
                if "iter" in line and "Top k runtime" in line:
                    try:
                        # 提取iter id
                        iter_id = int(line.split('iter')[1].split(',')[0].strip())
                        # 提取best_result_runtime
                        best_result_runtime = float(line.split('Top k runtime')[1].split(':')[1].strip().split(',')[0].strip('[]()'))
                        results.append(
                            {
                                'algo_name': algo_name,
                                'graph_name': graph_name,
                                'round': round_id,
                                'iter': iter_id,
                                'best_result_runtime': best_result_runtime
                            }
                        )
                            
                    except:
                        continue

    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def process_log_files_for_diff_m(log_dir, output_file):
    headers = ['m', 'algo_name', 'graph_name', 'Create graph time', 'Embed graph time', 
              'Embed algo time', 'Initialization active set time', 'Traversal search time',
              'Final result time', 'All time', 'best_result_runtime']
    
    results = []
    for filename in os.listdir(log_dir):
        if not filename.startswith("search_diff_topm_iter_") or not filename.endswith(".log"):
            continue
            
        # 解析文件名获取algo和graph
        # 示例：search_iter_log/search_diff_topm_iter_bfs_soc-LiveJournal1_top_m1.log
        parts = filename.replace("search_diff_topm_iter_", "").replace(".log", "").split("_")
        algo_name = parts[0]
        graph_name = "_".join(parts[1:-2])
        m_val = parts[-1]
        
        with open(os.path.join(log_dir, filename), 'r') as f:
            lines = f.readlines()
            
            group_data = {}
            group_count = 0
            
            for line in lines:
                if "Create graph time" in line:
                    # 新的一组开始
                    if group_data and len(group_data) > 0:
                        group_data['algo_name'] = algo_name
                        group_data['graph_name'] = graph_name
                        group_data['m'] = m_val
                        results.append(group_data.copy())
                    group_data = {}
                    group_count += 1
                
                # 提取时间值
                for key in headers[2:-2]:  # 除algo_name, graph_name, All time, best_result_runtime外的字段
                    if key in line:
                        try:
                            # time_value = float(re.search(f"{key}: (\d+\.?\d*)", line).group(1))
                            time_value = float(line.split(f'{key}:')[1].strip())
                            group_data[key] = time_value
                        except:
                            continue
                
                # 处理包含All time和best runtime的行
                if "All time" in line and "Top k nodes for" in line:
                    try:

                        group_data['All time'] = float(line.split('All time')[1].split(',')[0].strip())
                        # 提取best_result_runtime
                        best_result_runtime = float(line.split('Top k nodes for')[1].split(':')[1].strip().split(',')[0].strip('[]()'))
                        group_data['best_result_runtime'] = best_result_runtime
                            
                    except:
                        continue
            
            # 添加最后一组数据
            if group_data and len(group_data) > 0:
                group_data['algo_name'] = algo_name
                group_data['graph_name'] = graph_name
                group_data['m'] = m_val
                results.append(group_data)
    
    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def process_log_files_for_diff_k(log_dir, output_file):
    headers = ['k', 'algo_name', 'graph_name', 'Create graph time', 'Embed graph time', 
              'Embed algo time', 'Initialization active set time', 'Traversal search time',
              'Final result time', 'All time', 'best_result_runtime']
    
    results = []
    for filename in os.listdir(log_dir):
        if not filename.startswith("search_diff_k_iter_") or not filename.endswith(".log"):
            continue
            
        # 解析文件名获取algo和graph
        # 示例：search_diff_k_iter_sssp_roadNet-CA_k5.log
        parts = filename.replace("search_diff_k_iter_", "").replace(".log", "").split("_")
        algo_name = parts[0]
        graph_name = "_".join(parts[1:-1])
        k_val = parts[-1]
        
        with open(os.path.join(log_dir, filename), 'r') as f:
            lines = f.readlines()
            
            group_data = {}
            group_count = 0
            
            for line in lines:
                if "Create graph time" in line:
                    # 新的一组开始
                    if group_data and len(group_data) > 0:
                        group_data['algo_name'] = algo_name
                        group_data['graph_name'] = graph_name
                        group_data['k'] = k_val
                        results.append(group_data.copy())
                    group_data = {}
                    group_count += 1
                
                # 提取时间值
                for key in headers[2:-2]:  # 除algo_name, graph_name, All time, best_result_runtime外的字段
                    if key in line:
                        try:
                            # time_value = float(re.search(f"{key}: (\d+\.?\d*)", line).group(1))
                            time_value = float(line.split(f'{key}:')[1].strip())
                            group_data[key] = time_value
                        except:
                            continue
                
                # 处理包含All time和best runtime的行
                if "All time" in line and "Top k nodes for" in line:
                    try:

                        group_data['All time'] = float(line.split('All time')[1].split(',')[0].strip())
                        # 提取best_result_runtime
                        best_result_runtime = float(line.split('Top k nodes for')[1].split(':')[1].strip().split(',')[0].strip('[]()'))
                        group_data['best_result_runtime'] = best_result_runtime
                            
                    except:
                        continue
            
            # 添加最后一组数据
            if group_data and len(group_data) > 0:
                group_data['algo_name'] = algo_name
                group_data['graph_name'] = graph_name
                group_data['k'] = k_val
                results.append(group_data)
    
    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    # process_log_files('/home/zhuhaoran/AutoGraph/AutoGraph/search', '/home/zhuhaoran/AutoGraph/AutoGraph/search_results.csv')
    # process_log_files('/home/zhuhaoran/AutoGraph/AutoGraph/search_GPU', '/home/zhuhaoran/AutoGraph/AutoGraph/search_results_GPU.csv')
    # process_log_files('/home/zhuhaoran/AutoGraph/AutoGraph/search_YiTian', '/home/zhuhaoran/AutoGraph/AutoGraph/search_results_YiTian.csv')

    # process_log_files_get_iter('/home/zhuhaoran/AutoGraph/AutoGraph/search/search_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_iters.csv')
    # process_log_files_get_iter('/home/zhuhaoran/AutoGraph/AutoGraph/search_GPU/search_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_iters_GPU.csv')
    # process_log_files_get_iter('/home/zhuhaoran/AutoGraph/AutoGraph/search_YiTian/search_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_iters_YiTian.csv')
    
    # process_log_files_for_diff_m('/home/zhuhaoran/AutoGraph/AutoGraph/search/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_m.csv')
    # process_log_files_for_diff_m('/home/zhuhaoran/AutoGraph/AutoGraph/search_GPU/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_m_GPU.csv')
    # process_log_files_for_diff_m('/home/zhuhaoran/AutoGraph/AutoGraph/search_YiTian/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_m_YiTian.csv')

    process_log_files_for_diff_k('/home/zhuhaoran/AutoGraph/AutoGraph/search/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_k.csv')
    process_log_files_for_diff_k('/home/zhuhaoran/AutoGraph/AutoGraph/search_GPU/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_k_GPU.csv')
    process_log_files_for_diff_k('/home/zhuhaoran/AutoGraph/AutoGraph/search_YiTian/search_iter_log', '/home/zhuhaoran/AutoGraph/AutoGraph/search_diff_k_YiTian.csv')

