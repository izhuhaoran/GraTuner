import os


def process_time_log(input_file, output_file, pass_compile_time=19.9685342840988, fail_compile_time=0.0):
    accumulated_time = 0.0
    min_time = 1000000000.0
    
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

            # for part in parts:
            #     if 'run_time:' in part:
            #         runtime = float(part.split(':')[1])
            #         break
            
            # 计算累积时间
            if status == 'passed':
                accumulated_time += pass_compile_time + runtime
            else:
                accumulated_time += runtime + fail_compile_time
            
            # 记录最小时间
            if exectime < min_time:
                min_time = exectime
            
            # 写入新行
            new_line = f"{line}, all_time: {accumulated_time}, min_time: {min_time}\n"
            fout.write(new_line)

if __name__ == "__main__":
    # input_file = "time_log_iter.txt"
    # output_file = "time_log_iter_with_alltime.txt"
    # process_time_log(input_file, output_file)

    # 对{PROJECT_ROOT}/graphit/autotune/GraTuner_tests_2中所有一级子目录文件夹中time_log_iter.txt进行处理
    base_dir = "{PROJECT_ROOT}/graphit/autotune/GraTuner_tests_2"
    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if os.path.isdir(dirpath):
            input_file = os.path.join(dirpath, "time_log_iter.txt")
            output_file = os.path.join(dirpath, "time_log_iter_with_alltime.txt")
            process_time_log(input_file, output_file)