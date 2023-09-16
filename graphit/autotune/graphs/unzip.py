import os
import subprocess

# 压缩包所在目录路径
directory = '/data/zhr_data/AutoGraph/graphit/autotune/graphs'

# 遍历目录下的文件
for filename in os.listdir(directory):
    if filename.startswith('download') and filename.endswith('.tar.bz2'):
        filepath = os.path.join(directory, filename)
        try:
            # 构造解压命令
            command = f'tar -xvjf {filepath}'
            # 调用终端命令解压压缩包
            subprocess.run(command, shell=True, check=True)
            print(f'Successfully extracted: {filename}')
        except subprocess.CalledProcessError as e:
            print(f'Error extracting {filename}: {str(e)}')