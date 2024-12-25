#!/bin/bash

# 使用示例 ./start_auto.sh bfs 2
# 检查是否传入了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 algo_name round_id"
    exit 1
fi

# 从命令行参数读取算法文件名和轮数
algo_name=$1
round=$2

# 删除旧的数据库和输出文件
rm -rf /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/opentuner.db/node8.db
rm -rf /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/output_autotune/*

# 运行自动调优程序
python graphit_autotuner.py --enable_parallel_tuning 1 --algo_file apps/${algo_name}_benchmark.gt --graph /home/zhuhaoran/AutoGraph/graphs/github/github.el --stop-after 7200

# 复制output_autotune文件夹并重命名
cp -r /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/output_autotune /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/output_autotune_${algo_name}_v${round}
mv /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/final_config.json /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/output_autotune_${algo_name}_v${round}/final_config_${algo_name}_v${round}.json
mv /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/opentuner.log /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/output_autotune_${algo_name}_v${round}/opentuner.log