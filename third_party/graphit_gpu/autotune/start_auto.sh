#!/bin/bash

# 使用示例 ./start_auto.sh bfs github 1
# 检查是否传入了足够的参数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 algo_name data_name round_id"
    exit 1
fi

# 从命令行参数读取算法文件名和轮数
algo_name=$1
data_name=$2
round=$3

out_dir=/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/GraTuner_tests_exectime

# 删除旧的数据库和输出文件
rm -rf /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/opentuner.db/*
rm -rf ${out_dir}/output_autotune/*
rm -rf /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/final_config.json
rm -rf /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/opentuner.log

# 运行自动调优程序
python graphit_gpu_autotuner.py --algo_file ${algo_name}.gt --graph /home/zhuhaoran/AutoGraph/graphs/${data_name}/${data_name}.el --graph_name ${data_name} --stop-after 10800 --test-limit 500
# python graphit_gpu_autotuner.py --algo_file pagerank.gt --graph /home/zhuhaoran/AutoGraph/graphs/youtube-u-growth/youtube-u-growth.el --graph_name youtube-u-growth --stop-after 10800 --test-limit 500

# 复制output_autotune文件夹并重命名
rm -rf ${out_dir}/output_autotune_${algo_name}_${data_name}_v${round}
cp -r ${out_dir}/output_autotune ${out_dir}/output_autotune_${algo_name}_${data_name}_v${round}
mv /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/final_config.json ${out_dir}/output_autotune_${algo_name}_${data_name}_v${round}/final_config_${algo_name}_${data_name}_v${round}.json
mv /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/opentuner.log ${out_dir}/output_autotune_${algo_name}_${data_name}_v${round}/opentuner.log
mv /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/${algo_name}_${data_name}_${round}.log ${out_dir}/output_autotune_${algo_name}_${data_name}_v${round}/${algo_name}_${data_name}_${round}.log