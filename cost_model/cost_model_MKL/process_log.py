from common.common import PROJECT_ROOT

def process_log(input_log_path, output_path):
    with open(input_log_path, 'r') as f, open(output_path, 'w') as f_out:
        lines = f.readlines()
        for line in lines:
            if 'Train graph --- Epoch 199 : ' in line:
                # 示例：2025-01-19 07:36:17,218 --- Train graph --- Epoch 199 : Hardware intel Algo bfs Graph youtube-u-growth Train Acc 0.7743109724388976 Kendall 0.5486219448777951 ---
                part = line.split('Train graph --- Epoch 199 : ')[1].split(' ')
                hardware = part[1]
                algo = part[3]
                graph = part[5]
                train_acc = part[8]
                train_kendall = part[10]
                f_out.write(f'{hardware},{algo},{graph},{train_acc},{train_kendall}\n')


# input_log_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result_nograph/trainlog_no_graph_finetune_mkl.log'
# output_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result_nograph/epoch199.csv'
# process_log(input_log_path, output_path)

# input_log_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result/trainlog_gcn_transformer_finetune_mkl.log'
# output_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_result/epoch199.csv'
# process_log(input_log_path, output_path)

# input_log_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_with_algo/trainlog_gcn_transformer_algo_v2_mkl.log'
# output_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/train_with_algo/epoch199.csv'
# process_log(input_log_path, output_path)

input_log_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/trainlog_gcn_transformer_algo_mkl.log'
output_path = f'{PROJECT_ROOT}/cost_model/cost_model_MKL/finetune_result/epoch199.csv'
process_log(input_log_path, output_path)
