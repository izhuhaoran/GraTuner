import sys

def reorder_columns(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            # 跳过注释或空行
            if not line or line.startswith('//') or line.startswith('#'):
                fout.write(line + '\n')
                continue

            cols = line.split(',')
            # 去掉首尾空格
            cols = [c.strip() for c in cols]
            
            # 倒数第四列索引
            idx_minus_4 = len(cols) - 4
            # 倒数第三列索引
            idx_minus_3 = len(cols) - 3
            # 第二列索引（从0开始计）
            idx_2 = 1

            # 提取要移动的列
            col_minus_4 = cols[idx_minus_4]
            col_minus_3 = cols[idx_minus_3]

            # 构建新列列表
            new_cols = []
            for i, val in enumerate(cols):
                if i == idx_2:
                    # 先放第二列
                    new_cols.append(val)
                    # 插入倒数第四列和倒数第三列
                    new_cols.append(col_minus_4)
                    new_cols.append(col_minus_3)
                elif i in (idx_minus_4, idx_minus_3):
                    # 原本位置的倒数第四、倒数第三列跳过
                    continue
                else:
                    new_cols.append(val)

            fout.write(', '.join(new_cols) + '\n')

for algo in ['pagerank', 'sssp', 'bfs', 'cc']:
    in_path = f'/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/costmodel_data/{algo}.gt_runtime.txt'
    out_path = f'/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/costmodel_data/{algo}_new.gt_runtime.txt'
    reorder_columns(in_path, out_path)