
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import random_split, DataLoader

import dgl

import time
import sys
import os
import math

import subprocess
import shlex
from sys import exit

from train import create_dgl_graph, load_data
from data_loader import ScheduleDataset, GraphsDataset, ScheduleGraphDataset, ScheduleDataset_v2
from model import AutoGraphModel, AutoGraphModel_GAT
from common.common import PROJECT_ROOT

GraphitPath = f'{PROJECT_ROOT}/graphit'

py_graphitc_file = f"{GraphitPath}/build/bin/graphitc.py"
serial_compiler = "g++"

# if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++"

max_num_segments = 24

config = {
    'direction': 'SparsePush',
    'parallelization': 'serial',
    'numSSG': ['fixed-vertex-count', '1'],
    'DenseVertexSet': 'boolean-array',
}


config_numSSG = ['fixed-vertex-count']
config_NUMA = ['false', 'static-parallel', 'dynamic-parallel']    # dynamic-parallel doesn't seem to be implemented

config_direction = ['SparsePush', 'DensePull', 'DensePull-SparsePush', 'SparsePush-DensePull', 'DensePush-SparsePush']  # densepush unsupported
config_parallelization = ['serial', 'dynamic-vertex-parallel', 'static-vertex-parallel', 'edge-aware-dynamic-vertex-parallel']
config_DenseVertexSet = ['boolean-array', 'bitvector']


# just for priority graph
# config_bucket_update_strategy = ['eager_priority_update', 'eager_priority_update_with_merge', 'lazy_priority_update']

algo_file_dir = f"{GraphitPath}/autotune/benchmarks/"
# algo_list = ['cc.gt', 'pagerank.gt', 'sssp.gt', 'bfs.gt', 'cf.gt']
algo_list = ['sssp.gt', 'cc.gt', 'pagerank.gt', 'bfs.gt', 'cf.gt']

graph_file_dir = f"{GraphitPath}/autotune/graphs/"
# graph_list = ['dblp-cite', 'dbpedia-team', 'dimacs9-E', 'dimacs10-uk-2002', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#               'sx-stackoverflow', 'youtube-groupmemberships', 'zhishi-all'
#               ]
graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
              'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
              'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
               'youtube-groupmemberships', 
              ]
# graph_list = ['4']

class GraphItDataCreator():

    new_schedule_file_name = ''
    # a flag for testing if NUMA-aware schedule is specified
    use_NUMA = False
    use_eager_update = False

    # this flag is for testing on machine without NUMA library support
    # this would simply not tune NUMA-aware schedules
    enable_NUMA_tuning = True

    # this would simply not tune parallelization related schedules
    # for machines without CILK or openmp support
    enable_parallel_tuning = True

    # enable_denseVertexSet_tuning = True

    # command-line arguments in graphit_autotuner(all defalut)
    default_schedule_file = ""

    latest_schedule = ''
    cfg_to_schedule_pass = True

    def reset_flag(self):
        # a flag for testing if NUMA-aware schedule is specified
        self.use_NUMA = False
        self.use_eager_update = False

        # this flag is for testing on machine without NUMA library support
        # this would simply not tune NUMA-aware schedules
        self.enable_NUMA_tuning = True

        # this would simply not tune parallelization related schedules
        # for machines without CILK or openmp support
        self.enable_parallel_tuning = True

        # self.enable_denseVertexSet_tuning = True
        self.cfg_to_schedule_pass = True

    def call_program(self, command):
        
        cmd_args = command.split(' ')
        print(cmd_args)
        
        start_time = time.time()
        process = subprocess.run(
            cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        end_time = time.time()
        
        run_time = end_time - start_time
        if process.returncode != 0:
            print('call_program warning: ' + process.stderr)
        return {'returncode': process.returncode, 'run_time': run_time}
    
        
    # def schedule_fliter(self, cfg) -> bool:
    #     fliter_pass = True
        
    #     NUMA_enable = False   # 使用numa的要求（计算方向和图划分）是否满足
    #     NUMA_use = False     # cfg是否使用numa
        
    #     edge_aware_par_use = False  # 是否使用边感知并行
    #     edge_aware_par_enable = False # 边感知并行条件是否达到
        
    #     bitvector_enble = False     # bitvector使用条件是否达到
    #     bitvector_use = False       # 是否使用bitvector
        
    #     SSG_enble = False
    #     SSG_use = False
        
    #     if int(cfg['numSSG']) > 1:      # 使用numa时只能使用pull方向或者push-pull方向，且图划分数必须大于1
    #         SSG_use = True
        
    #     if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull' or cfg['direction'] == 'DensePull-SparsePush':
    #         edge_aware_par_enable = True    # 边感知并行只在pull方向或者push-pull方向有效
            
    #         bitvector_enble = True          # bitvector也只在pull方向或者push-pull方向有效
            
    #         SSG_enble = True                # SSG也只在pull方向或者push-pull方向有效
            
    #         if SSG_use and cfg['parallelization'] != 'serial':      # 使用numa时必须允许并行，且只能使用pull方向或者push-pull方向，且图划分数必须大于1, 
    #             NUMA_enable = True
        
    #     # 当使用SSG但SSG的使用条件不满足时，过滤不通过
    #     if SSG_use == True and SSG_enble == False:
    #         print(f'schedule_fliter: ssg use enable unpass')
    #         fliter_pass = False
    #         return False
                
    #     if cfg['NUMA'] == 'static-parallel' or cfg['NUMA'] == 'dynamic-parallel':
    #         NUMA_use = True
        
    #     # 当使用numa，但num使用要求不满足时，过滤不通过
    #     if NUMA_use == True and NUMA_enable == False:
    #         print(f'schedule_fliter: numa use enable unpass')
    #         fliter_pass = False
    #         return False
        
    #     if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
    #         edge_aware_par_use = True
        
    #     # 当使用numa时，不能使用边感知的并行，过滤不通过
    #     if edge_aware_par_use == True and NUMA_use == True:
    #         print(f'schedule_fliter: numa & edge_aware_par_use unpass')
    #         fliter_pass = False
    #         return False
            
    #     # 当使用边感知的并行但边感知的并行的使用条件不满足时，过滤不通过
    #     if edge_aware_par_use == True and edge_aware_par_enable == False:
    #         print(f'schedule_fliter: edge_aware_par use enable unpass')
    #         fliter_pass = False
    #         return False
        
    #     if cfg['DenseVertexSet'] == 'bitvector':
    #         bitvector_use = True
        
    #     # 当使用bitvector但bitvector的使用条件不满足时，过滤不通过
    #     if bitvector_use == True and bitvector_enble == False:
    #         print(f'schedule_fliter: bitvector use enable unpass')
    #         fliter_pass = False
    #         return False
        
    #     return fliter_pass

    # configures parallelization commands
    def write_par_schedule(self, cfg, new_schedule, direction):
        use_evp = False

        if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
            use_evp = True

        # if use_evp == False or self.use_NUMA == True:
        if use_evp == False:
            # if don't use edge-aware parallel (vertex-parallel)
            # edge-parallel don't work with NUMA (use vertex-parallel when NUMA is enabled)
            if cfg['parallelization'] == 'serial':
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"serial\");"
                    
            elif cfg['parallelization'] == 'dynamic-vertex-parallel':
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
                    
            elif cfg['parallelization'] == 'static-vertex-parallel':
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"static-vertex-parallel\");"
            else:
                self.cfg_to_schedule_pass = False
                print("Error in writing parallel schedule 1")
                # return None
        elif use_evp == True and self.use_NUMA == False:
            # use_evp is True
            if direction == "DensePull":
                # edge-aware-dynamic-vertex-parallel is only supported for the DensePull direction
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024, \"DensePull\");"

            elif direction == "SparsePush-DensePull" or direction == "DensePull-SparsePush":
                # For now, only the DensePull direction uses edge-aware-vertex-parallel
                # the SparsePush should still just use the vertex-parallel methodx
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024,  \"DensePull\");"

                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\",1024,  \"SparsePush\");"

            else:
                # use_evp for SparsePush, DensePush-SparsePush should not make a difference
                # new_schedule = new_schedule + \
                #     "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
                
                self.cfg_to_schedule_pass = False
                print("Error in writing parallel schedule 2")
                # return None
        else:
            self.cfg_to_schedule_pass = False
            print("Error in writing parallel schedule 3")
            # return None
        return new_schedule

    def write_numSSG_schedule(self, numSSG, new_schedule, direction):
        # No need to insert for a single SSG
        if numSSG == 0 or numSSG == 1:
            return new_schedule
        # configuring cache optimization for DensePull direction
        if direction == "DensePull" or direction == "SparsePush-DensePull" or direction == 'DensePull-SparsePush':
            new_schedule = new_schedule + \
                "\n    program->configApplyNumSSG(\"s1\", \"fixed-vertex-count\", " + str(numSSG) + ", \"DensePull\");"
        else:
            self.cfg_to_schedule_pass = False
            print("Error in writing numSSG schedule")
            # return None
        return new_schedule

    def write_NUMA_schedule(self, cfg, new_schedule, direction):
        # configuring NUMA optimization for DensePull direction
        if self.use_NUMA:
            if direction == "DensePull" or direction == "SparsePush-DensePull" or cfg['direction'] == 'DensePull-SparsePush':
                new_schedule = new_schedule + f"\n    program->configApplyNUMA(\"s1\", \"{cfg['NUMA']}\" , \"DensePull\");"
            else:
                self.cfg_to_schedule_pass = False
                print("Error in writing NUMA schedule")
                # return None
        
        return new_schedule

    def write_denseVertexSet_schedule(self, use_pull_bitvector, new_schedule, direction):
        # for now, we only use this for the src vertexset in the DensePull direciton
        if use_pull_bitvector:
            if direction == "DensePull" or direction == "SparsePush-DensePull" or direction == 'DensePull-SparsePush':
                new_schedule = new_schedule + "\n    program->configApplyDenseVertexSet(\"s1\",\"bitvector\", \"src-vertexset\", \"DensePull\");"
            else:
                self.cfg_to_schedule_pass = False
                print("Error in writing denseVertexSet schedule")
                # return None
        return new_schedule

    def write_cfg_to_schedule(self, cfg):
        # write into a schedule file the configuration
        direction = cfg['direction']
        numSSG = cfg['numSSG']

        new_schedule = ""
        direction_schedule_str = "\n    program->configApplyDirection(\"s1\", \"$direction\");"
        if self.default_schedule_file != "":
            f = open(self.default_schedule_file, 'r')
            default_schedule_str = f.read()
            f.close()
        else:
            default_schedule_str = "schedule: "

        new_schedule = default_schedule_str + \
            direction_schedule_str.replace('$direction', cfg['direction'])

        new_schedule = self.write_par_schedule(cfg, new_schedule, direction)
        new_schedule = self.write_numSSG_schedule(
            numSSG, new_schedule, direction)
        new_schedule = self.write_NUMA_schedule(cfg, new_schedule, direction)

        use_bitvector = False
        if cfg['DenseVertexSet'] == 'bitvector':
            use_bitvector = True
        new_schedule = self.write_denseVertexSet_schedule(
            use_bitvector, new_schedule, direction)
        
        if self.cfg_to_schedule_pass == False:
            print(f" write_cfg_to_schedule unpass: \n{new_schedule}")
            return -1
        
        # elif new_schedule == self.latest_schedule:
        #     print(f"same schedule: \n{new_schedule}")
        #     return -1
        
        # self.latest_schedule = new_schedule
        
        print(cfg)
        print(new_schedule)

        self.new_schedule_file_name = 'schedule_0'
        print(self.new_schedule_file_name)

        f1 = open(self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()
        
        return 0


    def compile(self, cfg, algo_file_, id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """

        # compile the schedule file along with the original algorithm file
        compile_graphit_cmd = 'python ' + py_graphitc_file + \
            ' -a {algo_file} -f {schedule_file} -i ../include/ -l ../build/lib/libgraphitlib.a -o test.cpp'.format(
                algo_file=algo_file_, schedule_file=self.new_schedule_file_name)

        if not self.use_NUMA:
            if not self.enable_parallel_tuning:
                # if parallel icpc compiler is not needed (only tuning serial schedules)
                compile_cpp_cmd = serial_compiler + \
                    ' -std=gnu++1y -I ../src/runtime_lib/ -O3 test.cpp -o test'
            else:
                # if parallel icpc compiler is supported and needed
                # compile_cpp_cmd = par_compiler + \
                #     ' -std=gnu++1y -DCILK -fcilkplus -I ../src/runtime_lib/ -O3 test.cpp -o test'
                compile_cpp_cmd = par_compiler + \
                    ' -std=gnu++1y -DOPENMP -fopenmp -I ../src/runtime_lib/ -O3 test.cpp -o test'
        else:
            # add the additional flags for NUMA
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -lnuma -DNUMA -fopenmp -I ../src/runtime_lib/ -O3 test.cpp -o test'

        if self.use_eager_update:
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -fopenmp -I ../src/runtime_lib/ -O3 test.cpp -o test'

        print(compile_graphit_cmd)
        print(compile_cpp_cmd)

        self.call_program(compile_graphit_cmd)
        # try:
        #     self.call_program(compile_graphit_cmd)
        # except:
        #     print ("fail to compile .gt file")
        #     return None
        
        return self.call_program(compile_cpp_cmd)

    def run_precompiled(self, cfg, compile_result, graph, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """

        if compile_result['returncode'] != 0:
            print(str(compile_result))

        assert compile_result['returncode'] == 0
        try:
            # run_result = self.call_program('./test ../test/graphs/socLive_gapbs.sg > test.out')
            # run_result = self.call_program('./test ../test/graphs/4.sg > test.out')
            if not self.use_NUMA:
                if not self.enable_parallel_tuning:
                    # don't use numactl when running serial
                    run_cmd = './test ' + graph + ' > test.out'

                else:
                    # use numactl when running parallel
                    run_cmd = 'numactl -i all ./test ' + graph + ' > test.out'
            else:
                run_cmd = 'OMP_PLACES=sockets ./test ' + graph + ' > test.out'

            print("run_cmd: " + run_cmd)

            # default value -1 for memory_limit translates into None (no memory upper limit)
            # setting memory limit does not quite work yet
            # process_memory_limit = None
            # if self.memory_limit != -1:
            #     process_memory_limit = self.memory_limit
            # print ("memory limit: " + str(process_memory_limit))
            run_result = self.call_program(run_cmd)
        finally:
            self.call_program('rm test')
            self.call_program('rm test.cpp')

        # self.call_program('rm test.out')

        if run_result['returncode'] != 0:
            print('running error, return code :' + str(run_result['returncode']))
            return None
        else:
            print("running over, running time: " + str(run_result['run_time']))
            return run_result['run_time']

    def compile_and_run(self, cfg, graph, algo_file):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        self.reset_flag()   # 重置一些标志位
        
        if cfg['NUMA'] == 'false':
            self.enable_NUMA_tuning = False
        if cfg['parallelization'] == 'serial':
            self.enable_parallel_tuning = False

        # only use NUMA when we are tuning parallel and NUMA schedules
        # if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] == 'static-parallel':
        if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] != 'false':
            if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull' or cfg['direction'] == 'DensePull-SparsePush':
                if int(cfg['numSSG']) > 1:
                    self.use_NUMA = True

        # if cfg['bucket_update_strategy'] == "eager_priority_update" or cfg['bucket_update_strategy'] == "eager_priority_update_with_merge":
        #     self.use_eager_update = True
        
        # converts the configuration into a schedule
        returncode = self.write_cfg_to_schedule(cfg)
        
        # write_cfg_to_schedule error return None
        if returncode == -1:
            return None
        
        print (" write_cfg_to_schedule over. ")

        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, algo_file, 0)
        if compile_result["returncode"] != 0:
            return None
        
        print (" compile_result over. ")
        return self.run_precompiled(cfg, compile_result, graph, 0)



def schedule_fliter(cfg) -> bool:
    fliter_pass = True
    
    NUMA_enable = False   # 使用numa的要求（计算方向和图划分）是否满足
    NUMA_use = False     # cfg是否使用numa
    
    edge_aware_par_use = False  # 是否使用边感知并行
    edge_aware_par_enable = False # 边感知并行条件是否达到
    
    bitvector_enble = False     # bitvector使用条件是否达到
    bitvector_use = False       # 是否使用bitvector
    
    SSG_enble = False
    SSG_use = False
    
    if int(cfg['numSSG']) > 1:      # 使用numa时只能使用pull方向或者push-pull方向，且图划分数必须大于1
        SSG_use = True
    
    if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull' or cfg['direction'] == 'DensePull-SparsePush':
        edge_aware_par_enable = True    # 边感知并行只在pull方向或者push-pull方向有效
        
        bitvector_enble = True          # bitvector也只在pull方向或者push-pull方向有效
        
        SSG_enble = True                # SSG也只在pull方向或者push-pull方向有效
        
        if SSG_use and cfg['parallelization'] != 'serial':      # 使用numa时必须允许并行，且只能使用pull方向或者push-pull方向，且图划分数必须大于1, 
            NUMA_enable = True
    
    # 当使用SSG但SSG的使用条件不满足时，过滤不通过
    if SSG_use == True and SSG_enble == False:
        print(f'schedule_fliter: ssg use enable unpass')
        fliter_pass = False
        return False
            
    if cfg['NUMA'] == 'static-parallel' or cfg['NUMA'] == 'dynamic-parallel':
        NUMA_use = True
    
    # 当使用numa，但num使用要求不满足时，过滤不通过
    if NUMA_use == True and NUMA_enable == False:
        print(f'schedule_fliter: numa use enable unpass')
        fliter_pass = False
        return False
    
    if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
        edge_aware_par_use = True
    
    # 当使用numa时，不能使用边感知的并行，过滤不通过
    if edge_aware_par_use == True and NUMA_use == True:
        print(f'schedule_fliter: numa & edge_aware_par_use unpass')
        fliter_pass = False
        return False
        
    # 当使用边感知的并行但边感知的并行的使用条件不满足时，过滤不通过
    if edge_aware_par_use == True and edge_aware_par_enable == False:
        print(f'schedule_fliter: edge_aware_par use enable unpass')
        fliter_pass = False
        return False
    
    if cfg['DenseVertexSet'] == 'bitvector':
        bitvector_use = True
    
    # 当使用bitvector但bitvector的使用条件不满足时，过滤不通过
    if bitvector_use == True and bitvector_enble == False:
        print(f'schedule_fliter: bitvector use enable unpass')
        fliter_pass = False
        return False
    
    return fliter_pass


def all_true_sche_create():

    cfg =  {
        'direction': 'SparsePush',
        'parallelization': 'serial',
        'DenseVertexSet': 'boolean-array',
        'NUMA': 'false',
        'numSSG': 0
    }

    # for i in range(0, max_num_segments, 2):
    for i in [0, 5, 10, 15, 20]:
        cfg['numSSG'] = i
        
        for NUMA_option in config_NUMA:     # numa 编译有问题
            cfg['NUMA'] = NUMA_option
            
            if cfg['NUMA'] != 'false':      # numa 编译有问题
                continue
            
            for direction_option in config_direction:
                cfg['direction'] = direction_option

                for parallelization_option in config_parallelization:
                    cfg['parallelization'] = parallelization_option

                    for DenseVertexSet_option in config_DenseVertexSet:
                        cfg['DenseVertexSet'] = DenseVertexSet_option

                        passed = schedule_fliter(cfg)
                        if passed: # 如果能够通过过滤，则调度组合有效
                            print(f"schedule pass: {cfg}")
                            
                            output_file_name = f'{PROJECT_ROOT}/dataset/dataset_MKL/all_true_schedule.csv'
                            with open(output_file_name, 'a') as file:
                                output_sche = 'none,'
                                for key, value in cfg.items():
                                    output_sche += f"{value},"
                                file.write(output_sche+'0.0'+'\n')


if __name__ == '__main__':
    # true_data_path = f'{PROJECT_ROOT}/test/grouped/dblp-cite_pagerank.gt_output.csv'
    # true_all_data = pd.read_csv(true_data_path, comment='#', sep=',', header=None)
    
    # y = true_all_data[6].values
    # y = torch.tensor(y)
    
    # test_file_path = f'{PROJECT_ROOT}/dataset/dataset_MKL/all_true_schedule.csv'
    # test_all_data = pd.read_csv(test_file_path, comment='#', sep=',', header=None)
    # test_all_data = test_all_data.values

    graph_name = 'dblp-cite'
    
    test_all_data = load_data(f'{PROJECT_ROOT}/dataset/dataset_MKL/train/pagerank.csv')
    test_all_data = test_all_data['dblp-cite']
    
    device = torch.device("cuda:" + str(2) if torch.cuda.is_available() else "cpu")
    model = AutoGraphModel()
    
    checkpoint = torch.load(f"{PROJECT_ROOT}/cost_model/cost_model_MKL/costmodel_v2_best.pth")
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # for graph_name in graph_list:
    
    Test_Schedules_Dataset = ScheduleDataset_v2(test_all_data)
    test_data_schedule = DataLoader(Test_Schedules_Dataset, batch_size=128, shuffle=True)
        
    g = create_dgl_graph(graph_name)
    g = g.to(device)

    model.eval()
    with torch.no_grad():
        for val_batchidx, (schedule, y) in enumerate(test_data_schedule):

            schedule = schedule.to(device)
            y = y.to(device)
            
            start_time = time.time()
            graph_feature = model.embed_sparse_matrix(g)
            graph_feature = graph_feature.expand((schedule.shape[0], graph_feature.shape[1]))

            # 前向传播
            predict = model.forward_after_query(graph_feature, schedule)
            
            predict = predict.squeeze()
            
            # values, indices = predict.topk(10, largest=False)
            end_time = time.time()
            
            print('time: ', end_time - start_time)
            
            # print(values)
            # print(indices)
            # for i in indices:
            #     print(test_all_data[i])

            # HingeRankingLoss
            iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
            pred1, pred2 = predict[iu[0]], predict[iu[1]]
            true1, true2 = y[iu[0]], y[iu[1]]
            sign = (true1-true2).sign()
            sign[sign==0] = 1

            # 计算排序情况
            pred_sign = torch.sign(pred1 - pred2)
            equal_count = (sign == pred_sign).sum().item()
            # unequal_count = (sign != pred_sign).sum().item()
            
            # rank_equal_num += equal_count
            # rank_all_num += len(sign)
            accuracy = equal_count / len(sign)
            print(accuracy)
    exit(1)
