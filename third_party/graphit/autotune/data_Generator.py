import random
import itertools
import time
import sys
import os
import math

import subprocess
import shlex

from sys import exit


GraphitPath = '/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit'

py_graphitc_file = f"{GraphitPath}/build/bin/graphitc.py"
serial_compiler = "g++-7"

output_file_dir = f"{GraphitPath}/autotune/costmodel_dataset/"

# if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++-7"

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
algo_list = ['sssp.gt', 'cc.gt', 'pagerank.gt', 'bfs.gt']

graph_file_dir = "/home/zhuhaoran/AutoGraph/graphs/"
# graph_list = ['dblp-cite', 'dbpedia-team', 'dimacs9-E', 'dimacs10-uk-2002', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#               'sx-stackoverflow', 'youtube-groupmemberships', 'zhishi-all'
#               ]
# 
graph_list = ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA']
# graph_list = ['4'], 

max_time = 10000

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

    def call_program(self, cmd):
        t0 = time.time()
        if type(cmd) in (str, str):
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        else:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t1 = time.time()
        return {'run_time': (t1 - t0),
                'returncode': p.returncode,
                'stdout': p.stdout,
                'stderr': p.stderr}
    
    def schedule_fliter(self, cfg) -> bool:
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
            if direction == "DensePull" or direction == "SparsePush-DensePull" or direction == 'DensePull-SparsePush':
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
        

        print(cfg)
        print(new_schedule)

        self.new_schedule_file_name = 'schedule_0'
        print(self.new_schedule_file_name)

        f1 = open(self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()
        
        return 0


    def compile(self, cfg, algo_file_, algo_name=''):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
        self.call_program('rm test')
        self.call_program('rm test.cpp')

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

        result1 = self.call_program(compile_graphit_cmd)
        if result1['returncode'] != 0:
            with open(f'{output_file_dir}/error_compile_graphit.txt', '+a') as fw:
                fw.write(f"error graphitc, {str(cfg)}, {compile_graphit_cmd}\n")
            return {'returncode': -1, 'run_time': 0.0}
        # try:
        #     self.call_program(compile_graphit_cmd)
        # except:
        #     print ("fail to compile .gt file")
        #     return None

        result2 = self.call_program(compile_cpp_cmd)
        if result2['returncode'] != 0:
            with open(f'{output_file_dir}/error_compile_gcc.txt', '+a') as fw:
                fw.write(f"error g++, {str(cfg)}, {compile_cpp_cmd}\n")
            return {'returncode': -1, 'run_time': 0.0}

        # 将编译后的test可执行文件移动到compiled_exe目录
        cp_cmd = f'cp ./test ./compiled_exe/{algo_name}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_{cfg["NUMA"]}_{cfg["numSSG"]}'
        print(cp_cmd)
        self.call_program(cp_cmd)
        return result2

    def parse_running_time(self, log_file_name='test.out'):
        """Returns the elapsed time only, from the HPL output file"""

        min_time = 10000

        with open(log_file_name) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        i = 0;
        for line in content:
            if line.find("elapsed time") != -1:
                next_line = content[i+1]
                time_str = next_line.strip()
                time = float(time_str)
                if time < min_time:
                    min_time = time
            i = i+1;

        return min_time

    def run_precompiled(self, cfg, compile_result, graph, algo):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """
        graph_file_path = graph_file_dir + graph + '/' + graph + '.el'
        if compile_result['returncode'] != 0:
            print(str(compile_result))

        assert compile_result['returncode'] == 0
        try:
            # run_result = self.call_program('./test ../test/graphs/socLive_gapbs.sg > test.out')
            # run_result = self.call_program('./test ../test/graphs/4.sg > test.out')
            if not self.use_NUMA:
                if not self.enable_parallel_tuning:
                    # don't use numactl when running serial
                    run_cmd = './test ' + graph_file_path + ' > test.out'

                else:
                    # use numactl when running parallel
                    run_cmd = 'numactl -i all ./test ' + graph_file_path + ' > test.out'
            else:
                run_cmd = 'OMP_PLACES=sockets ./test ' + graph_file_path + ' > test.out'

            print("run_cmd: " + run_cmd)

            # default value -1 for memory_limit translates into None (no memory upper limit)
            # setting memory limit does not quite work yet
            # process_memory_limit = None
            # if self.memory_limit != -1:
            #     process_memory_limit = self.memory_limit
            # print ("memory limit: " + str(process_memory_limit))
            run_result = self.call_program(run_cmd)
        finally:
            # self.call_program('rm test')
            # self.call_program('rm test.cpp')
            pass

        # self.call_program('rm test.out')

        if run_result['returncode'] != 0:
            print('running error, return code :' + str(run_result['returncode']))
            # 打开运行输出源文件test.out并读取内容
            with open(f'{GraphitPath}/autotune/test.out', 'r') as fr:
                content = fr.read()

            with open(f'{output_file_dir}/error_run.txt', 'a') as err_file:
                err_file.write(f'{str(cfg)}, {algo}, {graph}, {run_cmd}\n')
                err_file.write(content)
                err_file.write('\n\n')

            self.call_program('rm test.out')
            return None
        else:
            val = self.parse_running_time()
            print(f'running over, running time: {run_result["run_time"]}, round_time: {val}')
            output_file_name = output_file_dir + algo + '_output.txt'
            with open(output_file_name, 'a') as file:
                file.write(f'{graph} {cfg["direction"]}, {cfg["parallelization"]}, {cfg["DenseVertexSet"]}, {cfg["NUMA"]}, {cfg["numSSG"]}, {run_result["run_time"]}, {val}\n')
            self.call_program('rm test.out')
            return run_result['run_time'], val

    def compile_and_run(self, cfg, graph, algo_file):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        self.reset_flag()   # 重置一些标志位
        
        if cfg['NUMA'] == 'false':
            self.enable_NUMA_tuning = False
        # if cfg['parallelization'] == 'serial':
        #     self.enable_parallel_tuning = False

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
        compile_result = self.compile(cfg, algo_file)
        if compile_result["returncode"] != 0:
            return None
        
        print (" compile_result over. ")
        return self.run_precompiled(cfg, compile_result, graph, algo_file)


    def compile_and_run_v2(self, cfg):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        self.reset_flag()   # 重置一些标志位
        
        if cfg['NUMA'] == 'false':
            self.enable_NUMA_tuning = False
        # if cfg['parallelization'] == 'serial':
        #     self.enable_parallel_tuning = False

        # only use NUMA when we are tuning parallel and NUMA schedules
        # if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] == 'static-parallel':
        if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] != 'false':
            if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull' or cfg['direction'] == 'DensePull-SparsePush':
                if int(cfg['numSSG']) > 1:
                    self.use_NUMA = True

        # if cfg['bucket_update_strategy'] == "eager_priority_update" or cfg['bucket_update_strategy'] == "eager_priority_update_with_merge":
        #     self.use_eager_update = True
        
        # converts the configuration into a schedule
        t0 = time.time()
        returncode = self.write_cfg_to_schedule(cfg)
        t1 = time.time()
        
        # write_cfg_to_schedule error return None
        if returncode == -1:
            return None
        
        print (" write_cfg_to_schedule over. ")

        for algo_file_name in algo_list:
            output_file_name = output_file_dir + algo_file_name + '_output.txt'
            algo_file_path = algo_file_dir+algo_file_name

            # this pases in the id 0 for the configuration
            # 先读取compiled_exe目录下的可执行文件，如果有则直接运行，没有则编译
            exe_file_path = f'./compiled_exe/{algo_file_name}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_{cfg["NUMA"]}_{cfg["numSSG"]}'
            if os.path.exists(exe_file_path):
                self.call_program('rm test')
                self.call_program('rm test.cpp')
                self.call_program(f'cp {exe_file_path} ./test')
                compile_result = {'returncode': 0, 'run_time': 0.0}
            else:
                t2 = time.time()
                compile_result = self.compile(cfg, algo_file_path, algo_file_name)
                t3 = time.time()
                with open(f'{output_file_dir}/compile_result.txt', '+a') as fw:
                    if compile_result["returncode"] == 0:
                        fw.write(f'compile_passed, {exe_file_path}, write_cfg_to_sche_time: {t1-t0}, compile time: {t3-t2}\n')
                    else:
                        fw.write(f'compile_unpass, {exe_file_path}, write_cfg_to_sche_time: {t1-t0}, compile time: {t3-t2}\n')

                if compile_result["returncode"] != 0:
                    return None
                print (" compile_result over, compile time: " + str(t3-t2))

            for graph in graph_list:
                print(f"\n>>>>>>>>>>> algo: {algo_file_name}  graph: {graph} <<<<<<<<<<<<<<")
                result_time = self.run_precompiled(cfg, compile_result, graph, algo_file_name)

        

if __name__ == '__main__':

    # file = open(output_file_name, 'w')
    data_collector = GraphItDataCreator()
    cfg =  {
        'direction': 'SparsePush',
        'parallelization': 'serial',
        'DenseVertexSet': 'boolean-array',
        'NUMA': 'false',
        'numSSG': 0
    }
    
    choice_num = 1
    
    # for i in range(0, max_num_segments, 2):
    for i in [0, 5, 10, 15, 20]:
        cfg['numSSG'] = i
        
        # for NUMA_option in config_NUMA:     # numa 编译有问题
        #     cfg['NUMA'] = NUMA_option
            
        if cfg['NUMA'] != 'false':      # numa 编译有问题
            continue
        
        for direction_option in config_direction:
            cfg['direction'] = direction_option

            for parallelization_option in config_parallelization:
                cfg['parallelization'] = parallelization_option

                for DenseVertexSet_option in config_DenseVertexSet:
                    cfg['DenseVertexSet'] = DenseVertexSet_option

                    # for bucket_update_strategy_option in config_bucket_update_strategy:
                    #     cfg['bucket_update_strategy'] = bucket_update_strategy_option
                    
                    print(f"==================================== choiceNum {choice_num} ========================================")
                    choice_num += 1
                    # cfg = {
                    #     'direction': 'SparsePush-DensePull',
                    #     'parallelization': 'serial',
                    #     'DenseVertexSet': 'bitvector',
                    #     'NUMA': 'false',
                    #     'numSSG': 5
                    # }
                    
                    passed = data_collector.schedule_fliter(cfg)
                    if passed: # 如果能够通过过滤，则调度组合有效
                        print(f"schedule pass: {cfg}")

                        data_collector.compile_and_run_v2(cfg)
                                        
                    else:   # data_collector.schedule_fliter(cfg) 不通过
                        print(f"schedule unsupported: {cfg}")
                    print(f"==================================== choiceNum over ========================================\n")


    # 下面是用之前的compile_and_run()函数，现在用的是上面的compile_and_run_v2()函数
    # # file = open(output_file_name, 'w')
    # data_collector = GraphItDataCreator()
    # cfg =  {
    #     'direction': 'SparsePush',
    #     'parallelization': 'serial',
    #     'DenseVertexSet': 'boolean-array',
    #     'NUMA': 'false',
    #     'numSSG': 0
    # }
    
    # choice_num = 1
    
    # start = False
    # # for i in range(0, max_num_segments, 2):
    # for i in [0, 5, 10, 15, 20]:
    #     cfg['numSSG'] = i
        
    #     for NUMA_option in config_NUMA:     # numa 编译有问题
    #         cfg['NUMA'] = NUMA_option
                            
    #         if cfg['numSSG'] == 5 and cfg['NUMA'] == 'static-parallel':     # 这之前都采集过了
    #             start = True
    #         if not start:
    #             continue
            
    #         if cfg['NUMA'] != 'false':      # numa 编译有问题
    #             continue
            
    #         for direction_option in config_direction:
    #             cfg['direction'] = direction_option

    #             for parallelization_option in config_parallelization:
    #                 cfg['parallelization'] = parallelization_option

    #                 for DenseVertexSet_option in config_DenseVertexSet:
    #                     cfg['DenseVertexSet'] = DenseVertexSet_option

    #                     # for bucket_update_strategy_option in config_bucket_update_strategy:
    #                     #     cfg['bucket_update_strategy'] = bucket_update_strategy_option
                        
    #                     print(f"==================================== choiceNum {choice_num} ========================================")
    #                     choice_num += 1
                        
    #                     passed = data_collector.schedule_fliter(cfg)
    #                     if passed: # 如果能够通过过滤，则调度组合有效
    #                         print(f"schedule pass: {cfg}")
                            
    #                         for algo_file_name in algo_list:
    #                             output_file_name = output_file_dir + algo_file_name + '_output.txt'
                                
    #                             for graph in graph_list:
    #                                 print(f"\n>>>>>>>>>>> algo: {algo_file_name}  graph: {graph} <<<<<<<<<<<<<<")

    #                                 algo_file_path = algo_file_dir+algo_file_name
    #                                 graph_file_path = graph_file_dir + graph + '/' + graph + '.el'
                                    
    #                                 result_time = data_collector.compile_and_run(cfg, graph_file_path, algo_file_path)

    #                                 if result_time == None:
    #                                     with open('/home/zhuhr/AutoGraph/graphit/autotune/costmodel_dataset/error_options/error_options.txt', 'a') as err_file:
    #                                        err_file.write(f'{cfg} {algo_file_name} {graph}\n')
    #                                     continue
    #                                 else:
    #                                     with open(output_file_name, 'a') as file:
    #                                         output_sche = f'{graph} '
    #                                         for key, value in cfg.items():
    #                                             output_sche += f"{value}, "
    #                                         file.write(
    #                                             output_sche+str(result_time)+'\n')
                                            
    #                     else:   # data_collector.schedule_fliter(cfg) 不通过
    #                         print(f"schedule unsupported: {cfg}")
    #                     print(f"==================================== choiceNum over ========================================\n")
