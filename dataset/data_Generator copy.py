import random
import itertools
import time
import sys
import os
import math

import subprocess
import shlex

import numpy as np
from sys import exit


GraphitPath = '/home/zhuhaoran/AutoGraph/AutoGraph/graphit'

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

config_direction = ['SparsePush', 'DensePush', 'DensePull', 'DensePull-SparsePush', 'DensePush-SparsePush']
config_parallelization = ['serial', 'dynamic-vertex-parallel', 'static-vertex-parallel', 'edge-aware-dynamic-vertex-parallel']
config_DenseVertexSet = ['bitvector', 'boolean-array']
# config_bucket_update_strategy = ['eager_priority_update', 'eager_priority_update_with_merge']
config_NUMA = ['false', 'static-parallel', 'dynamic-parallel']
config_numSSG = ['fixed-vertex-count']

algo_file_dir = f"{GraphitPath}/autotune/benchmarks/"
algo_list = ['cc.gt', 'pagerank.gt', 'sssp.gt', 'bfs.gt']

graph_file_dir = f"{GraphitPath}/autotune/graphs/"
graph_list = ['4.sg']


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

    enable_denseVertexSet_tuning = True

    # command-line arguments in graphit_autotuner(all defalut)
    default_schedule_file = ""

    latest_schedule = ''

    def call_program(self, command):
        start_time = time.time()

        cmd_args = command.split(' ')
        print(cmd_args)
        process = subprocess.run(
            cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        end_time = time.time()
        run_time = end_time - start_time

        return {'returncode': process.returncode, 'run_time': run_time}

    # configures parallelization commands
    def write_par_schedule(self, cfg, new_schedule, direction):
        use_evp = False

        if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
            use_evp = True

        if use_evp == False or self.use_NUMA == True:
            # if don't use edge-aware parallel (vertex-parallel)
            # edge-parallel don't work with NUMA (use vertex-parallel when NUMA is enabled)
            if cfg['parallelization'] == 'serial':
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"serial\");"
            else:
                # if NUMA is used, then we only use dynamic-vertex-parallel as edge-aware-vertex-parallel do not support NUMA yet
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
        elif use_evp == True and self.use_NUMA == False:
            # use_evp is True
            if direction == "DensePull":
                # edge-aware-dynamic-vertex-parallel is only supported for the DensePull direction
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024, \"DensePull\");"

            elif direction == "SparsePush-DensePull":
                # For now, only the DensePull direction uses edge-aware-vertex-parallel
                # the SparsePush should still just use the vertex-parallel methodx
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024,  \"DensePull\");"

                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\",1024,  \"SparsePush\");"

            else:
                # use_evp for SparsePush, DensePush-SparsePush should not make a difference
                new_schedule = new_schedule + \
                    "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
        else:
            print("Error in writing parallel schedule")
            return None
        return new_schedule

    def write_numSSG_schedule(self, numSSG, new_schedule, direction):
        # No need to insert for a single SSG
        if numSSG == 0 or numSSG == 1:
            return new_schedule
        # configuring cache optimization for DensePull direction
        if direction == "DensePull" or direction == "SparsePush-DensePull":
            new_schedule = new_schedule + \
                "\n    program->configApplyNumSSG(\"s1\", \"fixed-vertex-count\", " + str(
                    numSSG) + ", \"DensePull\");"
        return new_schedule

    def write_bucket_update_schedule(self, bucket_update_strategy, new_schedule):
        new_schedules = new_schedule + \
            "\n    program->configApplyPriorityUpdate(\"s1\", \"" + \
            bucket_update_strategy + "\" );"
        return new_schedules

    def write_NUMA_schedule(self,  new_schedule, direction):
        # configuring NUMA optimization for DensePull direction
        if self.use_NUMA:
            if direction == "DensePull" or direction == "SparsePush-DensePull":
                new_schedule = new_schedule + \
                    "\n    program->configApplyNUMA(\"s1\", \"static-parallel\" , \"DensePull\");"
        return new_schedule

    def write_denseVertexSet_schedule(self, enable_pull_bitvector, new_schedule, direction):
        # for now, we only use this for the src vertexset in the DensePull direciton
        if direction == "DensePull" or direction == "SparsePush-DensePull":
            if enable_pull_bitvector:
                new_schedule = new_schedule + \
                    "\n    program->configApplyDenseVertexSet(\"s1\",\"bitvector\", \"src-vertexset\", \"DensePull\");"
        return new_schedule

    def write_cfg_to_schedule(self, cfg):
        # write into a schedule file the configuration
        direction = cfg['direction']
        numSSG = cfg['numSSG']
        #   delta = cfg['delta']
        bucket_update_strategy = cfg['bucket_update_strategy']

        new_schedule = ""
        direction_schedule_str = "\n    program->configApplyDirection(\"s1\", \"$direction\");"
        if self.default_schedule_file != "":
            f = open(self.default_schedule_file, 'r')
            default_schedule_str = f.read()
            f.close()
        else:
            default_schedule_str = "schedule: "

        # eager only works with SparsePush for now
        if bucket_update_strategy == 'eager_priority_update':
            new_schedule = default_schedule_str + \
                direction_schedule_str.replace('$direction', 'SparsePush')
        else:
            new_schedule = default_schedule_str + \
                direction_schedule_str.replace('$direction', cfg['direction'])

        new_schedule = self.write_par_schedule(cfg, new_schedule, direction)
        new_schedule = self.write_numSSG_schedule(
            numSSG, new_schedule, direction)
        new_schedule = self.write_NUMA_schedule(new_schedule, direction)
        #   new_schedule = self.write_delta_schedule(delta, new_schedule)
        new_schedule = self.write_bucket_update_schedule(
            bucket_update_strategy, new_schedule)

        use_bitvector = False
        if cfg['DenseVertexSet'] == 'bitvector':
            use_bitvector = True
        new_schedule = self.write_denseVertexSet_schedule(
            use_bitvector, new_schedule, direction)
        
        if new_schedule == self.latest_schedule:
            print(f"===========================>  same schedule: \n{new_schedule}")
            return -1
        
        self.latest_schedule = new_schedule
        
        print(cfg)
        print(new_schedule)

        self.new_schedule_file_name = 'schedule_0'
        print(self.new_schedule_file_name)

        f1 = open(self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()
        
        return 1
    
    def write_cfg_to_schedule_new(self, cfg):
        new_schedule = ''
        
        new_schedule = new_schedule + \
            "\n    program->configApplyDenseVertexSet(\"s1\",\"bitvector\", \"src-vertexset\", \"DensePull\");"
        pass
        

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
                compile_cpp_cmd = par_compiler + \
                    ' -std=gnu++1y -DCILK -fcilkplus -I ../src/runtime_lib/ -O3 test.cpp -o test'
        else:
            # add the additional flags for NUMA
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -lnuma -DNUMA -fopenmp -I ../src/runtime_lib/ -O3 test.cpp -o test'

        if self.use_eager_update:
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -fopenmp -I ../src/runtime_lib/ -O3 test.cpp -o test'

        print(compile_graphit_cmd)
        print(compile_cpp_cmd)

        self.call_program(compile_graphit_cmd)
        #   try:
        #       self.call_program(compile_graphit_cmd)
        #   except:
        #       print ("fail to compile .gt file")
        #       return None
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

        self.call_program('rm test.out')
        print("running time: " + str(run_result['run_time']))

        if run_result['returncode'] != 0:
            return None
        else:
            return run_result['run_time']

    def compile_and_run(self, cfg, graph, algo_file):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        print("input graph: " + graph)

        self.use_NUMA = False
        # only use NUMA when we are tuning parallel and NUMA schedules
        if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] == 'static-parallel':
            if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull':
                if int(cfg['numSSG']) > 1:
                    self.use_NUMA = True

        if cfg['bucket_update_strategy'] == "eager_priority_update" or cfg['bucket_update_strategy'] == "eager_priority_update_with_merge":
            self.use_eager_update = True
        # converts the configuration into a schedule
        returncode = self.write_cfg_to_schedule(cfg)
        
        # return None
        if returncode == -1:
            return None

        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, algo_file, 0)
        if compile_result["returncode"] != 0:
            return None
        # print "compile_result: " + str(compile_result)
        return self.run_precompiled(cfg, compile_result, graph, 0)


if __name__ == '__main__':
    output_file_name = f"{GraphitPath}/autotune/costmodel_dataset/output.txt"

    # file = open(output_file_name, 'w')
    data_collector = GraphItDataCreator()
    cfg = {}

    for algo_file_name in algo_list:
        for graph in graph_list:

            for direction_option in config_direction:
                cfg['direction'] = direction_option

                for parallelization_option in config_parallelization:
                    cfg['parallelization'] = parallelization_option

                    for DenseVertexSet_option in config_DenseVertexSet:
                        cfg['DenseVertexSet'] = DenseVertexSet_option

                        # for bucket_update_strategy_option in config_bucket_update_strategy:
                        #     cfg['bucket_update_strategy'] = bucket_update_strategy_option

                        for NUMA_option in config_NUMA:
                            cfg['NUMA'] = NUMA_option

                            
                            for i in range(1, max_num_segments+1):
                                cfg['numSSG'] = i

                                algo_file_path = algo_file_dir+algo_file_name
                                graph_file_path = graph_file_dir+graph
                                result_time = data_collector.compile_and_run(cfg, graph_file_path, algo_file_path)

                                if result_time == None:
                                    continue
                                else:
                                    with open(output_file_name, 'a') as file:
                                        output_sche = ''
                                        for key, value in cfg.items():
                                            output_sche += f"{value}, "
                                        file.write(
                                            output_sche+str(result_time)+'\n')
