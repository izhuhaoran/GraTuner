#!/usr/bin/env python                                                           
#
# Autotune schedules for PageRank in the GraphIt language
#                                                                               

# import adddeps  # fix sys.path
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from sys import exit
import argparse
import time
import os

all_time = 0.0
write_cfg_time = 0.0
compile_time = 0.0
run_time = 0.0
run_time_once = 0.0
exec_time = 0.0
exec_time_once = 0.0

max_time = 10000

GraphitPath = f'/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit'
algo_file_dir = f"{GraphitPath}/autotune/benchmarks/"

runtimes_data_dir = f'/home/zhuhaoran/AutoGraph/GraTuner/dataset/dataset_YiTian/finetune/train'

output_dir = f"{GraphitPath}/autotune/GraTuner_tests_exectime_YiTian"


time_0 = time.time()

py_graphitc_file = "../build/bin/graphitc.py"
serial_compiler = "g++-7"

#if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++-7"


def read_runtime_dict() -> dict[str, tuple[float, float]]:
    runtime_dict = {}
    for algo in ["bfs", "pagerank", "sssp", 'cc']:
        file_path = f"{runtimes_data_dir}/{algo}.csv"
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                
                # 提取数据
                graph_name = parts[0]
                configs = parts[1:4] + [parts[5]]  # 跳过false列(parts[4])

                _runtime = float(parts[-2])
                _exectime = float(parts[-1])

                # 构建key
                key = algo +'_' + graph_name + '_' + '_'.join(configs)
                
                # 存储到dict
                runtime_dict[key] = ( _runtime, _exectime) 
    return runtime_dict

runtimes_dict = read_runtime_dict()


class GraphItTuner(MeasurementInterface):
    new_schedule_file_name = ''
    # a flag for testing if NUMA-aware schedule is specified
    use_NUMA = False
    use_eager_update = False
    
    # this flag is for testing on machine without NUMA library support
    # this would simply not tune NUMA-aware schedules
    enable_NUMA_tuning = False
    
    # this would simply not tune parallelization related schedules 
    # for machines without CILK or openmp support 
    enable_parallel_tuning = True
    
    enable_denseVertexSet_tuning = True
    
    cfg_to_schedule_pass = True


    def manipulator(self):
        """                                                                          
        Define the search space by creating a                                        
        ConfigurationManipulator                                                     
        """

        # set the global flags needed for printing schedules
        if self.args.enable_NUMA_tuning == 0:
            self.enable_NUMA_tuning = False
        if self.args.enable_parallel_tuning == 0:
            self.enable_parallel_tuning = False
        if self.args.enable_denseVertexSet_tuning == 0:
            self.enable_denseVertexSet_tuning = False

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter('direction', ['SparsePush', 'DensePull', 'DensePull-SparsePush', 'SparsePush-DensePull', 'DensePush-SparsePush']))

        manipulator.add_parameter(EnumParameter('parallelization',['serial', 'dynamic-vertex-parallel', 'static-vertex-parallel', 'edge-aware-dynamic-vertex-parallel'])) 

        manipulator.add_parameter(EnumParameter('numSSG', [0, 5, 10, 15, 20]))

        manipulator.add_parameter(EnumParameter('DenseVertexSet', ['boolean-array', 'bitvector']))

        # if self.enable_NUMA_tuning:
        #     manipulator.add_parameter(EnumParameter('NUMA',['serial','static-parallel']))

        # manipulator.add_parameter(
        #     EnumParameter('bucket_update_strategy', 
        #                   ['eager_priority_update','eager_priority_update_with_merge', 'lazy_priority_update']))

        return manipulator

    def schedule_fliter(self, cfg) -> bool:

        fliter_pass = True
        
        NUMA_enable = False   # 使用numa的要求（计算方向和图划分）是否满足
        NUMA_use = False     # cfg是否使用numa
        
        edge_aware_par_use = False  # 是否使用边感知并行
        edge_aware_par_enable = False # 边感知并行条件是否达到
        
        eager_priority_update = False
        
        bitvector_enble = False     # bitvector使用条件是否达到
        bitvector_use = False       # 是否使用bitvector
        
        SSG_enble = False
        SSG_use = False
        
        if 'bucket_update_strategy' in cfg and cfg['bucket_update_strategy'] == 'eager_priority_update':
            eager_priority_update = True
        
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
        
        if 'NUMA' in cfg:
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

        if eager_priority_update:
            if cfg['direction'] != 'SparsePush':
                fliter_pass = False
        
        return fliter_pass
    
    # configures parallelization commands
    def write_par_schedule(self, cfg, new_schedule, direction):
        use_evp = False

        if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
            use_evp = True

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
        if self.args.default_schedule_file != "":
            f = open(self.args.default_schedule_file,'r')
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

    def compile(self, cfg,  id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """

        #compile the schedule file along with the original algorithm file
        algo_file_path = algo_file_dir + self.args.algo_file
        compile_graphit_cmd = 'python ' + py_graphitc_file +  ' -a  {algo_file} -f {schedule_file} -i ../include/ -l ../build/lib/libgraphitlib.a  -o test.cpp'.format(algo_file=algo_file_path, schedule_file=self.new_schedule_file_name) 

        if not self.use_NUMA:
            if not self.enable_parallel_tuning:
                # if parallel icpc compiler is not needed (only tuning serial schedules)
                compile_cpp_cmd = serial_compiler + ' -std=gnu++1y  -I ../src/runtime_lib/ -O3  test.cpp -o test'
            else:
                # if parallel icpc compiler is supported and needed
                compile_cpp_cmd = par_compiler + ' -std=gnu++1y -DCILK -fcilkplus  -I ../src/runtime_lib/ -O3  test.cpp -o test'
        else:
            #add the additional flags for NUMA
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -lnuma -DNUMA -fopenmp -I ../src/runtime_lib/ -O3  test.cpp -o test'

        if self.use_eager_update:
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -fopenmp -I ../src/runtime_lib/ -O3  test.cpp -o test'
        

        print(compile_graphit_cmd)
        print(compile_cpp_cmd)
        try:
            self.call_program(compile_graphit_cmd)
        except:
            print ("fail to compile .gt file")
        
        result2 = self.call_program(compile_cpp_cmd)
        if result2['returncode'] != 0:
            print (str(result2))
            return {'returncode': -1, 'run_time': 0.0, 'stderr': 'compile error for cpp to exe'}

        # 将编译后的test可执行文件移动到compiled_exe目录
        cp_cmd = f'cp ./test ./compiled_exe/{self.args.algo_file}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_false_{cfg["numSSG"]}'
        print(cp_cmd)
        self.call_program(cp_cmd)
        return result2

    def parse_running_time(self, log_file_name='test.out'):
        """Returns the elapsed time only, from the HPL output file"""

        min_time = max_time

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

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """
        # start_t = time.time()
        cfg = desired_result.configuration.data
        
        if compile_result['returncode'] != 0:
            print (str(compile_result))

        assert compile_result['returncode'] == 0
        try:
            if not self.use_NUMA:
                if not self.enable_parallel_tuning:
                    # don't use numactl when running serial
                    run_cmd = './test ' + self.args.graph +  '  > test.out'

                else:
                    # use numactl when running parallel
                    run_cmd = 'numactl -i all ./test ' + self.args.graph +  '  > test.out'
            else:
                run_cmd = 'OMP_PLACES=sockets ./test ' + self.args.graph + '  > test.out'

            print ("run_cmd: " + run_cmd)

            # default value -1 for memory_limit translates into None (no memory upper limit)
            # setting memory limit does not quite work yet
            process_memory_limit = None
            if self.args.memory_limit != -1:
                process_memory_limit = self.args.memory_limit
            # print ("memory limit: " + str(process_memory_limit))
            run_result = self.call_program(run_cmd, limit=self.args.runtime_limit, memory_limit=process_memory_limit)  
        finally:
            self.call_program('rm test')
            self.call_program('rm test.cpp')

        if run_result['timeout'] == True or run_result['returncode'] != 0:
            # runtime = self.args.runtime_limit
            runtime = run_result['time']
            val = self.args.runtime_limit
            with open('test.out', 'r') as f:
                print(f.read())
        else:
            val = self.parse_running_time()
            runtime = run_result['time']

        self.call_program('rm test.out')       
    
        # end_t = time.time()
        print ("run result: " + str(run_result))
        print ("running time: " + str(val))

        runtime_key = f'{self.args.algo_file[0:-3]}_{self.args.graph_name}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_{cfg["numSSG"]}'
        if runtime_key in runtimes_dict:
            runtime, val = runtimes_dict[runtime_key]
            run_result = {'timeout': False, 'time': runtime, 'returncode': 0}
        else:
            with open(f'{output_dir}/output_autotune/no_runtime_key.txt', '+a') as fw:
                fw.write(f'{runtime_key}\n')
            runtimes_dict[runtime_key] = (runtime, val)

        global exec_time_once, run_time_once
        exec_time_once = val
        run_time_once = runtime

        if run_result['timeout'] == True:
            print ("Timed out after " + str(self.args.runtime_limit) + " seconds")
            return opentuner.resultsdb.models.Result(time=val)
        elif run_result['returncode'] != 0:
            if self.args.killed_process_report_runtime_limit == 1 and run_result['stderr'] == 'Killed\n':
                print ("process killed " + str(run_result))
                return opentuner.resultsdb.models.Result(time=self.args.runtime_limit)
            else:
                print (str(run_result))
                exit()
        else:
            return opentuner.resultsdb.models.Result(time=val)


    def run_precompiled_v2(self, desired_result, input, limit, compile_result, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """
        # start_t = time.time()
        cfg = desired_result.configuration.data
        
        if compile_result['returncode'] != 0:
            print (str(compile_result))

        assert compile_result['returncode'] == 0

        runtime_key = f'{self.args.algo_file[0:-3]}_{self.args.graph_name}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_{cfg["numSSG"]}'
        if runtime_key in runtimes_dict:
            runtime, val = runtimes_dict[runtime_key]
            run_result = {'timeout': False, 'time': run_time, 'returncode': 0}
        else:
            with open(f'{output_dir}/output_autotune/no_runtime_key.txt', '+a') as fw:
                fw.write(f'{runtime_key}\n')
            
            try:    
                if not self.use_NUMA:
                    if not self.enable_parallel_tuning:
                        # don't use numactl when running serial
                        run_cmd = './test ' + self.args.graph +  '  > test.out'

                    else:
                        # use numactl when running parallel
                        run_cmd = 'numactl -i all ./test ' + self.args.graph +  '  > test.out'
                else:
                    run_cmd = 'OMP_PLACES=sockets ./test ' + self.args.graph + '  > test.out'

                print ("run_cmd: " + run_cmd)

                # default value -1 for memory_limit translates into None (no memory upper limit)
                # setting memory limit does not quite work yet
                process_memory_limit = None
                if self.args.memory_limit != -1:
                    process_memory_limit = self.args.memory_limit
                # print ("memory limit: " + str(process_memory_limit))
                run_result = self.call_program(run_cmd, limit=self.args.runtime_limit, memory_limit=process_memory_limit)  
            finally:
                self.call_program('rm test')
                self.call_program('rm test.cpp')

            if run_result['timeout'] == True or run_result['returncode'] != 0:
                # runtime = self.args.runtime_limit
                runtime = run_result['time']
                val = self.args.runtime_limit
                with open('test.out', 'r') as f:
                    print(f.read())
            else:
                val = self.parse_running_time()
                runtime = run_result['time']
                runtimes_dict[runtime_key] = (runtime, val)

            self.call_program('rm test.out')       

            # end_t = time.time()
            print ("run result: " + str(run_result))
            print ("running time: " + str(val))

        global exec_time_once, run_time_once
        run_time_once = runtime
        exec_time_once = val

        if run_result['timeout'] == True:
            print ("Timed out after " + str(self.args.runtime_limit) + " seconds")
            return opentuner.resultsdb.models.Result(time=val)
        elif run_result['returncode'] != 0:
            if self.args.killed_process_report_runtime_limit == 1 and run_result['stderr'] == 'Killed\n':
                print ("process killed " + str(run_result))
                return opentuner.resultsdb.models.Result(time=self.args.runtime_limit)
            else:
                print (f'unexpected error: {run_result}, tune exit')
                exit()
        else:
            return opentuner.resultsdb.models.Result(time=val)


    def compile_and_run(self, desired_result, input, limit):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        print("input graph: " + self.args.graph)

        cfg = desired_result.configuration.data
        cfg['NUMA'] = 'false'

        self.use_NUMA = False;
        # only use NUMA when we are tuning parallel and NUMA schedules
        if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] == 'static-parallel':
            if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull':
                if int(cfg['numSSG']) > 1:
                    self.use_NUMA = True;

        # if cfg['bucket_update_strategy'] == "eager_priority_update" or cfg['bucket_update_strategy'] == "eager_priority_update_with_merge":
        #     self.use_eager_update = True

        if not self.schedule_fliter(cfg):
            with open(f'{output_dir}/output_autotune/cfg.txt', '+a') as cfg_file:
                cfg_file.write(f'unpass, {str(cfg)}\n')

            with open(f'{output_dir}/output_autotune/time_log_iter.txt', '+a') as fw_time:
                fw_time.write(f"unpass, write_cfg_time: 0.00000000000000, compile_time: 0.00000000000000, run_time: 0.00000000000000, exec_time: 0.00000000000000\n")

            return opentuner.resultsdb.models.Result(time=max_time)

        exe_file_path = f'./compiled_exe/{self.args.algo_file}_{cfg["direction"]}_{cfg["parallelization"]}_{cfg["DenseVertexSet"]}_false_{cfg["numSSG"]}'
        if os.path.exists(exe_file_path):
            self.call_program('rm test')
            self.call_program('rm test.cpp')
            self.call_program(f'cp {exe_file_path} ./test')
            compile_result = {'returncode': 0, 'run_time': 0.0}
            t1 = time.time()
            t2 = time.time()
            t3 = time.time()
        else:
            t1 = time.time()
            # converts the configuration into a schedule
            self.write_cfg_to_schedule(cfg)

            t2 = time.time()
            # this pases in the id 0 for the configuration
            compile_result = self.compile(cfg, 0)
            # print "compile_result: " + str(compile_result)
            t3 = time.time()
            print(f'compile_time: {t3 - t2}')
        
        # result_ = self.run_precompiled(desired_result, input, limit, compile_result, 0)
        result_ = self.run_precompiled_v2(desired_result, input, limit, compile_result, 0)
        t4 = time.time()

        with open(f'{output_dir}/output_autotune/cfg.txt', '+a') as cfg_file:
                cfg_file.write(f'passed, {str(cfg)}\n')

        with open(f'{output_dir}/output_autotune/time_log_iter.txt', '+a') as fw_time:
            fw_time.write(f"passed, write_cfg_time: {t2 - t1}, compile_time: {t3 - t2}, run_time: {run_time_once}, exec_time: {exec_time_once}\n")

        # print('one case over.\n\n')

        return result_


    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print ('Final Configuration:', configuration.data)
        self.manipulator().save_to_file(configuration.data,'final_config.json', mode='+a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('--graph', type=str, default="/home/zhuhaoran/AutoGraph/graphs/github/github.el",
                    help='the graph to tune on')
    parser.add_argument('--graph_name', type=str, default="github")
    parser.add_argument('--enable_NUMA_tuning', type=int, default=0, help='enable tuning NUMA-aware schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--enable_parallel_tuning', type=int, default=1, help='enable tuning paralleliation schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--enable_denseVertexSet_tuning', type=int, default=1, help='enable tuning denseVertexSet schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--algo_file', type=str, default="pagerank.gt", help='input algorithm file')
    parser.add_argument('--default_schedule_file', type=str, required=False, default="", help='default schedule file')
    parser.add_argument('--runtime_limit', type=float, default=1000, help='a limit on the running time of each program')
    parser.add_argument('--max_num_segments', type=int, default=24, help='maximum number of segments to try for cache and NUMA optimizations')
    parser.add_argument('--max_delta', type=int, default=800000, help='maximum delta used for priority coarsening')
    parser.add_argument('--memory_limit', type=int, default=-1,help='set memory limit on unix based systems [does not quite work yet]')    
    parser.add_argument('--killed_process_report_runtime_limit', type=int, default=0, help='reports runtime_limit when a process is killed by the shell. 0 for disable (default), 1 for enable')
    args = parser.parse_args()
    # pass the argumetns into the tuner

    time_0 = time.time()
    GraphItTuner.main(args)
    t2 = time.time()
    
    