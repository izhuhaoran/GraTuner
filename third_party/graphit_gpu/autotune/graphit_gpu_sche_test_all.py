#!/usr/bin/env python                                                           
#
# Autotune schedules for DeltaStepping in the GraphIt language
#                                                                               

# import adddeps  # fix sys.path
import subprocess
import time
from sys import exit
import argparse
from itertools import product
import os


py_graphitc_file = "../build/bin/graphitc.py"
serial_compiler = "g++"
serial_compiler = "g++"

#if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++"

graph_v_nums ={
    'sx-stackoverflow': 2601977, 'dblp-cite': 12590, 'dbpedia-team': 935627, 'dimacs9-E': 3598623, 'douban': 154908,
    'facebook-wosn-wall': 46952, 'github': 177386, 'komarix-imdb': 	871982, 'moreno_blogs': 1224, 'opsahl-usairport': 1574,
    'patentcite': 3774768, 'petster-friendships-dog': 426820, 'roadNet-CA': 1965206, 'subelj_cora': 23166, 'sx-mathoverflow': 24818,
    'youtube-groupmemberships': 124325, 
}

# graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#                'youtube-groupmemberships', 
#               ]

graph_list = ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA']

algo_file_dir = f"/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/gpu_apps/"
# algo_list = ['sssp.gt', 'bfs.gt', 'pagerank.gt', 'cc.gt', 'cf.gt', 'sssp_delta_stepping.gt']
algo_list = ['pagerank.gt', 'sssp.gt', 'bfs.gt', 'cc.gt']
# algo_list = ['cc.gt']

graph_file_dir = f"/home/zhuhaoran/AutoGraph/graphs/"

class GraphIt_GPU_Tester():
    new_schedule_file_name = ''
    # config_dict = {}
    
    def __init__(self, args) -> None:
        self.args = args
    
    def call_program(self, cmd):
        t0 = time.time()
        if type(cmd) in (str, str):
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        else:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t1 = time.time()
        return {'time': (t1 - t0),
                'returncode': p.returncode,
                'stdout': p.stdout,
                'stderr': p.stderr}
    
    def cfg_filter(self, cfg) -> bool:
        passed = True
        if(cfg):
            return True


    def write_cfg_to_schedule(self, cfg):
        #write into a schedule file the configuration

        direction_0 = cfg['direction_0']
        if self.args.tune_delta:
            delta_0 = cfg['delta']
        dedup_0 = cfg['dedup_0']
        frontier_output_0 = cfg['frontier_output_0']
        pull_rep_0 = cfg['pull_rep_0']
        LB_0 = cfg['LB_0']

        new_schedule = "schedule:\n"

        new_schedule += "SimpleGPUSchedule s1;\n";
        if LB_0 == "EDGE_ONLY" and cfg['EB_0'] == "ENABLED":
            new_schedule += "s1.configLoadBalance(EDGE_ONLY, BLOCKED, " + str(int(int(self.args.num_vertices)/cfg['BS_0'])) + ");\n"
            direction_0 = "PUSH"
            
        else:
            new_schedule += "s1.configLoadBalance(" + LB_0 + ");\n"
        new_schedule += "s1.configFrontierCreation(" + frontier_output_0 + ");\n"
        if direction_0 == "PULL":
            new_schedule += "s1.configDirection(PULL, " + pull_rep_0 + ");\n"
        else:
            new_schedule += "s1.configDirection(PUSH);\n"
        if self.args.tune_delta:
            new_schedule += "s1.configDelta(" + str(delta_0) + ");\n"
        new_schedule += "s1.configDeduplication(" + dedup_0 + ");\n"

        if self.args.hybrid_schedule:
            direction_1 = cfg['direction_1']
            if self.args.tune_delta:
                delta_1 = cfg['delta']
            dedup_1 = cfg['dedup_1']
            frontier_output_1 = cfg['frontier_output_1']
            pull_rep_1 = cfg['pull_rep_1']
            LB_1 = cfg['LB_1']

            #threshold = self.args.hybrid_threshold
            threshold = cfg['threshold']
            
            new_schedule += "SimpleGPUSchedule s2;\n";
            new_schedule += "s2.configLoadBalance(" + LB_1 + ");\n"
            new_schedule += "s2.configFrontierCreation(" + frontier_output_1 + ");\n"
            if direction_1 == "PULL":
                new_schedule += "s2.configDirection(PULL, " + pull_rep_1 + ");\n"
            else:
                new_schedule += "s2.configDirection(PUSH);\n"
            if self.args.tune_delta:
                new_schedule += "s2.configDelta(" + str(delta_1) + ");\n"
            new_schedule += "s2.configDeduplication(" + dedup_1 + ");\n"
            
            new_schedule += "HybridGPUSchedule h1(INPUT_VERTEXSET_SIZE, " + str(threshold/1000) + ", s1, s2);\n"
            new_schedule += "program->applyGPUSchedule(\"s0:s1\", h1);\n"

        else:
            new_schedule += "program->applyGPUSchedule(\"s0:s1\", s1);\n"



        if self.args.kernel_fusion:
            kernel_fusion = cfg['kernel_fusion']
            new_schedule += "SimpleGPUSchedule s0;\n"
            new_schedule += "s0.configKernelFusion(" + kernel_fusion + ");\n"
            new_schedule += "program->applyGPUSchedule(\"s0\", s0);\n"

        print (cfg)
        #print (new_schedule)

        self.new_schedule_file_name = 'schedule_0' 
        #print (self.new_schedule_file_name)
        f1 = open (self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()

    def compile(self, cfg):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
        try:
            self.call_program('rm compile.o')
            self.call_program('rm test')
            self.call_program('rm test.cu')
        
            self.call_program("cp " + self.args.algo_file + " algotorun.gt")
            compile_result = self.call_program("bash compile_gpu.sh")
            
            if compile_result['returncode'] != 0:
                print (str(compile_result))
                with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/compiled_exe/error_compile.log', '+a') as fw:
                    fw.write(self.args.algo_file + '\n')
                    fw.write(str(cfg)+'\n')
                    fw.write('return_code: ' + str(compile_result['returncode']) + '\n')
                    fw.write('stderr: ' + compile_result['stderr'] + '\n')
                    fw.write('stdout: ' + compile_result['stdout'] + '\n')
                    fw.write('\n\n') 
            
                with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/compiled_exe/error_compile_file.txt', '+a') as fw:
                    fw.write(f'{self.args.algo}_{cfg["LB_0"]}_{cfg["EB_0"]}_{cfg["BS_0"]}_{cfg["direction_0"]}_{cfg["dedup_0"]}_{cfg["frontier_output_0"]}_{cfg["pull_rep_0"]}_{cfg["kernel_fusion"]}' + '\n')

            else:
                cp_cmd = f'cp ./test /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/compiled_exe/{self.args.algo}_{cfg["LB_0"]}_{cfg["EB_0"]}_{cfg["BS_0"]}_{cfg["direction_0"]}_{cfg["dedup_0"]}_{cfg["frontier_output_0"]}_{cfg["pull_rep_0"]}_{cfg["kernel_fusion"]}'
                print(cp_cmd)
                self.call_program(cp_cmd)
            return compile_result
        except:
            print ("fail to compiler .gt file")
            return {'time': 0.0, 'returncode': 255, 'stdout': "fail to compiler .gt file", 'stderr': "fail to compiler .gt file"}


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
        
        try:    
            run_cmd = "./test " + self.args.graph + " " + self.args.start_vertex + " > test.out"
            print ("run_cmd: " + run_cmd)

            t1 = time.time()
            run_result = self.call_program(run_cmd)  
            t2 = time.time()
            run_time = t2 - t1
        finally:
            # 调到编译之前删除了，这里不用了
            # self.call_program('rm test')
            # self.call_program('rm test.cpp')
            pass
        
        if run_result['returncode'] != 0:
            print ("run error: " + str(run_result))
            with open('test.out') as f:
                content = f.read()
            with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/costmodel_data/error_run.txt', '+a') as fw:
                fw.write(algo + ' ' + graph + ' ' + run_cmd + '\n')
                fw.write(str(cfg)+'\n')
                fw.write(f'test.out: {content}\n')
                fw.write('return_code: ' + str(run_result['returncode']) + '\n')
                fw.write('stderr: ' + run_result['stderr'] + '\n')
                fw.write('stdout: ' + run_result['stdout'] + '\n')
                fw.write('\n\n') 
            # return
        else:
            val = self.parse_running_time();
            with open(f'/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/costmodel_data/{algo}_runtime.txt', '+a') as fw2:
                output_sche = f'{graph} '
                for key, value in cfg.items():
                    output_sche += f"{value}, "
                fw2.write(
                    output_sche + f'{run_time}, {val}\n')
                
            res = self.call_program('rm test.out')
            assert res['returncode'] == 0
            
            print ("run result: " + str(run_result))
            print ("running time: " + str(val))


    def compile_and_run(self, cfg, graph='', algo=''):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        for algo_ in algo_list:
        # for algo_ in ['bfs.gt']:
            algo = algo_
            self.args.algo_file = algo_file_dir + algo
            self.args.algo = algo
            
            # 此时BS_0无效, 说明不需要edge block 那不同图数据下编译不变
            if cfg['BS_0'] == 0:  
                exe_name = f'/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/compiled_exe/{self.args.algo}_{cfg["LB_0"]}_{cfg["EB_0"]}_{cfg["BS_0"]}_{cfg["direction_0"]}_{cfg["dedup_0"]}_{cfg["frontier_output_0"]}_{cfg["pull_rep_0"]}_{cfg["kernel_fusion"]}'  
                if os.path.exists(exe_name):
                    self.call_program('rm ./test')
                    cp_cmd = f'cp {exe_name} ./test'
                    print(cp_cmd)
                    self.call_program(cp_cmd)
                    compile_result = {'returncode': 0}
                else:
                    # print ("input graph: " + self.args.graph)
                    t0 = time.time()
                    self.write_cfg_to_schedule(cfg)
                    t1 = time.time()
                    compile_result = self.compile(cfg)
                    t2 = time.time()
                    with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/GraTuner_tests/test_iter.txt', '+a') as fw:
                        if compile_result['returncode'] == 0:
                            fw.write(f'compile passed, write schedule time: {t1-t0}, compile time: {t2-t1}\n')
                        else:
                            fw.write(f'compile unpass, write schedule time: {t1-t0}, compile time: {t2-t1}\n')
                
                # print "compile_result: " + str(compile_result)
                if compile_result['returncode'] == 0:
                    for graph_ in graph_list:
                        graph = graph_
                        self.args.graph = graph_file_dir + graph + '/' + graph + '.el'
                        # self.args.num_vertices = graph_v_nums[graph]
                        self.run_precompiled(cfg, compile_result, graph, algo)
            
            # 此时BS_0不为0, 那不同图数据节点数目不同，blocksize也不同，sche就不同，需要一一编译
            else:
                print("error: BS_0 != 0")
                exit(1)
                for graph_ in graph_list:
                    graph = graph_
                    self.args.graph = graph_file_dir + graph + '/' + graph + '.el'
                    self.args.num_vertices = graph_v_nums[graph]
                    # print ("input graph: " + self.args.graph)
                    self.write_cfg_to_schedule(cfg)
                    
                    compile_result = self.compile(cfg)
                    # print "compile_result: " + str(compile_result)
                    if compile_result['returncode'] == 0:
                        self.run_precompiled(cfg, compile_result, graph, algo)
                        
        with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/costmodel_data/cfg_over.txt', '+a') as fw:
            fw.write(str(cfg)+'\n')

    
    def generate_config_combinations(self, config_dict:dict):
        # 获取所有的键名
        keys = config_dict.keys()
        # 生成所有的值组合
        value_combinations = product(*config_dict.values())
        # 为每个值组合生成一个字典
        dict_combinations = [dict(zip(keys, values)) for values in value_combinations]
        print (f'before unique config lens: {len(dict_combinations)}')
        # print(dict_combinations[0])
        for combination in dict_combinations:
            # 应用条件性逻辑
            if combination['LB_0'] != 'EDGE_ONLY':
                combination['EB_0'] = 'None'
                combination['BS_0'] = 0

            if combination['LB_0'] == 'EDGE_ONLY' and combination['EB_0'] == 'ENABLED':
                combination['direction_0'] = 'PUSH'

            if combination['EB_0'] != 'ENABLED':
                combination['BS_0'] = 0

            if combination['direction_0'] != 'PULL':
                combination['pull_rep_0'] = 'None'

            if self.args.hybrid_schedule:
                if combination['direction_1'] != 'PULL':
                    combination['pull_rep_1'] = 'None'

        # 去重
        unique_dict_combinations = [dict(t) for t in {tuple(d.items()) for d in dict_combinations}]
        
        return unique_dict_combinations


    def test_all(self):
        config_dict : dict[str, list] = {}  # 所有可选配置集合
        
        if self.args.tune_delta:
            config_dict['delta'] = list(range(1, self.args.max_delta))

        if self.args.kernel_fusion:
            config_dict['kernel_fusion'] = ['DISABLED', 'ENABLED']

        if self.args.edge_only:
            config_dict['LB_0'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM', 'EDGE_ONLY']
            config_dict['EB_0'] = ['ENABLED', 'DISABLED']
            config_dict['BS_0'] = list(range(1, 21))
            # config_dict['BS_0'] = [5, 10, 15, 20]
        else:
            config_dict['LB_0'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM']
            config_dict['EB_0'] = ['None']
            config_dict['BS_0'] = [0]

        config_dict['direction_0'] = ['PUSH', 'PULL']
        config_dict['dedup_0'] = ['ENABLED', 'DISABLED']
        config_dict['frontier_output_0'] = ['FUSED', 'UNFUSED_BITMAP', 'UNFUSED_BOOLMAP']
        config_dict['pull_rep_0'] = ['BITMAP', 'BOOLMAP']

        if self.args.hybrid_schedule:
            config_dict['LB_1'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM']
            
            config_dict['direction_1'] = ['PUSH', 'PULL']
            config_dict['dedup_1'] = ['ENABLED', 'DISABLED']
            config_dict['frontier_output_1'] = ['FUSED', 'UNFUSED_BITMAP', 'UNFUSED_BOOLMAP']
            config_dict['pull_rep_1'] = ['BITMAP', 'BOOLMAP']
            
            # # We also choose the hybrid schedule threshold here
            # config_dict['threshold'] = list(range(0, 1001))
            # config_dict['threshold'] = [10, 50, 100, 200, 500, 1000, 1500, 2000]
            # config_dict['threshold'] = [10, 50, 100, 200, 500, 1000]
            config_dict['threshold'] = [500, 1000]

    
        # 生成所有配置组合的cfg字典
        config_combinations = self.generate_config_combinations(config_dict)
        print (f'unique config lens: {len(config_combinations)}')
        
        with open('/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/GraTuner_tests/cfg_all.txt', 'w')as fw:
            for cfg in config_combinations:
                fw.write(str(cfg)+'\n')

        for cfg in config_combinations:
                self.compile_and_run(cfg)

    def test_one(self):
        cfg = {'kernel_fusion': 'DISABLED', 'LB_0': 'TWC', 'direction_0': 'PULL', 'dedup_0': 'ENABLED', 'frontier_output_0': 'FUSED', 'pull_rep_0': 'BITMAP', 'EB_0': 'None', 'BS_0': 0}

        self.compile_and_run(cfg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default="", help='the graph to tune on')
    parser.add_argument('--start_vertex', type=str, default="1", help="Start vertex if applicable")
    parser.add_argument('--num_vertices', type=int, help='Supply number of vertices in the graph')

    parser.add_argument('--algo_file', type=str, help='input algorithm file')
    # parser.add_argument('--final_config', type=str, help='Final config file', default="final_config.json")
    # parser.add_argument('--default_schedule_file', type=str, required=False, default="", help='default schedule file')
    # parser.add_argument('--runtime_limit', type=float, default=300, help='a limit on the running time of each program')
    parser.add_argument('--max_delta', type=int, default=800000, help='maximum delta used for priority coarsening')
    # parser.add_argument('--memory_limit', type=int, default=-1,help='set memory limit on unix based systems [does not quite work yet]')    
    # parser.add_argument('--killed_process_report_runtime_limit', type=int, default=0, help='reports runtime_limit when a process is killed by the shell. 0 for disable (default), 1 for enable')

    parser.add_argument('--kernel_fusion', type=bool, default=True, help='Choose if you want to also tune kernel fusion')
    parser.add_argument('--hybrid_schedule', type=bool, default=False, help='Choose if you want to also explore hybrid schedules')
    parser.add_argument('--edge_only', type=bool, default=False, help='Choose if you want to also enable EDGE_ONLY schedules')
    parser.add_argument('--tune_delta', type=bool, default=False, help='Also tune the delta parameter')
    parser.add_argument('--hybrid_threshold', type=int, default=1000, help='Threshold value on 1000')

    args = parser.parse_args()
    # pass the argumetns into the tuner
    tester = GraphIt_GPU_Tester(args)
    tester.test_all()
    
