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


    def compile(self, cfg,  id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
        try:
            self.call_program("cp " + self.args.algo_file + " algotorun.gt")
            return self.call_program("bash compile_gpu.sh")
        except:
            print ("fail to compiler .gt file")
            self.call_program("false")


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

    def run_precompiled(self, cfg, compile_result):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """

        if compile_result['returncode'] != 0:
            print (str(compile_result))

        assert compile_result['returncode'] == 0
        try:    
            run_cmd = "./test " + self.args.graph + " " + self.args.start_vertex + " > test.out"
            print ("run_cmd: " + run_cmd)

            # print ("memory limit: " + str(process_memory_limit))
            run_result = self.call_program(run_cmd)  
        finally:
            pass
	
            #self.call_program('rm test')
            #self.call_program('rm test.cpp')

        if run_result['timeout'] == True:
            val = self.args.runtime_limit
        else:
            val = self.parse_running_time();
        
        self.call_program('rm test.out')
        print ("run result: " + str(run_result))
        print ("running time: " + str(val))

        if run_result['returncode'] != 0:
            if self.args.killed_process_report_runtime_limit == 1 and run_result['stderr'] == 'Killed\n' or True:
                print ("process killed " + str(run_result))
                # return opentuner.resultsdb.models.Result(time=self.args.runtime_limit)
            else:
                print (str(run_result))
                exit()
        else:
            # return opentuner.resultsdb.models.Result(time=val)
            pass
        

    def compile_and_run(self, cfg):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        # print ("input graph: " + self.args.graph)

        self.write_cfg_to_schedule(cfg)
        
        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, 0)
        # print "compile_result: " + str(compile_result)
        return self.run_precompiled(cfg, compile_result)


    def test_one(self):
        # TWC_None_0_PULL_ENABLED_FUSED_BITMAP_DISABLED
        cfg = {'kernel_fusion': 'DISABLED', 'LB_0': 'TWC', 'direction_0': 'PULL', 'dedup_0': 'ENABLED', 'frontier_output_0': 'FUSED', 'pull_rep_0': 'BITMAP', 'EB_0': 'None', 'BS_0': 0}
        self.compile_and_run(cfg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default="/home/zhuhaoran/AutoGraph/graphs/youtube-u-growth/youtube-u-growth.el", help='the graph to tune on')
    parser.add_argument('--start_vertex', type=str, default="1", help="Start vertex if applicable")
    parser.add_argument('--num_vertices', type=int, default=1000, help='Supply number of vertices in the graph')

    parser.add_argument('--algo_file', type=str, default='/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/gpu_apps/sssp.gt', help='input algorithm file')
    # parser.add_argument('--final_config', type=str, help='Final config file', default="final_config.json")
    # parser.add_argument('--default_schedule_file', type=str, required=False, default="", help='default schedule file')
    parser.add_argument('--runtime_limit', type=float, default=300, help='a limit on the running time of each program')
    parser.add_argument('--max_delta', type=int, default=800000, help='maximum delta used for priority coarsening')
    parser.add_argument('--memory_limit', type=int, default=-1,help='set memory limit on unix based systems [does not quite work yet]')    
    parser.add_argument('--killed_process_report_runtime_limit', type=int, default=0, help='reports runtime_limit when a process is killed by the shell. 0 for disable (default), 1 for enable')

    parser.add_argument('--kernel_fusion', type=bool, default=True, help='Choose if you want to also tune kernel fusion')
    parser.add_argument('--hybrid_schedule', type=bool, default=False, help='Choose if you want to also explore hybrid schedules')
    parser.add_argument('--edge_only', type=bool, default=False, help='Choose if you want to also enable EDGE_ONLY schedules')
    parser.add_argument('--tune_delta', type=bool, default=False, help='Also tune the delta parameter')
    parser.add_argument('--hybrid_threshold', type=int, default=1000, help='Threshold value on 1000')

    args = parser.parse_args()
    # pass the argumetns into the tuner
    tester = GraphIt_GPU_Tester(args)
    tester.test_one()
    
