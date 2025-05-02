import subprocess
import os
import shutil
import sys
import time

GRAPHIT_BUILD_DIRECTORY="/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/build".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu".strip().rstrip("/")
CXX_COMPILER="/usr/bin/c++"

NVCC_COMPILER="/usr/local/cuda/bin/nvcc"


class GraphIt_GPU_Compiler():
    
    def test(self):
        self.scratch_directory = '/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/output'
        self.nvcc_command = NVCC_COMPILER + " -ccbin " + CXX_COMPILER + " "
        self.test_input_directory = GRAPHIT_SOURCE_DIRECTORY + "/test/gpu_tests/test_input"

        self.get_command_output_class_v2(self.nvcc_command + self.test_input_directory + "/obtain_gpu_cc.cu -o " + self.scratch_directory + "/obtain_gpu_cc")
        output = self.get_command_output_class_v2(self.scratch_directory + "/obtain_gpu_cc")[1].split()

        if len(output) != 2:
            print ("Cannot obtain GPU information")
            exit(-1)
        else:
            print(f"GPU information: num_of_sm ({output[1]}), compute_capability ({output[0]})")
    
    def setUpClass(self):
        if NVCC_COMPILER == "CUDA_NVCC_EXECUTABLE-NOTFOUND":
            print ("Cannot find CUDA compiler")
            exit(-1)	
            
        self.build_directory = GRAPHIT_BUILD_DIRECTORY
        # self.scratch_directory = GRAPHIT_BUILD_DIRECTORY + "/scratch"
        self.scratch_directory = "/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/output"
        self.verifier_dirbunectory = self.build_directory + "/bin"	
        if os.path.isdir(self.scratch_directory):
            shutil.rmtree(self.scratch_directory)
        os.mkdir(self.scratch_directory)

        self.nvcc_command = NVCC_COMPILER + " -ccbin " + CXX_COMPILER + " "
        self.test_input_directory = GRAPHIT_SOURCE_DIRECTORY + "/test/gpu_tests/test_input"

        self.get_command_output_class(self.nvcc_command + self.test_input_directory + "/obtain_gpu_cc.cu -o " + self.scratch_directory + "/obtain_gpu_cc")
        output = self.get_command_output_class(self.scratch_directory + "/obtain_gpu_cc")[1].split()

        if len(output) != 2:
            print ("Cannot obtain GPU information")
            exit(-1)
        else:
            print(f"GPU information: num_of_sm ({output[1]}), compute_capability ({output[0]})")
        compute_capability = output[0]
        num_of_sm = output[1]

        self.nvcc_command += " -rdc=true -DNUM_CTA=" + str(int(num_of_sm)*2) + " -DCTA_SIZE=512 -gencode arch=compute_" + compute_capability + ",code=sm_" + compute_capability
        self.nvcc_command += " -std=c++11 -O3 -I " + GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/ -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" "


        # shutil.copytree(GRAPHIT_SOURCE_DIRECTORY + "/test/graphs", self.scratch_directory + "/graphs")
        self.graph_directory = GRAPHIT_SOURCE_DIRECTORY + "/test/graphs"
        self.executable_name = self.scratch_directory + "/test_executable"	
        self.cuda_filename = self.scratch_directory + "/test_cpp.cu"

        self.graphitc_py = GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py"
        self.verifier_input = self.scratch_directory + "/verifier_input"
        
    def assertEqual(self, input, expected):
        if input != expected:
            exit(f'unexpected values: {input}')
            
    def get_command_output_class(self, command):
        output = ""
        if isinstance(command, list):
            proc = subprocess.Popen(command, stdout=subprocess.PIPE)
        else:
            print(command)
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        exitcode = proc.wait()
        for line in proc.stdout.readlines():
            if isinstance(line, bytes):
                line = line.decode()
            output += line.rstrip() + "\n"

        proc.stdout.close()
        return exitcode, output
            
    def get_command_output_class_v2(self, command):
        output = ""
        if isinstance(command, list):
            proc = subprocess.run(command, capture_output=True)
        else:
            print(command)
            proc = subprocess.run(command, shell=True, capture_output=True)
            
        proc.check_returncode()
        exitcode = proc.returncode
        
        line = proc.stdout
        if isinstance(line, bytes):
            line = line.decode()
        output += line.rstrip() + "\n"

        return exitcode, output

    def get_command_output(self, command):
        (exitcode, output) = self.get_command_output_class(command)
        self.assertEqual(exitcode, 0)
        # if exitcode != 0:
        #     exit(f'unexpected exitcode: {exitcode}')
        return output

    def get_command_output_time(self, command):
        exec_time = 0.0
        if isinstance(command, list):
            t0 = time.perf_counter()
            proc = subprocess.run(command, capture_output=True)
            t1 = time.perf_counter()
            exec_time = t1 - t0
        else:
            print(command)
            t0 = time.perf_counter()
            proc = subprocess.run(command, shell=True, capture_output=True)
            t1 = time.perf_counter()
            exec_time = t1 - t0
            
        if proc.returncode: # 当返回值不为0，说明程序非正常结束
            exec_time = 999999999999
        # proc.check_returncode()
        return exec_time

    def run_test(self, input_file_name, use_delta=False):
        exec_time = 0.0
        self.cpp_compile_test(input_file_name, [])
        if use_delta:
            #start point 0, delta 10, verified
            exec_time = self.get_command_output_time(self.executable_name + " " + self.graph_directory + "/4.wel 0 10 v > " + self.verifier_input)
        else:
            exec_time = self.get_command_output_time(self.executable_name + " " + self.graph_directory + "/4.wel 0 v > " + self.verifier_input)
        
        return exec_time


    def cpp_compile_test(self, input_file_name, extra_cpp_args=[]):
        if input_file_name[0] == "/":
            compile_command = self.nvcc_command + input_file_name + " -o " + self.executable_name + " " + " ".join(extra_cpp_args)
        else:
            compile_command = self.nvcc_command + self.test_input_directory + "/" + input_file_name + " -o " + self.executable_name + " " + " ".join(extra_cpp_args)
        self.get_command_output(compile_command)

    def cpp_exec_test(self, input_file_name, extra_cpp_args=[], extra_exec_args=[]):
        self.cpp_compile_test(input_file_name, extra_cpp_args)
        return self.get_command_output(self.executable_name + " " + " ".join(extra_exec_args))

    def graphit_generate_test(self, input_file_name, input_schedule_name=""):
        # if input_file_name[0] != "/":
        #     input_file_name = self.test_input_directory + "/" + input_file_name
        # if input_schedule_name != "" and input_schedule_name[0] != "/":
        #     input_schedule_name = self.test_input_directory + "/" + input_schedule_name

        if input_schedule_name != "":
            self.get_command_output("python " + self.graphitc_py + " -a " + input_file_name + " -f " + input_schedule_name + " -o " + self.cuda_filename)
        else:
            self.get_command_output("python " + self.graphitc_py + " -f " + input_file_name + " -o " + self.cuda_filename)
    
    def compile_and_run(self, gt_path, sch_path):
        self.setUpClass()
        self.graphit_generate_test(gt_path, sch_path)
        t0 = self.run_test(self.cuda_filename, False)
        print(f'exec_time: {t0} s')
        
    def compile(self, gt_path, sch_path):
        self.setUpClass()
        self.graphit_generate_test(gt_path, sch_path)
        
    # def graphit_compile_test(self, input_file_name, input_schedule_name="", extra_cpp_args=[]):	
    #     self.graphit_generate_test(input_file_name, input_schedule_name)
    #     self.cpp_compile_test(self.cuda_filename, extra_cpp_args)

    # def graphit_exec_test(self, input_file_name, input_schedule_name="", extra_cpp_args=[], extra_exec_args=[]):
    #     self.graphit_generate_test(input_file_name, input_schedule_name)
    #     return self.cpp_exec_test(self.cuda_filename, extra_cpp_args, extra_exec_args)

    # def test_simple_graphit_sssp_basic_schedule(self):
    #     self.graphit_generate_test("/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/test_input/inputs/sssp.gt", "/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/test_input/schedules/sssp_default_schedule.gt")
    #     self.run_test(self.cuda_filename, False)


def test():
    compiler = GraphIt_GPU_Compiler()
    
    compiler.compile_and_run("/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/test_input/inputs/sssp.gt", "/home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/test/gpu_tests/test_input/schedules/sssp_default_schedule.gt")
    
    
test()