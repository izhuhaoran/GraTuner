import random 
import itertools
import time
import sys
import random
import os
import math
import numpy as np

GraphitPath = '/data/zhr_data/AutoGraph/graphit'

py_graphitc_file = f"{GraphitPath}/build/bin/graphitc.py"
serial_compiler = "g++"

#if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++"

config = {
  'direction' : 'SparsePush',
  'parallelization' : 'serial',
  'numSSG' : ['fixed-vertex-count', '1'],
  'DenseVertexSet' : 'boolean-array',
  'bucket_update_strategy' : 'eager_priority_update'
}


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
  

  #configures parallelization commands
  def write_par_schedule(self, cfg, new_schedule, direction):
      use_evp = False

      if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
          use_evp = True   

      if use_evp == False or self.use_NUMA == True:
          # if don't use edge-aware parallel (vertex-parallel)
          # edge-parallel don't work with NUMA (use vertex-parallel when NUMA is enabled) 
          if cfg['parallelization'] == 'serial': 
              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"serial\");"
          else:
              # if NUMA is used, then we only use dynamic-vertex-parallel as edge-aware-vertex-parallel do not support NUMA yet
              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
      elif use_evp == True and self.use_NUMA == False:   
          #use_evp is True
          if direction == "DensePull": 
              # edge-aware-dynamic-vertex-parallel is only supported for the DensePull direction
              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024, \"DensePull\");"

          elif direction == "SparsePush-DensePull":
              # For now, only the DensePull direction uses edge-aware-vertex-parallel
              # the SparsePush should still just use the vertex-parallel methodx
              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024,  \"DensePull\");"

              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\",1024,  \"SparsePush\");"

          else:
              #use_evp for SparsePush, DensePush-SparsePush should not make a difference
              new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
      else:
          print ("Error in writing parallel schedule")
          exit()
      return new_schedule

  def write_numSSG_schedule(self, numSSG, new_schedule, direction):
      # No need to insert for a single SSG
      if numSSG == 0 or numSSG ==1:
          return new_schedule
      # configuring cache optimization for DensePull direction
      if direction == "DensePull" or direction == "SparsePush-DensePull":
          new_schedule = new_schedule + "\n    program->configApplyNumSSG(\"s1\", \"fixed-vertex-count\", " + str(numSSG) + ", \"DensePull\");"
      return new_schedule

  def write_delta_schedule(self, delta, new_schedule):
      new_schedule = new_schedule + "\n    program->configApplyPriorityUpdateDelta(\"s1\", " + str(delta) + " );"
      return new_schedule

  def write_bucket_update_schedule(self, bucket_update_strategy, new_schedule):
      new_schedules = new_schedule + "\n    program->configApplyPriorityUpdate(\"s1\", \"" + bucket_update_strategy + "\" );"
      return new_schedules

  def write_NUMA_schedule(self,  new_schedule, direction):
      # configuring NUMA optimization for DensePull direction
      if self.use_NUMA:
          if direction == "DensePull" or direction == "SparsePush-DensePull":
              new_schedule = new_schedule + "\n    program->configApplyNUMA(\"s1\", \"static-parallel\" , \"DensePull\");"
      return new_schedule

  def write_denseVertexSet_schedule(self, enable_pull_bitvector, new_schedule, direction):
      # for now, we only use this for the src vertexset in the DensePull direciton
      if direction == "DensePull" or direction == "SparsePush-DensePull":
          if enable_pull_bitvector:
              new_schedule = new_schedule + "\n    program->configApplyDenseVertexSet(\"s1\",\"bitvector\", \"src-vertexset\", \"DensePull\");"
      return new_schedule

  def write_cfg_to_schedule(self, cfg):
      #write into a schedule file the configuration
      direction = cfg['direction']
      numSSG = cfg['numSSG']
      delta = cfg['delta']
      bucket_update_strategy = cfg['bucket_update_strategy']

      new_schedule = ""
      direction_schedule_str = "\n    program->configApplyDirection(\"s1\", \"$direction\");" 
      if self.args.default_schedule_file != "":
          f = open(self.args.default_schedule_file,'r')
          default_schedule_str = f.read()
          f.close()
      else:
          default_schedule_str = "schedule: "
      
          
      #eager only works with SparsePush for now
      if bucket_update_strategy == 'eager_priority_update':
          new_schedule = default_schedule_str + direction_schedule_str.replace('$direction', 'SparsePush')
      else:
          new_schedule = default_schedule_str + direction_schedule_str.replace('$direction', cfg['direction'])

      new_schedule = self.write_par_schedule(cfg, new_schedule, direction)
      new_schedule = self.write_numSSG_schedule(numSSG, new_schedule, direction)
      new_schedule = self.write_NUMA_schedule(new_schedule, direction)
      new_schedule = self.write_delta_schedule(delta, new_schedule)
      new_schedule = self.write_bucket_update_schedule(bucket_update_strategy, new_schedule)

      use_bitvector = False
      if cfg['DenseVertexSet'] == 'bitvector':
          use_bitvector = True
      new_schedule = self.write_denseVertexSet_schedule(use_bitvector, new_schedule, direction)


      print (cfg)
      print (new_schedule)

      self.new_schedule_file_name = 'schedule_0' 
      print (self.new_schedule_file_name)
      f1 = open (self.new_schedule_file_name, 'w')
      f1.write(new_schedule)
      f1.close()

  def compile(self, cfg,  id):
      """                                                                          
      Compile a given configuration in parallel                                    
      """


      #compile the schedule file along with the original algorithm file
      compile_graphit_cmd = 'python ' + py_graphitc_file +  ' -a  {algo_file} -f {schedule_file} -i ../include/ -l ../build/lib/libgraphitlib.a  -o test.cpp'.format(algo_file=self.args.algo_file, schedule_file=self.new_schedule_file_name) 

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
      return self.call_program(compile_cpp_cmd)






def filter(l) :
  i1 = l.index('i1')
  k1 = l.index('k1')
  i0 = l.index('i0')
  k0 = l.index('k0')
  return (i1<i0) and (k1<k0) 

if __name__ == '__main__':
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix == None : 
    print("Err : environment variable WACO_HOME is not defined")
    quit() 

  with open(sys.argv[1]) as f :
    lines = f.read().splitlines()
  
  for idx, mtx in enumerate(lines) :
    csr = np.fromfile(waco_prefix + "/dataset/{0}.csr".format(mtx), count=3, dtype='<i4')
    num_row, num_col, num_nonzero = csr[0], csr[1], csr[2]

    cfgs = set()
    cfg = {}
    while (len(cfgs) < 100) :
      cfg['isplit'] = random.choice([1<<p for p in range(int(math.log(num_row,2)))]) 
      cfg['ksplit'] = random.choice([1<<p for p in range(int(math.log(num_col,2)))]) 
      cfg['rankorder'] = random.choice([p for p in list(itertools.permutations(['i1','i0','k1','k0'])) if filter(p)]) 
      cfg['i1f'] = random.choice([0,1])
      cfg['i0f'] = random.choice([0,1])
      cfg['k1f'] = random.choice([0,1])
      cfg['k0f'] = random.choice([0,1])
      cfg['paridx'] = random.choice(['i1'])
      cfg['parnum'] = random.choice([48])
      cfg['parchunk'] = random.choice([1<<p for p in range(9)])
      isplit, ksplit = cfg['isplit'], cfg['ksplit']
      rankorder= " ".join(cfg['rankorder'])
      i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
      paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
      cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(isplit, ksplit, rankorder, i1f, i0f, k1f, k0f, paridx, parnum, parchunk))
    
    f = open("./config/{}.txt".format(mtx), 'w')
    for sched in cfgs : f.write(sched)
    f.close()
