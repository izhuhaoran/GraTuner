python ./test_one_cfg.py
python ./graphitc_test.py -a /home/zhuhaoran/AutoGraph/GraTuner/third_party/graphit_gpu/autotune/gpu_apps/test.gt -f schedule_2 -o test_2.cu
/usr/local/cuda/bin/nvcc  -ccbin /usr/bin/c++ -std=c++11 -I ../src/runtime_lib/ -o test_2 -Xcompiler "-w" -O3 test_2.cu -DNUM_CTA=160 -DCTA_SIZE=512 -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=sm_70 --use_fast_math -Xptxas "-v -dlcm=ca --maxrregcount=64" -rdc=true -DFRONTIER_MULTIPLIER=3
#/usr/local/cuda/bin/nvcc  -ccbin /usr/bin/c++ -std=c++11 -I ../src/runtime_lib/ -o test -Xcompiler "-w" -O3 test.cu -DNUM_CTA=60 -DCTA_SIZE=512 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 --use_fast_math -Xptxas "-v -dlcm=ca --maxrregcount=64" -rdc=true -DFRONTIER_MULTIPLIER=2

# ./test_2 /home/zhuhaoran/AutoGraph/graphs/sx-stackoverflow/sx-stackoverflow.el 1

# rm -rf ./schedule_2
# rm -rf ./test_2
# rm -rf ./test_2.cu
# rm -rf ./compile_2.cpp
# rm -rf ./compile_2.o