export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/zhuhaoran/AutoGraph/AutoGraph:$PYTHONPATH

python train_no_graph.py
python train.py
# python train_with_algo.py

# python train_finetune.py