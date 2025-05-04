# GraTuner: 高效图计算程序自动调优器

## 项目简介
GraTuner是一个基于评估与探索双效增强的图计算程序自动调优系统，通过创新的轻量级代价模型和图结构化搜索策略，将调优时间从数小时压缩至1分钟内，同时实现平均1.34倍的程序性能提升。

## 核心优势
- **极速调优**：端到端调优时间降低161.11倍（平均<1分钟）
- **精准预测**：8ms级单次评估时延的多维度特征融合代价模型
- **高效搜索**：基于图分析的调优方法减少93%迭代轮次
- **跨平台支持**：支持Intel/NVIDIA/YiTian等多硬件平台

## 目录结构

```text
GraTuner/
├── cost_model/          # 代价模型实现
│   ├── cost_model_Intel # Intel平台特征编码器
│   └── ...              # 其他硬件平台模型
├── dataset/             # 代价模型训练数据集
│   ├── dataset_Intel    # Intel平台特征数据
│   └── ...              # 其他平台数据
├── search/              # 图结构化搜索算法
│   ├── search_Intel     # Intel平台搜索策略
│   └── ...              # 其他平台实现
├── graphs/              # 输入图数据集（如社交网络、交通图等）
├── scripts/             # 快捷脚本
├── third_party/         # 依赖库（主要含GraphIt及其系列框架）
└── common/              # 公共函数与配置
```

## 快速开始
### 1. 环境准备
```bash
# 创建虚拟环境（可选）
conda create -n gratuner python=3.9
conda activate gratuner

# 安装依赖
pip install -r requirements.txt

# 编译第三方库
cd third_party/graphit
mkdir build
cd build
cmake ..
make -j
```


### 2. 预训练代价模型和图结构搜索空间

- 代价模型训练

```bash
# 基础训练命令（以Intel平台为例）
cd GraTuner/cost_model/cost_model_Intel
python train_finetune.py
```

- 搜索空间构建和边权更新

```bash
# 基础构建命令（以Intel平台为例）
cd GraTuner/search/search_Intel
python graph_create.py
python graph_train.py
```

### 3. 运行调优
```bash
# 基础调优命令（以Intel平台为例）
cd ./search
python graph_search.py \
  --algo=bfs \
  --graph=youtube-u-growth \
  --k=10 \
  --m=1

```


