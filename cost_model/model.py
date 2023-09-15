import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv
from dgl import DGLGraph


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphCostModel(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=256, out_dim=128) -> None:
        super(GraphCostModel, self).__init__()
        
        # graph data feature
        
        # hidden_dim = 256
        # dropout = 0.5
        # self.gc1 = GraphConvolution(in_dim, hidden_dim)
        # self.gc2 = GraphConvolution(hidden_dim, out_dim)
        # self.dropout = dropout
        
        self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.fc1 = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层
        
        # Super Schedule
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(5, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_option = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(10, 32)
        self.NUMA = nn.Embedding(3, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*6,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    # # waco style
    # def embed_sparse_matrix(self, x1, x2) :
    #     # Sparse Matrix
    #     y1 =  F.relu(self.gc1(x1, x2))
    #     # y1 =  F.dropout(y1, self.dropout, training=self.training)
    #     y2 = self.gc2(y1, x2)

    #     return y2
        
    def embed_sparse_matrix(self, g : DGLGraph) :
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        
        # 执行图卷积和激活函数
        h = F.relu(self.gc1(g, h))  # [N, hidden_dim]
        h = F.relu(self.gc2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h    # 将特征赋予到图的节点
        
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.fc1(hg)  # [1*out_dim=128]

    
    def embed_super_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[0].long())
        parallel_embed = self.parallel(y[1].long())
        frontier_embed = self.frontier(y[2].long())
        SSG_option_embed = self.SSG_option(y[3].long())
        SSG_Num_embed = self.SSG_Num(y[4].long())
        NUMA_embed = self.NUMA(y[5].long())
        
        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_option_embed,SSG_Num_embed,NUMA_embed), dim=1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
    
    def forward(self, x1, x2, y):
        # Concat - Final
        x = self.embed_sparse_matrix(x1,x2)
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy

