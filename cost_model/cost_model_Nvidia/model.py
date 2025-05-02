import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from dgl import DGLGraph


class AutoGraphModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel, self).__init__()
        
        # # algorithm feature
        # self.algo_op1 = nn.Embedding(5, 32)
        # self.algo_op2 = nn.Embedding(5, 32)
        # self.algo_linear = nn.Sequential(
        #     nn.Linear(32*2,128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )

        # graph data feature
        self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积

        # self.gc1 = GATConv(in_dim, 64, 2)  # 定义第一层图卷积
        # self.gc2 = GATConv(64*2, hidden_dim, 1)  # 定义第二层图卷积
        self.graph_linear = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Schedule feature
        self.kernel_fusion = nn.Embedding(2, 32)
        self.LB = nn.Embedding(5, 32)
        # default edge_only = false, so don't need to embed EB and BS
        # self.EB = nn.Embedding(3, 32, padding_idx=0)
        # self.BS = nn.Embedding(21, 32, padding_idx=0)
        self.direction = nn.Embedding(2, 32)
        self.depulication = nn.Embedding(2, 32)
        self.frontier_output = nn.Embedding(3, 32)
        self.pull_rep = nn.Embedding(3, 32, padding_idx=0)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*6,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        mlp_dim = 128 * 2

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(mlp_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

    # def embed_algo(self, op1, op2):
    #     y1 = self.algo_op1(op1)
    #     y2 = self.algo_op2(op2)
    #     y = torch.cat((y1, y2), dim=1)
    #     y = self.algo_linear(y)
    #     return y 

    def embed_sparse_matrix(self, g : DGLGraph) -> torch.Tensor:
        g = dgl.add_self_loop(g)
        # 我们用节点的度作为初始节点特征。
        h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
        h = torch.cat([h1, h2], dim=1)  # [N, 2]

        # 执行图卷积和激活函数 gcn使用
        h = F.elu(self.gc1(g, h))  # [N, hidden_dim]
        h = F.elu(self.gc2(g, h))  # [N, hidden_dim]
        
        # 执行图卷积和激活函数 gat使用
        # h = self.gc1(g, h).flatten(1)  # [N, num_heads, hidden_dim]
        # h = F.elu(h)  # [N, hidden_dim]
        # h = self.gc2(g, h).flatten(1)  # [N, hidden_dim]
        # h = F.elu(h)  # [N, hidden_dim]
    
        g.ndata['h'] = h    # 将特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.graph_linear(hg)  # [1*out_dim=128]

    def embed_schedule(self, y) :
        y_kernel_fusion = self.kernel_fusion(y[:, 0].long())
        y_LB = self.LB(y[:, 1].long())
        # y_EB = self.EB(y[:, 2].long())
        # y_BS = self.BS(y[:, 3].long())
        y_direction = self.direction(y[:, 2].long())
        y_depulication = self.depulication(y[:, 3].long())
        y_frontier_output = self.frontier_output(y[:, 4].long())
        y_pull_rep = self.pull_rep(y[:, 5].long())
        
        y1 = torch.cat((y_kernel_fusion, y_LB, y_direction, y_depulication, y_frontier_output, y_pull_rep), dim=1)
        y = self.schedule_embedding(y1)
        return y

    def forward_after_query(self, graph_feature, schedule):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 8, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 8, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 256]

        y = self.final_mlp(cat_feature)
        return y

    def forward(self, graph, schedule):
        # Concat - Final
        graph_feature = self.embed_sparse_matrix(graph)
        schedule_feature = self.embed_schedule(schedule)

        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1) # shape [batch, 256]
        
        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 8, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 8, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 256]

        y = self.final_mlp(cat_feature)
        return y


class AutoGraphModel_algo(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel_algo, self).__init__()

        # algorithm feature
        # msg = create_op(vporp, w), create_op option {first, +,...}
        self.msg_create = nn.Embedding(num_embeddings=2, embedding_dim=32)
        # new_vprop = reduce_op(new_vprop, msg), reduce_op option(=，min=, +=,...)
        self.msg_reduce = nn.Embedding(num_embeddings=3, embedding_dim=32)
        # compute mode,option{active nodes, all nodes}
        self.compute_mode = nn.Embedding(num_embeddings=2, embedding_dim=32)

        self.algo_linear = nn.Sequential(
            nn.Linear(32*3,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # graph data feature
        self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.graph_linear = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Schedule feature
        self.kernel_fusion = nn.Embedding(2, 32)
        self.LB = nn.Embedding(5, 32)
        # default edge_only = false, so don't need to embed EB and BS
        # self.EB = nn.Embedding(3, 32, padding_idx=0)
        # self.BS = nn.Embedding(21, 32, padding_idx=0)
        self.direction = nn.Embedding(2, 32)
        self.depulication = nn.Embedding(2, 32)
        self.frontier_output = nn.Embedding(3, 32)
        self.pull_rep = nn.Embedding(3, 32, padding_idx=0)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*6,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        mlp_dim = 128 * 3

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(mlp_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

        # # ========== 针对不同硬件的多任务预测头 ==========
        # # 可根据硬件数量灵活添加/调整不同的头部
        # self.hardware_heads = nn.ModuleDict({
        #     'HW0': nn.Sequential(
        #         nn.Linear(128*3, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 1)
        #     ),
        #     'HW1': nn.Sequential(
        #         nn.Linear(128*3, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 1)
        #     )
        #     # 可以继续添加其他硬件：'HW2', 'HW3', ...
        # })
        # forward中根据不同硬件选择不同预测头
        # y = self.hardware_heads[hardware_id](cat_feature)

    def embed_algo(self, algo_ops):
        msg_create_y = self.msg_create(algo_ops[:, 0].long())
        msg_reduce_y = self.msg_reduce(algo_ops[:, 1].long())
        compute_mode_y = self.compute_mode(algo_ops[:, 2].long())
        y = torch.cat((msg_create_y, msg_reduce_y, compute_mode_y), dim=1)
        y = self.algo_linear(y)
        return y 

    def embed_sparse_matrix(self, g : DGLGraph) -> torch.Tensor:
        g = dgl.add_self_loop(g)
        # 我们用节点的度作为初始节点特征。
        h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
        h = torch.cat([h1, h2], dim=1)  # [N, 2]

        # 执行图卷积和激活函数 gcn使用
        h = F.elu(self.gc1(g, h))  # [N, hidden_dim]
        h = F.elu(self.gc2(g, h))  # [N, hidden_dim]

        g.ndata['h'] = h    # 将特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.graph_linear(hg)  # [1*out_dim=128]

    def embed_schedule(self, y) :
        y_kernel_fusion = self.kernel_fusion(y[:, 0].long())
        y_LB = self.LB(y[:, 1].long())
        # y_EB = self.EB(y[:, 2].long())
        # y_BS = self.BS(y[:, 3].long())
        y_direction = self.direction(y[:, 2].long())
        y_depulication = self.depulication(y[:, 3].long())
        y_frontier_output = self.frontier_output(y[:, 4].long())
        y_pull_rep = self.pull_rep(y[:, 5].long())
        
        y1 = torch.cat((y_kernel_fusion, y_LB, y_direction, y_depulication, y_frontier_output, y_pull_rep), dim=1)
        y = self.schedule_embedding(y1)
        return y

    def forward_after_query(self, algo_feature, graph_feature, schedule):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]

        y = self.final_mlp(cat_feature)
        return y

    def forward_with_graph_feature(self, graph_feature, algo_ops, schedule):
        algo_feature = self.embed_algo(algo_ops)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]

        y = self.final_mlp(cat_feature)
        return y

    def forward(self, algo_ops, graph, schedule):
        algo_feature = self.embed_algo(algo_ops)
        graph_feature = self.embed_sparse_matrix(graph)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)
        
        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]
        
        y = self.final_mlp(cat_feature)
        return y


class AutoGraphModel_NoGraph(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel_NoGraph, self).__init__()

        # # algorithm feature
        # # msg = create_op(vporp, w), create_op option {first, +,...}
        # self.msg_create = nn.Embedding(num_embeddings=2, embedding_dim=32)
        # # new_vprop = reduce_op(new_vprop, msg), reduce_op option(=，min=, +=,...)
        # self.msg_reduce = nn.Embedding(num_embeddings=3, embedding_dim=32)
        # # compute mode,option{active nodes, all nodes}
        # self.compute_mode = nn.Embedding(num_embeddings=2, embedding_dim=32)

        # self.algo_linear = nn.Sequential(
        #     nn.Linear(32*3,128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )

        # # graph data feature
        # self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        # self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        # self.graph_linear = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Schedule feature
        self.kernel_fusion = nn.Embedding(2, 32)
        self.LB = nn.Embedding(5, 32)
        # default edge_only = false, so don't need to embed EB and BS
        # self.EB = nn.Embedding(3, 32, padding_idx=0)
        # self.BS = nn.Embedding(21, 32, padding_idx=0)
        self.direction = nn.Embedding(2, 32)
        self.depulication = nn.Embedding(2, 32)
        self.frontier_output = nn.Embedding(3, 32)
        self.pull_rep = nn.Embedding(3, 32, padding_idx=0)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*6,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        mlp_dim = 128 * 1

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(mlp_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

    def embed_schedule(self, y) :
        y_kernel_fusion = self.kernel_fusion(y[:, 0].long())
        y_LB = self.LB(y[:, 1].long())
        # y_EB = self.EB(y[:, 2].long())
        # y_BS = self.BS(y[:, 3].long())
        y_direction = self.direction(y[:, 2].long())
        y_depulication = self.depulication(y[:, 3].long())
        y_frontier_output = self.frontier_output(y[:, 4].long())
        y_pull_rep = self.pull_rep(y[:, 5].long())
        
        y1 = torch.cat((y_kernel_fusion, y_LB, y_direction, y_depulication, y_frontier_output, y_pull_rep), dim=1)
        y = self.schedule_embedding(y1)
        return y

    def forward(self, schedule):
        cat_feature = self.embed_schedule(schedule)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 4 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 4, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 128]

        y = self.final_mlp(cat_feature)
        return y
