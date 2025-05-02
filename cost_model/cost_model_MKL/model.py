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
        
        # algorithm feature
        self.algo_op1 = nn.Embedding(5, 32)
        self.algo_op2 = nn.Embedding(5, 32)
        self.algo_linear = nn.Sequential(
            nn.Linear(32*2,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # graph data feature
        self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积

        # self.gc1 = GATConv(in_dim, 64, 2)  # 定义第一层图卷积
        # self.gc2 = GATConv(64*2, hidden_dim, 1)  # 定义第二层图卷积
        self.graph_linear = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        # self.schedule_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=256,batch_first=True)
        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # self.schedule_embedding = nn.Sequential(
        #     nn.Linear(16,64),
        #     nn.ReLU(),
        #     nn.Linear(64,128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )
        # self.pre_attn_linear = nn.Linear(256, 256)
        # self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0.3)
        # self.post_attn_linear = nn.Linear(256, 256)

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

    def embed_algo(self, op1, op2):
        y1 = self.algo_op1(op1)
        y2 = self.algo_op2(op2)
        y = torch.cat((y1, y2), dim=1)
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
        
        # 执行图卷积和激活函数 gat使用
        # h = self.gc1(g, h).flatten(1)  # [N, num_heads, hidden_dim]
        # h = F.elu(h)  # [N, hidden_dim]
        # h = self.gc2(g, h).flatten(1)  # [N, hidden_dim]
        # h = F.elu(h)  # [N, hidden_dim]
    
        g.ndata['h'] = h    # 将特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.graph_linear(hg)  # [1*out_dim=128]

    # def embed_schedule(self, y) -> torch.Tensor:
    #     y = self.schedule_embedding(y)
    #     return y

    def embed_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())

        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
        # 将上面四个embed, 构成[batch, 4, 32], 然后过transformer
        # y1 = y1.view(y1.size(0), 4, -1)
        # y1 = self.schedule_encoder_layer(y1)
        # y1 = y1.view(y1.size(0), -1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y


    # def forward_with_cat_feature(self, cat_feature):
    #     cat_feature = self.pre_attn_linear(cat_feature)
    #     cat_feature = cat_feature.permute(1, 0).unsqueeze(-1)
    #     cat_feature, _ = self.attention(cat_feature, cat_feature, cat_feature)
    #     cat_feature = cat_feature.squeeze(-1).permute(1, 0)
    #     cat_feature = self.post_attn_linear(cat_feature)
    #     y = self.final_mlp(cat_feature)
    #     return y

    def forward_after_query(self, graph_feature, schedule, algo_feature=None):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 8, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 8, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 256]

        y = self.final_mlp(cat_feature)
        return y

    def forward(self, graph, schedule, algo_op1=None, algo_op2=None):
        # Concat - Final
        # algo_feature = self.embed_algo(algo_op1, algo_op2)
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
        # 消息产生方式msg = create_op(vporp, w), create_op 可选 {first, +,...}
        self.msg_create = nn.Embedding(num_embeddings=2, embedding_dim=32)
        # 消息聚合方式new_vprop = reduce_op(new_vprop, msg), reduce_op可选(=，min=, +=,...)
        self.msg_reduce = nn.Embedding(num_embeddings=3, embedding_dim=32)
        # 计算模式,可选{部分激活, 全图计算}
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

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        # self.schedule_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=256,batch_first=True)
        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(128*3, 128),
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
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())

        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
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


class AutoGraphModel_MKL(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel_MKL, self).__init__()
        
        # algorithm feature
        # 消息产生方式msg = create_op(vporp, w), create_op 可选 {first, +,...}
        self.msg_create = nn.Embedding(num_embeddings=2, embedding_dim=32)
        # 消息聚合方式new_vprop = reduce_op(new_vprop, msg), reduce_op可选(=，min=, +=,...)
        self.msg_reduce = nn.Embedding(num_embeddings=3, embedding_dim=32)
        # 计算模式,可选{部分激活, 全图计算}
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

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )

        # # Final Layer
        # self.final_mlp = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
            
        #     nn.Linear(64, 1)
        # )

        # ========== 针对不同硬件的多任务预测头 ==========
        # 可根据硬件数量灵活添加/调整不同的头部
        self.final_mlp_mkl = nn.ModuleDict({
            'intel': nn.Sequential(
                nn.Linear(128*3, 128),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ),
            'yitian': nn.Sequential(
                nn.Linear(128*3, 128),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        })


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
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())

        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
        y = self.schedule_embedding(y1)
        return y

    def forward_after_query(self, algo_feature, graph_feature, schedule, hardware_id):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]

        # y = self.final_mlp(cat_feature)
        y = self.final_mlp_mkl[hardware_id](cat_feature)
        return y

    def forward_with_graph_feature(self, graph_feature, algo_ops, schedule, hardware_id):
        algo_feature = self.embed_algo(algo_ops)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]

        # y = self.final_mlp(cat_feature)
        y = self.final_mlp_mkl[hardware_id](cat_feature)
        return y

    def forward_with_graph_feature_v2(self, graph_feature, algo_ops, schedule):
        algo_feature = self.embed_algo(algo_ops)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]

        # y = self.final_mlp(cat_feature)
        # 生成所有头的结果

        y_intel = self.final_mlp_mkl['intel'](cat_feature)
        y_yitian = self.final_mlp_mkl['yitian'](cat_feature)
        return y_intel, y_yitian

    def forward(self, algo_ops, graph, schedule, hardware_id):
        algo_feature = self.embed_algo(algo_ops)
        graph_feature = self.embed_sparse_matrix(graph)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((algo_feature, graph_feature, schedule_feature), dim=1)
        
        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 12, 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 12, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 12*32]
        
        # y = self.final_mlp(cat_feature)
        y = self.final_mlp_mkl[hardware_id](cat_feature)
        return y


class AutoGraphModel_GCN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel_GCN, self).__init__()
        
        # algorithm feature
        self.algo_op1 = nn.Embedding(5, 32)
        self.algo_op2 = nn.Embedding(5, 32)
        self.algorithm_embedding = nn.Sequential(
            nn.Linear(32*2,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # graph data feature
        # self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        # self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积

        self.gc1 = GATConv(in_dim, 64, 2)  # 定义第一层图卷积
        self.gc2 = GATConv(64*2, hidden_dim, 1)  # 定义第二层图卷积
        self.fc1 = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        # self.schedule_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=256,batch_first=True)
        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # self.schedule_embedding = nn.Sequential(
        #     nn.Linear(16,64),
        #     nn.ReLU(),
        #     nn.Linear(64,128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )

        # Transformer Encoder
        # self.pre_attn_linear = nn.Linear(256, 256)
        # self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0.3)
        # self.post_attn_linear = nn.Linear(256, 256)

        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def embed_algo(self, op1, op2):
        y1 = self.algo_op1(op1)
        y2 = self.algo_op2(op2)
        y = torch.cat((y1, y2), dim=1)
        y = self.algorithm_embedding(y)
        return y 

    def embed_sparse_matrix(self, g : DGLGraph) -> torch.Tensor:
        g = dgl.add_self_loop(g)
        # 我们用节点的度作为初始节点特征。
        h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
        h = torch.cat([h1, h2], dim=1)  # [N, 2]

        # 执行图卷积和激活函数 gcn使用
        # h = F.elu(self.gc1(g, h))  # [N, hidden_dim]
        # h = F.elu(self.gc2(g, h))  # [N, hidden_dim]
        
        # 执行图卷积和激活函数 gat使用
        h = self.gc1(g, h).flatten(1)  # [N, num_heads, hidden_dim]
        h = F.elu(h)  # [N, hidden_dim]
        h = self.gc2(g, h).flatten(1)  # [N, hidden_dim]
        h = F.elu(h)  # [N, hidden_dim]
    
        g.ndata['h'] = h    # 将特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.fc1(hg)  # [1*out_dim=128]

    # def embed_schedule(self, y) -> torch.Tensor:
    #     y = self.schedule_embedding(y)
    #     return y

    def embed_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())

        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
        # 将上面四个embed, 构成[batch, 4, 32], 然后过transformer
        # y1 = y1.view(y1.size(0), 4, -1)
        # y1 = self.schedule_encoder_layer(y1)
        # y1 = y1.view(y1.size(0), -1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y


    # def forward_with_cat_feature(self, cat_feature):
    #     cat_feature = self.pre_attn_linear(cat_feature)
    #     cat_feature = cat_feature.permute(1, 0).unsqueeze(-1)
    #     cat_feature, _ = self.attention(cat_feature, cat_feature, cat_feature)
    #     cat_feature = cat_feature.squeeze(-1).permute(1, 0)
    #     cat_feature = self.post_attn_linear(cat_feature)
    #     y = self.final_mlp(cat_feature)
    #     return y

    def forward_after_query(self, graph_feature, schedule, algo_feature=None):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1)
        
        y = self.final(cat_feature)
        return y

    def forward(self, graph, schedule, algo_op1=None, algo_op2=None):
        # Concat - Final
        # algo_feature = self.embed_algo(algo_op1, algo_op2)
        graph_feature = self.embed_sparse_matrix(graph)
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1) # shape [batch, 256]

        y = self.final(cat_feature)
        return y


class AutoGraphModel_GAT(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128, num_heads=2) -> None:
        super(AutoGraphModel_GAT, self).__init__()
        
        # algorithm feature
        self.algo_op1 = nn.Embedding(5, 32)
        self.algo_op2 = nn.Embedding(5, 32)
        self.algorithm_embedding = nn.Sequential(
            nn.Linear(32*2,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # graph data feature
        self.gc1 = GATConv(in_dim, 64, num_heads)  # 定义第一层图卷积
        self.gc2 = GATConv(64*num_heads, hidden_dim, 1)  # 定义第二层图卷积
        
        self.fc1 = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层
        
        # Super Schedule
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
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
    
    
    def embed_algo(self, op1, op2):
        y1 = self.algo_op1(op1)
        y2 = self.algo_op2(op2)
        y = torch.cat((y1, y2), dim=1)
        y = self.algorithm_embedding(y)
        return y 
        
    def embed_sparse_matrix(self, g : DGLGraph) :
        g = dgl.add_self_loop(g)
        # 我们用节点的度作为初始节点特征。

        h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
        h = torch.cat([h1, h2], dim=1)  # [N, 2]
        
        # 执行图卷积和激活函数
        h = self.gc1(g, h).flatten(1)  # [N, num_heads, hidden_dim]
        h = F.elu(h)  # [N, hidden_dim]
        h = self.gc2(g, h).flatten(1)  # [N, hidden_dim]
        h = F.elu(h)  # [N, hidden_dim]
        g.ndata['h'] = h    # 将特征赋予到图的节点
        
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.fc1(hg)  # [1*out_dim=128]

    
    def embed_super_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())
        
        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
    
    def forward(self, graph, schedule):
        # Concat - Final
        # x1 = self.embed_algo(x1)
        graph_feature = self.embed_sparse_matrix(graph)
        schedule_feature = self.embed_super_schedule(schedule)
        
        xy = torch.cat((graph_feature, schedule_feature), dim=1)
        xy = self.final(xy)
        return xy




class AutoGraphModel_gcn_onehot_attn(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel, self).__init__()
        
        # algorithm feature
        self.algo_op1 = nn.Embedding(5, 32)
        self.algo_op2 = nn.Embedding(5, 32)
        self.algo_linear = nn.Sequential(
            nn.Linear(32*2,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # graph data feature
        self.gc1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gc2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.graph_linear = nn.Linear(hidden_dim, out_dim)   # 定义图嵌入线性层

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.schedule_embedding = nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # attention
        self.pre_attn_linear = nn.Linear(256, 256)
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0.3)
        self.post_attn_linear = nn.Linear(256, 256)

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def embed_algo(self, op1, op2):
        y1 = self.algo_op1(op1)
        y2 = self.algo_op2(op2)
        y = torch.cat((y1, y2), dim=1)
        y = self.algo_linear(y)
        return y 

    def embed_graph(self, g : DGLGraph) -> torch.Tensor:
        g = dgl.add_self_loop(g)
        # 我们用节点的度作为初始节点特征。
        h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
        h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
        h = torch.cat([h1, h2], dim=1)  # [N, 2]

        # 执行图卷积和激活函数
        h = F.relu(self.gc1(g, h))  # [N, hidden_dim]
        h = F.relu(self.gc2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h    # 将特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [1*hidden_dim]
        return self.graph_linear(hg)  # [1*out_dim=128]

    def embed_schedule(self, y) -> torch.Tensor:
        y = self.schedule_embedding(y)
        return y

    # def forward_with_cat_feature(self, cat_feature):
    #     cat_feature = self.pre_attn_linear(cat_feature)
    #     cat_feature = cat_feature.permute(1, 0).unsqueeze(-1)
    #     cat_feature, _ = self.attention(cat_feature, cat_feature, cat_feature)
    #     cat_feature = cat_feature.squeeze(-1).permute(1, 0)
    #     cat_feature = self.post_attn_linear(cat_feature)
    #     y = self.final_mlp(cat_feature)
    #     return y

    def forward_after_query(self, graph_feature, schedule, algo_feature=None):
        schedule_feature = self.embed_schedule(schedule)
        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1)

        # cat_feature = self.pre_attn_linear(cat_feature)
        # cat_feature = cat_feature.permute(1, 0).unsqueeze(-1)  # [embed_dim, batch_size, 1]
        # cat_feature, _ = self.attention(cat_feature, cat_feature, cat_feature)
        # cat_feature = cat_feature.squeeze(-1).permute(1, 0)  # [batch_size, embed_dim]
        # cat_feature = self.post_attn_linear(cat_feature)

        y = self.final_mlp(cat_feature)
        return y

    def forward(self, graph, schedule, algo_op1=None, algo_op2=None):
        # Concat - Final
        # algo_feature = self.embed_algo(algo_op1, algo_op2)
        graph_feature = self.embed_graph(graph)
        schedule_feature = self.embed_schedule(schedule)

        cat_feature = torch.cat((graph_feature, schedule_feature), dim=1)

        # cat_feature = self.pre_attn_linear(cat_feature)
        # cat_feature = cat_feature.permute(1, 0).unsqueeze(-1)  # [embed_dim, batch_size, 1]
        # cat_feature, _ = self.attention(cat_feature, cat_feature, cat_feature)
        # cat_feature = cat_feature.squeeze(-1).permute(1, 0)  # [batch_size, embed_dim]
        # cat_feature = self.post_attn_linear(cat_feature)

        y = self.final_mlp(cat_feature)
        return y




class AutoGraphModel_NoGraph(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128) -> None:
        super(AutoGraphModel_NoGraph, self).__init__()

        # Super Schedule one-hot 编码
        # 10000 1000 10 10000
        self.direction = nn.Embedding(5, 32)
        self.parallel = nn.Embedding(4, 32)
        self.frontier = nn.Embedding(2, 32)
        self.SSG_Num = nn.Embedding(5, 32)
        
        # self.SSG_option = nn.Embedding(2, 32)
        # self.NUMA = nn.Embedding(3, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,      # 每个token的特征维度
            nhead=4,         # attention heads
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4     # 使用更深的encoder
        )
        # self.pre_attn_linear = nn.Linear(256, 256)
        # self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0.3)
        # self.post_attn_linear = nn.Linear(256, 256)

        # Final Layer
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )


    def embed_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[:, 0].long())
        parallel_embed = self.parallel(y[:, 1].long())
        frontier_embed = self.frontier(y[:, 2].long())
        SSG_Num_embed = self.SSG_Num(y[:, 3].long())

        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_Num_embed), dim=1)
        # 将上面四个embed, 构成[batch, 4, 32], 然后过transformer
        # y1 = y1.view(y1.size(0), 4, -1)
        # y1 = self.schedule_encoder_layer(y1)
        # y1 = y1.view(y1.size(0), -1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y


    def forward(self, schedule):
        # Concat - Final
        # algo_feature = self.embed_algo(algo_op1, algo_op2)

        cat_feature = self.embed_schedule(schedule)

        cat_feature = cat_feature.view(cat_feature.size(0), -1, 32)   # [batch, 4 32]
        cat_feature = self.transformer_encoder(cat_feature)  # [batch, 4, 32]
        cat_feature = cat_feature.view(cat_feature.size(0), -1)  # [batch, 128]

        y = self.final_mlp(cat_feature)
        return y
