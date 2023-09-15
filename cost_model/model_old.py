import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME


class GraphCostModel_MinkowskiConvolution(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32 
    PLANES = (16,32,64,64)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query 
        self.inplanes = self.INIT_DIM
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))           
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer5 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True)) 
        self.layer6 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer7 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer8 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer9 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer10 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer11 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer12 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer13 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer14 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))

        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature())

        self.feature = nn.Sequential(
          nn.Linear(3, 64),
          nn.ReLU(),
          nn.Linear(64,32),
        )
        
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM*14+32, 256),
          nn.ReLU(),
          nn.Linear(256,128),
        )

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

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
   
    def embed_sparse_matrix(self, x1: ME.SparseTensor, x2) :
        # Sparse Matrix
        y1 = self.layer1(x1)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)
        y14 = self.layer14(y13)


        y1  = self.glob_pool(y1)
        y2  = self.glob_pool(y2)
        y3  = self.glob_pool(y3)
        y4  = self.glob_pool(y4)
        y5  = self.glob_pool(y5)
        y6  = self.glob_pool(y6)
        y7  = self.glob_pool(y7)
        y8  = self.glob_pool(y8)
        y9  = self.glob_pool(y9)
        y10 = self.glob_pool(y10)
        y11 = self.glob_pool(y11)
        y12 = self.glob_pool(y12)
        y13 = self.glob_pool(y13)
        y14 = self.glob_pool(y14)
        
        #y = F.normalize(torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1))
        y = torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1)

        x2 = self.feature(x2[:, :3])
        x1x2 = torch.cat((y,x2), dim=1)
        x1x2 = self.matrix_embedding(x1x2)
        
        #x1x2 = F.normalize(x1x2)

        return x1x2

    def embed_super_schedule(self, y) :
        # Super Schedule
        direction_embed = self.direction(y[0].long())
        parallel_embed = self.parallel(y[1].long())
        frontier_embed = self.frontier(y[1].long())
        SSG_option_embed = self.SSG_option(y[1].long())
        SSG_Num_embed = self.SSG_Num(y[1].long())
        NUMA_embed = self.NUMA(y[1].long())
        
        y1 = torch.cat((direction_embed,parallel_embed,frontier_embed,SSG_option_embed,SSG_Num_embed,NUMA_embed), dim=1)
        y = self.schedule_embedding(y1)

        #y = F.normalize(y)
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
    
    def forward(self, x1: ME.SparseTensor, x2, y):
        # Concat - Final
        x = self.embed_sparse_matrix(x1,x2)
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
