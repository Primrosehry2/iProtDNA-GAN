import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
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


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, generate_node=None, min_node=None):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nhid, 2)
        self.attention = Attention(nfeat*2, 1)
        self.generate_node = generate_node
        self.min_node = min_node
        self.dropout = dropout
        self.eps = 1e-10

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.gc2(x, adj)
        x2 = self.gc3(x, adj)
        return F.log_softmax(x1, dim=1), F.log_softmax(x2, dim=1), F.softmax(x1, dim=1)[:,-1]

    def get_embedding(self,x , adj):
        x = F.relu(self.gc1(x, adj))
        x = torch.spmm(adj, x)
        return x


class Generator(nn.Module):
    def __init__(self,  dim):
        super(Generator, self).__init__( )
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, dim)
        self.fc4 = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = (x+1)/2
        return x


# 实例化放入 SAGE+MLP 组合模型
class SageConv(Module):

    def __init__(self, in_features, out_features, bias=False):   # in_features是输入特征的数量，out_features是输出特征的数量。
        super(SageConv, self).__init__()
        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)   # 使用正态分布来初始化权重

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)   # 如果存在偏置项，nn.init.constant_ 将偏置初始化为 0

    def forward(self, features, adj):   # features（节点特征）和 adj（邻接矩阵）作为输入
        if adj.layout != torch.sparse_coo:  # adj.layout: torch.strided
            if len(adj.shape) == 3:   # 如果 adj 是一个三维张量（len(adj.shape) == 3），则它可能表示一批图的邻接矩阵
                neigh_feature = torch.bmm(adj, features) / (
                        adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)   # 使用 torch.bmm（批量矩阵乘法）来计算邻居特征，并对结果进行归一化
            else:   # 如果 adj 是二维的，表示单个图的邻接矩阵，使用 torch.mm（矩阵乘法）来计算邻居特征，并对结果进行归一化。
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            # print("spmm not implemented for batch training. Note!")
            # 如果 adj 是稀疏格式的，使用 torch.spmm（稀疏矩阵乘法）来计算邻居特征，并对结果进行归一化。
            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)   # 将每个节点自身的特征和对应的邻居特征拼接起来，形成一个新的特征矩阵。
        combined = self.proj(data)   # 将拼接后的特征映射到一个新的特征空间。这是图卷积操作的核心，其中 data 是拼接后的特征矩阵
        return combined


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,out_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),   # 创建第一个线性层，将输入特征从 input_dim 维度映射到 hidden_dim1 维度
            nn.LeakyReLU(inplace=True),   # 创建一个 LeakyReLU 激活函数层，inplace=True 表示在原地进行操作，可以减少内存消耗，但不允许在计算图中进行梯度跟踪
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim2, out_dim),   # 将特征从 hidden_dim2 维度映射到输出维度 out_dim
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):   # 每个模块的前向传播函数。它接收输入 x。
        x = self.model(x)
        return x


class Sage_En(nn.Module):
    def __init__(self, nfeat, nembed, dropout,input_dim,hidden_dim1,hidden_dim2,out_dim):
        super(Sage_En, self).__init__()
        self.sage1 = SageConv(nfeat, nembed)   # 聚合图中节点的邻居特征
        self.mlp=MLP(input_dim,hidden_dim1,hidden_dim2,out_dim)   # 学习节点特征到类别标签的映射
        self.dropout = dropout

    def forward(self, x, adj):   # 模型的前向传播函数。它接收两个参数：x 是节点特征矩阵，adj 是邻接矩阵
        x = F.relu(self.sage1(x, adj))
        # print(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)   # 将经过 GraphSAGE 层和 dropout 处理的特征 x 输入到多层感知机 mlp 中进行进一步的处理
        preds_probability= F.softmax(x, dim=1)   # 应用 softmax 函数对 MLP 的输出进行归一化，得到每个节点属于各个类别的概率分布

        return preds_probability