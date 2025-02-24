import torch
import sklearn
from sklearn.metrics import classification_report
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def accuracy(output, labels, output_AUC):
    preds = output.max(1)[1].type_as(labels)
    recall = sklearn.metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1_score = sklearn.metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    AUC = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), output_AUC.detach().cpu().numpy())
    acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = sklearn.metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    return recall, f1_score, AUC, acc, precision


# 处理训练集
def process_train_dataset(train_data):
    new_train_data = []
    for i in range(len(train_data)):
        features = train_data[i]['x'].to(device)
        old_labels = train_data[i]['y'].squeeze().to(device)

        original_adj = train_data[i]['edge_index'].to(device)
        adj_dim = features.shape[0]
        # 创建一个 adj_dimxadj_dim 的零张量作为邻接矩阵
        adjacency_matrix = torch.zeros(adj_dim, adj_dim).to(device)

        # 遍历原始张量，设置邻接矩阵的连接关系
        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1  # 如果是无向图，需要同时设置对称位置为1
        adj = adjacency_matrix

        labels_indices = torch.arange(old_labels.numel()).reshape(old_labels.shape)
        # 打乱索引
        idx_train = labels_indices.flatten()[torch.randperm(labels_indices.numel())].reshape(labels_indices.shape)
        adj, features, new_labels, idx_train = adj, features, old_labels, idx_train   # (注意）

        # 创建 Data 对象
        new_graph = Data(x=features, edge_index=adj, y=old_labels)
        new_train_data.append(new_graph)

    return new_train_data


def process_val_dataset(val_data):
    new_val_data = []
    for i in range(len(val_data)):
        features = val_data[i]['x'].to(device)
        labels = val_data[i]['y'].squeeze().to(device)
        original_adj = val_data[i]['edge_index'].to(device)

        adj_dim = features.shape[0]
        # 创建一个 adj_dimxadj_dim 的零张量作为邻接矩阵
        adjacency_matrix = torch.zeros(adj_dim, adj_dim).to(device)
        # 遍历原始张量，设置邻接矩阵的连接关系
        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1  # 如果是无向图，需要同时设置对称位置为1
        adj = adjacency_matrix

        # 创建 Data 对象
        new_graph = Data(x=features, edge_index=adj, y=labels)
        new_val_data.append(new_graph)
    return new_val_data


def process_test_dataset(test_data):
    new_test_data = []
    for i in range(len(test_data)):
        features = test_data[i]['x']
        labels = test_data[i]['y'].squeeze()
        original_adj = test_data[i]['edge_index']

        adj_dim = features.shape[0]
        # 创建一个 adj_dimxadj_dim 的零张量作为邻接矩阵
        adjacency_matrix = torch.zeros(adj_dim, adj_dim)
        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1
        adj = adjacency_matrix

        # 创建 Data 对象
        new_graph = Data(x=features, edge_index=adj, y=labels)
        new_test_data .append(new_graph)
    return new_test_data


# 传入损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=5, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss