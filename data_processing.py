import warnings
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import random
from models import Generator, GCN
from utils import euclidean_dist, accuracy

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


# 忽略所有警告
warnings.filterwarnings("ignore")

def train(discriminator,optimizer,num_real,num_false,idx_train,idx_val,idx_test,labels,minority_index,features, adj,fastmode=False):
    global max_recall, test_recall, test_f1, test_AUC, test_acc, test_pre
    discriminator.train()
    output, output_gen, output_AUC = discriminator(features, adj)   # output 和 output_gen 分别代表了模型对真实节点和生成节点的预测结果
    labels_true = torch.cat((torch.LongTensor(num_real).fill_(0), torch.LongTensor(num_false).fill_(1))).to(device)   # num_real 是真实数据的数量，num_false 是生成数据的数量
    # if args.cuda:
    # if True:
    #     labels_true=labels_true.cuda()

    # 计算损失函数，包括负对数似然损失和特征空间中的距离损失
    loss_dis = - euclidean_dist(features[minority_index], features[minority_index]).mean()
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) \
                     + F.nll_loss(output_gen[idx_train], labels_true) \
                     + loss_dis
    optimizer.zero_grad()
    loss_train.backward(retain_graph=True)
    optimizer.step()

    if not fastmode:
        discriminator.eval()
        output, output_gen, output_AUC = discriminator(features, adj)

    recall_val, f1_val, AUC_val, acc_val, pre_val = accuracy(output[idx_val], labels[idx_val], output_AUC[idx_val])
    recall_train, f1_train, AUC_train, acc_train, pre_train = accuracy(output[idx_train], labels[idx_train], output_AUC[idx_train])

    # 如果当前验证集的性能超过了之前的最佳性能，则更新最佳性能指标，并在测试集上评估模型
    if max_recall < (recall_val + acc_val)/2:
        output, output_gen, output_AUC = discriminator(features, adj)
        recall_tmp, f1_tmp, AUC_tmp, acc_tmp, pre_tmp = accuracy(output[idx_test], labels[idx_test], output_AUC[idx_test])
        test_recall = recall_tmp
        test_f1 = f1_tmp
        test_AUC = AUC_tmp
        test_acc = acc_tmp
        test_pre = pre_tmp
        max_recall = (recall_val + acc_val)/2

    return recall_val, f1_val, acc_val, recall_train, f1_train, acc_train


# 第一步：加载数据
pt_file_path = './Graph/573.pt'
data = torch.load(pt_file_path, map_location=device)

data_1 = []   # 存储节点标签为1个数大于5的图
data_2 = []   # 存储节点标签为1个数小于5的图
for i in range(len(data)):
    example = data[i]
    label = example.y
    num_zeros = torch.sum(label == 0).item()
    num_ones = torch.sum(label == 1).item()
    if num_ones < 5:
        data_2.append(example)
    else:
        data_1.append(example)

random.shuffle(data_1)

num_to_extract = int(len(data)*0.7)
data_train = data_1[:num_to_extract]
data_test = data_1[num_to_extract:]+data_2

print(len(data_train), len(data_test))

# 保存列表到 .pt 文件
torch.save(data_test, './Graph_2.0/573_val.pt')

new_data = []
for i in range(len(data_train)):
    # print(type(data))
    # 第二步：获得一个样本案例
    example = data_train[i].to(device)
    label = example.y
    features = example.x
    num_zeros = torch.sum(label == 0).item()
    num_ones = torch.sum(label == 1).item()
    print(i, example)
    # print(f"0的个数: {num_zeros}")
    # print(f"1的个数: {num_ones}")
    if num_ones < 5:
        continue

    # 判断少数类样本
    if num_zeros > num_ones:
        gap = abs(num_zeros-num_ones)
        # print(f"少数类样本的标签为'1'，差距为{ gap }")
    else:
        gap = abs(num_zeros - num_ones)
        # print(f"少数类样本的标签为'0'，差距为{ gap }")

    dim = num_ones
    model_generator = Generator(dim).to(device)
    weight_decay = 0.0005
    lr = 0.0001
    optimizer_G = torch.optim.Adam(model_generator.parameters(), lr=lr, weight_decay=weight_decay)

    # 将生成的图放入判别器，然后计算损失进行反向传播
    hidden = 1280
    nclass = 2
    dropout = 0.1
    discriminator = GCN(nfeat=2560,
                        nhid=hidden,
                        nclass=nclass,
                        dropout=dropout).to(device)

    optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    num_false = gap
    labels_true = torch.LongTensor(num_false).fill_(0).to(device)
    labels_min = torch.LongTensor(num_false).fill_(1).to(device)
    generate_node_index = torch.tensor(list(range(features.shape[0] ,features.shape[0]+num_false))).to(device)

    # 按照3：1：1的比例划分为训练、验证和测试
    labels = label.flatten()
    # 获取正负样本的索引
    positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
    negative_indices = (labels == 0).nonzero(as_tuple=True)[0]

    # 划分正样本：3：1：1
    total_size_pos = len(positive_indices)
    train_size_pos = int(total_size_pos * 3/5)
    val_size_pos = test_size_pos = int(total_size_pos * 1/5)
    # 随机打乱索引
    permuted_indices_pos = positive_indices[torch.randperm(total_size_pos)].to(device)
    # 划分正索引
    idx_train_pos = permuted_indices_pos[:train_size_pos]
    idx_val_pos = permuted_indices_pos[train_size_pos:train_size_pos + val_size_pos]
    idx_test_pos = permuted_indices_pos[train_size_pos + val_size_pos:]

    # 划分负样本：3：1：1
    total_size_neg = len(negative_indices)
    train_size_neg = int(total_size_neg * 0.6)
    val_size_neg = test_size_neg = int(total_size_neg * 0.2)
    # 随机打乱索引
    permuted_indices_neg = negative_indices[torch.randperm(total_size_neg)].to(device)
    # 划分负索引
    idx_train_neg = permuted_indices_neg[:train_size_neg]
    idx_val_neg = permuted_indices_neg[train_size_neg:train_size_neg + val_size_neg]
    idx_test_neg = permuted_indices_neg[train_size_neg + val_size_neg:]

    # 合并索引
    num_real = torch.cat((idx_train_pos, idx_train_neg)).shape[0]
    idx_train = torch.cat((idx_train_pos, idx_train_neg, generate_node_index))
    idx_val = torch.cat((idx_val_pos, idx_val_neg))
    idx_test = torch.cat((idx_test_pos, idx_test_neg))

    generate_node_label = torch.ones((1, gap), dtype=label.dtype).to(device)

    # 将两个张量连接起来
    new_labels = torch.cat((label, generate_node_label), dim=1)
    new_labels = new_labels.flatten()

    max_recall = 0
    test_recall = 0
    test_f1 = 0
    test_AUC = 0
    test_acc = 0
    test_pre = 0

    for i in range(10):
        # generate_node_z:生成节点的噪声
        generate_node_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (gap, 100)))).to(device)

        # 实例化生成器Generator ：生成带有噪声的节点特征
        adj_min = model_generator(generate_node_z)
        minority_index = torch.nonzero(label == 1)[:, 1]
        feature_minority = features[minority_index]

        link_relationship = F.softmax(adj_min[:, 0:minority_index.shape[0]], dim=1)
        generate_node_feature = torch.mm(link_relationship, features[minority_index])  # 𝑋𝑔 = 𝑇𝑋𝑚𝑖n
        new_feature = torch.cat((features,generate_node_feature), dim=0)
        # print('new_feature:',new_feature.shape)

        # 定义阈值
        threshold = 0.038
        # 应用阈值化
        adj_generate_node = (link_relationship > threshold).int()
        mapping_dict_row = {idx: value+features.shape[0] for idx, value in enumerate(range(gap))}
        mapping_dict_col = {idx: value.item() for idx, value in enumerate(minority_index)}

        # 获得非零元素的坐标
        nonzero_coords = torch.nonzero(adj_generate_node)
        for i in range(nonzero_coords.shape[0]):
            row = nonzero_coords[i].clone()  # 使用 clone() 方法确保我们对原张量的操作不会影响到原张量
            row[0] = mapping_dict_row[row[0].item()]
            row[1] = mapping_dict_col[row[1].item()]
            nonzero_coords[i] = row

        edge_index_orginal = example.edge_index
        new_edge_index = torch.cat((edge_index_orginal, nonzero_coords.T), dim=1)
        # print('new_edge_index:',new_edge_index.shape)

        # 初始化邻接矩阵
        adjacency_matrix = torch.zeros((new_feature.shape[0], new_feature.shape[0]), dtype=torch.int).to(device)
        # 填充邻接矩阵
        for i in range(new_edge_index.shape[1]):
            start_node = new_edge_index[0, i]
            end_node = new_edge_index[1, i]
            adjacency_matrix[start_node, end_node] = 1
            adjacency_matrix[end_node, start_node] = 1
        # print('adjacency_matrix:',adjacency_matrix.shape)

        # 获取邻接矩阵的非零元素的位置
        indices = adjacency_matrix.nonzero().t()  # 转置为 COO 格式
        # 获取邻接矩阵的非零元素的值
        values = adjacency_matrix[indices[0], indices[1]]
        # 创建稀疏矩阵
        sparse_adj = torch.sparse_coo_tensor(indices, values, size=adjacency_matrix.size(),dtype=torch.float32)

        for epoch in range(100):
            recall_val, f1_val, acc_val, recall_train, f1_train, acc_train=train(discriminator,optimizer,num_real,num_false,idx_train,idx_val,idx_test,new_labels,minority_index,new_feature, sparse_adj)
        print("Epoch:", i + 1,
              "train_recall=", "{:.5f}".format(recall_train), "train_f1=", "{:.5f}".format(f1_train), "train_acc=",
              "{:.5f}".format(acc_train),
              "val_recall=", "{:.5f}".format(recall_val), "val_f1=", "{:.5f}".format(f1_val), "val_acc=",
              "{:.5f}".format(acc_val))

        # 将特征、边索引和标签移动到GPU上
        new_feature = new_feature.to(device)
        new_edge_index = new_edge_index.to(device)
        new_labels = new_labels.to(device)

    # print("Test Recall: ", test_recall)
    print("Test Accuracy: ", test_acc)
    print("Test F1: ", test_f1)
    print("Test precision: ", test_pre)
    print("Test AUC: ", test_AUC)
    new_data.append(Data(x=new_feature, edge_index=new_edge_index, y=new_labels.view(1,-1)))

# 保存列表到 .pt 文件
torch.save(new_data, './Graph_2.0/573_train.pt')