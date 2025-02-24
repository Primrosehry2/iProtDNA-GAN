import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, recall_score, f1_score
from models import Sage_En
import utils

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
train_balance_path = './Graph_2.0/573_train.pt'
val_imbalance_path = './Graph_2.0/573_val.pt'
train_balance_data = torch.load(train_balance_path, map_location=device)
val_imbalance_data = torch.load(val_imbalance_path, map_location=device)
# print(data)   # Data(x=[73, 2560], edge_index=[2, 572], y=[1, 73])


print('train_data:', train_balance_data)
new_train_data = utils.process_train_dataset(train_balance_data)
# for i in new_train_data[0:2]:
#     print(i)
#     print(i.x)
# data=new_train_data[0]
# in_features=data.x.shape[1]

in_features = 2560
out_features = 64
dropout = 0.4

input_dim = 64
hidden_dim1 = 32
hidden_dim2 = 16
out_dim = 2

Model = Sage_En(in_features, out_features, dropout, input_dim, hidden_dim1, hidden_dim2, out_dim).to(device)
# out=sage_En(feature,adj)
# print(out)

crit = utils.FocalLoss(alpha=0.2, gamma=5)

# 传入优化器
optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)

enc = OneHotEncoder(sparse = False)
for epoch in range(250):
    loss_all = 0
    Model.train()
    for data in new_train_data:
        optimizer.zero_grad()
        feature = data.x.to(device)
        adj = data.edge_index.to(device)
        output = Model(feature,adj)
        # print(output)

        label = data.y.cpu().unsqueeze(1)
        label = torch.tensor(enc.fit_transform(label.numpy()), dtype=torch.float32).to(device)
        # print(type(output),type(label))
        # print(output, label)
        loss = crit(output, label)
        loss_all += loss.item()
        loss.backward()
        optimizer.step()
    print('loss_all:',loss_all)

torch.save(Model.state_dict(), './Weights/573_129_181.pt')

Model.eval()
new_val_data = utils.process_val_dataset(val_imbalance_data)

real_label = []
predict_label = []
probabilities = []  # 存储每个样本的预测概率
val_loss = 0

for data in new_val_data:
    feature = data.x.to(device)
    adj = data.edge_index.to(device)
    output = Model(feature, adj)
    # print(type(output),output.shape)   # 第一列代表负类的概率,第二列代表正类的概率

    # 获取预测概率
    # 第二个元素是正类的概率
    positive_class_probabilities = output[:, 1].cpu().detach().numpy().tolist()
    probabilities.extend(positive_class_probabilities)

    label = data.y.cpu().unsqueeze(1)
    # print("1111111111",label)   # [1,0]表示0，[0,1]表示1
    onehot_label = torch.tensor(enc.fit_transform(label.numpy()), dtype=torch.float32).to(device)
    # print("222222222222",onehot_label)

    loss = crit(output, onehot_label)
    val_loss += loss.item()
    label = label.view(-1).tolist()
    real_label += label

    # print(output)
    output = output.detach().cpu().numpy()
    pred_label = np.argmax(output, axis=1).tolist()
    predict_label += pred_label

TN, FP, FN, TP = confusion_matrix(real_label, predict_label).ravel()
spe = TN / (TN + FP)
# 计算召回率（Recall）
rec = recall_score(real_label, predict_label)
pre = TP / (TP + FP)
# 计算F1分数
f1 = f1_score(real_label, predict_label)
mcc = matthews_corrcoef(real_label, predict_label)

print(len(real_label))
print(len(probabilities))
# 计算AUC
auc = roc_auc_score(real_label, probabilities)

print('Test Set Spe: {:.4f}'.format(spe))
print('Test Set Rec: {:.4f}'.format(rec))
print('Test Set Pre: {:.4f}'.format(pre))
print('Test Set F1: {:.4f}'.format(f1))
print('Test Set MCC: {:.4f}'.format(mcc))
print('Test Set AUC: {:.4f}'.format(auc))
