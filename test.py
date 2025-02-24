import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score, recall_score, roc_curve
import models
import utils
import warnings
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_features = 2560
out_features = 64
dropout = 0.4

input_dim = 64
hidden_dim1 = 32
hidden_dim2 = 16
out_dim = 2
Model = models.Sage_En(in_features, out_features, dropout, input_dim, hidden_dim1, hidden_dim2, out_dim).to(device)
print('sage_En:', Model)  # 输出模型的结构信息

# 加载测试集并处理测试集数据
path = './Graph/129.pt'
# 读取.pt文件
test_data = torch.load(path)
new_test_data = utils.process_test_dataset(test_data)
# print(new_test_data)

# 加载模型
model_path = f'Weights/573_129_181.pt'
Model.load_state_dict(torch.load(model_path))
Model.eval()

# 将测试集传入模型
real_label = []
predict_label = []
probabilities = []

for data in new_test_data:
    feature = data.x.to(device)
    adj = data.edge_index.to(device)
    output = Model(feature, adj)
    # 获取预测概率
    positive_class_probabilities = output[:, 1].cpu().detach().numpy().tolist()
    probabilities.extend(positive_class_probabilities)
    label = data.y.unsqueeze(1).to(device)
    label = label.view(-1).tolist()
    real_label += label

    # print(output)
    # 将输出的张量移动到CPU并转换为NumPy数组
    output_cpu = output.detach().cpu()
    output_np = output_cpu.numpy()
    pred_label = np.argmax(output_np, axis=1).tolist()
    predict_label += pred_label

TN, FP, FN, TP = confusion_matrix(real_label, predict_label).ravel()
spe = TN / (TN + FP)
# 计算召回率（Recall）
rec = recall_score(real_label, predict_label)
pre = TP / (TP + FP)
# 计算F1分数
f1 = f1_score(real_label, predict_label)
mcc = matthews_corrcoef(real_label, predict_label)
# 计算AUC
auc = roc_auc_score(real_label, probabilities)

print('Test Set Spe: {:.4f}'.format(spe))
print('Test Set Rec: {:.4f}'.format(rec))
print('Test Set Pre: {:.4f}'.format(pre))
print('Test Set F1: {:.4f}'.format(f1))
print('Test Set MCC: {:.4f}'.format(mcc))
print('Test Set AUC: {:.4f}'.format(auc))

# # 绘制混淆矩阵
# ConfusionMatrixDisplay.from_predictions(real_label, predict_label)
# plt.title('Confusion Matrix')
# plt.savefig('./Plots/129confusion_matrix.png')
# plt.show()

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(real_label, probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./Plots/129_ROC.png')
plt.show()