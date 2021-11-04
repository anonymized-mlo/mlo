import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy as sp
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import sys
import math

import warnings
warnings.filterwarnings('ignore')

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_dict = pd.read_pickle('./data_dict_w_valid_fix_full_all_hparams.pkl')
data_dict.keys()

no_unsup = [x for x in data_dict['edge_features'].columns if 'eatt' in x]

train_edge_feature = data_dict['edge_features'].loc[data_dict['sigat_train'].index, no_unsup]
valid_edge_feature = data_dict['edge_features'].loc[data_dict['sigat_valid'].index, no_unsup]
test_edge_feature = data_dict['edge_features'].loc[data_dict['sigat_test'].index, no_unsup]


'''Load best model'''

gnn_names = ['sgcn_native', 'slf_native', 'sigat_native']

native_train = data_dict['{}_train_x_{}_{}'.format(gnn_names[2], <best_emb>, <best_epoch>)]
native_valid =data_dict['{}_valid_x_{}_{}'.format(gnn_names[2], <best_emb>, <best_epoch>)]
native_test = data_dict['{}_test_x_{}_{}'.format(gnn_names[2], <best_emb>, <best_epoch>)]


train_x = np.array(native_train)
valid_x = np.array(native_valid)
test_x = np.array(native_test)

train_y = data_dict['train_y']
valid_y = data_dict['valid_y']
test_y = data_dict['test_y']

print(len(native_train), native_train[0].shape)

print('--PCA to decrease size--')

from sklearn.decomposition import PCA
pca = PCA(n_components=16, svd_solver='full')
pca.fit(np.array(native_train))
train_x_ = pca.transform(np.array(train_x))
valid_x_ = pca.transform(np.array(valid_x))
test_x_ = pca.transform(np.array(test_x))

train_x_ = np.concatenate((train_x_, train_edge_feature.iloc[:, :]), axis=1)
valid_x_ = np.concatenate((valid_x_, valid_edge_feature.iloc[:, :]), axis=1)
test_x_ = np.concatenate((test_x_, test_edge_feature.iloc[:, :]), axis=1)

print(train_x_.shape, test_x_.shape)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

y_train_t = torch.tensor(train_y, dtype=torch.long).to(device)
y_valid_t = torch.tensor(valid_y, dtype=torch.long).to(device)
y_test_t = torch.tensor(test_y, dtype=torch.long).to(device)

x_train_t = torch.tensor(train_x_, dtype=torch.float).to(device)
x_valid_t = torch.tensor(valid_x_, dtype=torch.float).to(device)
x_test_t = torch.tensor(test_x_, dtype=torch.float).to(device)
print(x_train_t.shape)

x_train_t = torch.utils.data.TensorDataset(x_train_t, y_train_t)
x_valid_t = torch.utils.data.TensorDataset(x_valid_t, y_valid_t)
x_test_t = torch.utils.data.TensorDataset(x_test_t, y_test_t)

batch_size = 4
trainloader = torch.utils.data.DataLoader(x_train_t, batch_size=batch_size,
                                          shuffle=False, num_workers=0) # for windows num_workers should be zero w/ cuda

validloader = torch.utils.data.DataLoader(x_valid_t,
                                          shuffle=False, num_workers=0)

testloader = torch.utils.data.DataLoader(x_test_t,
                                          shuffle=False, num_workers=0)

class TieValence(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):

        def make_sequential(input_dim, hidden_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 2 * hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
            )

        super(TieValence, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = make_sequential(n_hidden, n_hidden)
        self.hidden2 = make_sequential(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc(x)
        x_ = self.hidden1(x)
        x = x + x_
        x_ = self.hidden2(x)
        x = x + x_
        last_x = x
        x = self.out(x).squeeze(dim=1)
        return x, last_x


n_feature = (np.array(train_x_).shape[1])
n_hidden = 16

net = TieValence(n_feature=n_feature, n_hidden=n_hidden).to(device)


opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
lf = torch.nn.BCEWithLogitsLoss()

aucs = []
t_aucs = []

auc_max = 0
t_auc_max = 0

results_max = 0
t_results_max = 0

test_best_emb = 0

factor = 1e5

for epoch in tqdm(range(500)):
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs, _ = net(inputs)
        loss = lf(outputs, labels.float())

        loss.backward()
        opt.step()
    scheduler.step()

    results = []
    t_results = []

    net.eval()
    with torch.no_grad():
        for data in validloader:
            ins, labels = data
            outputs, _ = net(ins)
            results.append(outputs.cpu().numpy())

    with torch.no_grad():
        for t_data in testloader:
            t_ins, t_labels = t_data
            t_outputs, last_x = net(t_ins)
            t_results.append(t_outputs.cpu().numpy())


        auc_score = roc_auc_score(valid_y, np.concatenate(results), average='macro')

        results_max = np.concatenate(results) if auc_score > auc_max else results_max
        t_results_max = np.concatenate(t_results) if auc_score > auc_max else t_results_max
        test_best_emb = last_x if auc_score > auc_max else test_best_emb

        auc_max = auc_score if auc_score > auc_max else auc_max

        aucs.append(auc_score)

        t_auc_score = roc_auc_score(test_y, np.concatenate(t_results), average='macro')

        t_auc_max = t_auc_score if t_auc_score > t_auc_max else t_auc_max
        t_aucs.append(t_auc_score)

print(auc_max)

plt.plot(aucs)
plt.plot(t_aucs)


'''Find thresholds'''

for x,y in zip(aucs, t_aucs):
    if x == auc_max:
        print(y)
        break


max_f1 = 0
final_thresh = 0
for i in range(101):
    threshold = np.percentile(results_max, i)
    round_results = (results_max > threshold).astype(int)
    pr, re, macro_f1_score, _ = precision_recall_fscore_support(valid_y, round_results, average='macro')
    final_thresh = threshold if macro_f1_score > max_f1 else final_thresh
    max_f1 = macro_f1_score if macro_f1_score > max_f1 else max_f1


round_test_results = (t_results_max > final_thresh).astype(int)
pr, re, macro_f1_score, _ = precision_recall_fscore_support(test_y, round_test_results, average='macro')
print("Precision ", pr*1, '\n',
      "Recall ", re*1, '\n',
      "F Score ", macro_f1_score)
