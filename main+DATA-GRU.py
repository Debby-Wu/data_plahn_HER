#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:39:52 2023
1.BayesianOptimization 调节超参数
@author: wudan
"""
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from pyts.image import GramianAngularField
# from pyts.approximation import PiecewiseAggregateApproximation
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, roc_curve
from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from bayes_opt import BayesianOptimization
from torch.autograd import Variable
import warnings


# -----------------!!!GAF (patient_num, time_num*time_num)
# GAF(all_feature_impute , delta_t , all_missingdata_indicator )
# 会缩放到[-1,1]之间
# -----------------

# def gafPatient(feature, min_size):
#     gaf_feature = []
#     scaler = MinMaxScaler(feature_range=(0,1)) 
#     # scaler1 = StandardScaler()
#     for patient in range(0, len(feature)):
#         patient_records = np.array(feature[patient]).reshape(-1,1).T #(1,n)
        
#         #输入(n,1)
#         # gaf_feature.append(scaler.fit_transform(patient_records.T))#原始数据
#         #输入(1,n)
#         # gaf_feature.append(scaler.fit_transform(patient_records.T).T)#原始数据
#         #输入(1,min_size)
#         paa = PiecewiseAggregateApproximation(window_size=None, output_size=min_size)
#         patient_records_paa = paa.fit_transform(patient_records)
#         patient_records_scale = scaler.fit_transform(patient_records_paa.T)
#         gaf_feature.append(patient_records_scale.T)#原始数据

#     return gaf_feature


# def gafFeature(all_feature, min_size):
#     gaf_all_feature = [] 
#     for f in range(0, len(all_feature)):
#         temp_f = list(all_feature[f])
#         gaf_all_feature.append(gafPatient(temp_f, min_size))
        
#     return gaf_all_feature

# ----------------->>>>>>>>>>
# 返回 gaf_delta_t, gaf_all_feature_impute, gaf_all_missingdata_indicator



# -----------------!!! 构建特征图x（patient_num, feature_num, time_num*time_num）和 Data loader
# 1.x
# 2.DataLoader, train_loader, test_loader
# -----------------

# -----------------1.patient( deta t + feature + missing data indicator)
# def featureMap(gaf_delta_t, gaf_all_feature_impute, gaf_all_missingdata_indicator):
# def featureMap(gaf_delta_t, gaf_all_feature_impute, gaf_all_missingdata_indicator):
#     x = []

#     for patient in range(0, len(gaf_delta_t)):
#         patient_feature = []
#         patient_feature.append(gaf_delta_t[patient][0,:])
        
#         for feature in range(0, len(gaf_all_feature_impute)):
#             patient_feature.append(list(gaf_all_feature_impute[feature][patient][0,:]))
            
#         for feature in range(0, len(gaf_all_missingdata_indicator)):
#             patient_feature.append(list(gaf_all_missingdata_indicator[feature][patient][0,:]))
        
#         #x(time_stamp, feature_size)
#         x.append(np.array(patient_feature).T)

#     return x

# -----------------
# 2.构建Dataloader, train_loader & test_loader
# -----------------
class MyDataset(Dataset):
    def __init__(self, t, x, m, y):
        self.values = x
        self.missing = m
        self.deltas = t
        self.labels = y
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, index):
        value = self.values[index].clone().detach() #相同大小
        missing = self.missing[index].clone().detach()
        delta = self.deltas[index].clone().detach()
        label = self.labels[index].clone().detach()

        return delta, value, missing, label



# def train_test_dataSave():
#     inputData = np.load('inputData_xy.npz', allow_pickle=True)
#     x = inputData['arr_0'].tolist()
#     y = inputData['arr_1'].tolist()
    
#     # x = torch.tensor(x)
#     # x = x.to(torch.float32)
#     # y = torch.tensor(np.array(y).reshape(-1,1), dtype = torch.float32 )

#     # -----------------2.构建Dataloader, train_loader & test_loader
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify = y)
#     x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify = y_train)
    
#     np.savez('train_data.npz',  x_train, y_train)
#     np.savez('valid_data.npz',  x_valid, y_valid)
#     np.savez('test_data.npz',  x_test, y_test)
      
    

def train_test_dataset(inputData):
    x = inputData['arr_0']
    y = inputData['arr_1']
    x = torch.tensor(x)
    x = x.to(torch.float32)
    y = torch.tensor(np.array(y).reshape(-1,1), dtype = torch.float32 )
    # dataSet = MyDataset(x[:,:,:1], x[:,:,1:51], x[:,:,51:101], y)
    # dataSet = MyDataset(x[:,:,:1], x[:,:,1:53], x[:,:,53:105], y)
    dataSet = MyDataset(x[:,:,:1], x[:,:,1:97], x[:,:,97:193], y)
    
    return dataSet

# ----------------->>>>>>>>>>
# 返回 train_loader, valid_loader, test_loader

class GRU_D(nn.Module):
    def __init__(self, feature_size, hidden_size, num_classes):
        super(GRU_D, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.c1 = torch.tensor(1, dtype=torch.float32).to(device)
        self.ce = torch.tensor(2.7813, dtype=torch.float32).to(device)
        
        self.ct = nn.Linear(feature_size, feature_size)
        self.xa = nn.Linear(feature_size * 2, feature_size)
        
        self.W_all = nn.Linear(feature_size, hidden_size * 2)
        self.U_all = nn.Linear(hidden_size, hidden_size * 2)
        
        self.Wh = nn.Linear(feature_size + hidden_size, hidden_size)

        # #FC
        # self.W_fc = torch.rand(self.hidden_size, self.num_classes)
        # self.b_fc = torch.rand(self.num_classes)
        #第一种，取最后一个t的h
        self.fc = nn.Linear(hidden_size, num_classes)
        #第二种，所有输出flatten
        # self.fc = nn.Linear(hidden_size * 3, num_classes)
        
        
    def forward(self, t, x, m): 
        #input(b, t, f)
        batch_size, seq_len, feature_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        # h = torch.rand(batch_size, self.hidden_size)

        outputs = []
        d = torch.var(m * x)
        for s in range(seq_len):
            # d = torch.var(m[:, s, :] * x[:, s, :])
            # ut (b ,f)
            ut = []
            for b in range(batch_size):
                u = []
                cov = torch.cov((1 - m[b, s:s+1, :]) * x[b, s:s+1, :]) - torch.cov(x[b, s:s+1, :]) * (torch.cov(m[b, s:s+1,] * x[b, s:s+1, :]) + d*d)
                for f in range(feature_size):
                    if m[b, s, f] == 1:
                        u.append(0)
                    else:
                        u.append(cov)
                ut.append(u)
            ut = torch.tensor(ut)
             # 1.x_ut (b, f)
            ct = 1 - ut
            a_ut = torch.sigmoid(self.ct(ct.to(device)))
            x_ut = x[:, s, :] * a_ut.to(device)
            
            # x_st
            c_st01 = (ct - 0.5).floor()
            x_st0true = x[:, s, :] * c_st01.to(device)
            # TGRU(x_st0true)
            x_stdeep = x_st0true * (self.c1/torch.log(self.ce + t[:, s, :]))
                                 
            a_st01 = (a_ut - 0.5).floor()
            a_st0true = x[:, s, :] * a_st01.to(device)
            # TGRU(a_st0true)
            a_stdeep = a_st0true * (self.c1/torch.log(self.ce + t[:, s, :]))
            # 2.x_st (b, f)
            x_st = x_stdeep * a_stdeep
            
            # x_adj
            x_adj = torch.cat((x_ut, x_st), 1)
            x_adj = torch.relu(self.xa(x_adj.to(device)))


            outs = self.W_all(x_adj) + self.U_all(h)
            z, r = torch.chunk(outs, 2, 1)
            z = torch.sigmoid(z)
            r = torch.sigmoid(r)
            
            h_tilde = torch.tanh(self.Wh(torch.cat((x_adj, r*h), 1)))
            h = (1 - z) * h +z * h_tilde
            
            outputs.append(h)
        
        # out = torch.sigmoid(self.fc(outputs[-1]))
        
        outputs = torch.stack(outputs, 1)
        #第一种，取最后一个t的h
        out = self.fc(outputs[:, -1, :])
        out = torch.sigmoid(out)

        #第二种，所有输出flatten
        # out = torch.sigmoid(self.fc(outputs.view(batch_size, -1)))
        
        return out

  
        
# -----------------train, test, performance
# 
# -----------------

def calculate_performance(phase, y, y_score, y_pred):
    # Calculate Evaluation Metrics
    # acc = accuracy_score(y_pred, y)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        # sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        # sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        acc = (tp + tn) / total
        p, r, t = precision_recall_curve(y, y_score)
        auprc = np.nan_to_num(metrics.auc(r, p))
    # spec = np.nan_to_num(tn / (tn + fp))
    # balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
        fscore = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))
        fscore = 2 * (prec * recall)/(prec + recall)
    

    try:
        fpr, tpr, thersholds = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_score)
        # if phase == 'test':
        #     plt.plot(fpr, tpr, 'k--', label = 'ROC (area = {:.2f})'.format(auc), lw = 2)
        #     plt.legend(loc = 'lower right')
        #     plt.xlim([-0.05, 1.05])
        #     plt.ylim([-0.05, 1.05])
        #     plt.xlabel('FPR')
        #     plt.ylabel('TPR')
        #     plt.title('ROC Curve')
        #     plt.show()
    except ValueError:
        auc = 0

    # return auc, auprc, acc, balacc, sen, spec, prec, recall
    return auc, auprc, acc, prec, recall, fscore
 


def train(epochs, batch_size, learning_rate, hidden_size):
    epochs = int(epochs)
    batch_size = int(batch_size)
    hidden_size = int(hidden_size)

    input_size = 96
        
    inputData = np.load('./train_data.npz', allow_pickle=True)
    train_data = train_test_dataset(inputData)
    inputData = np.load('./valid_data.npz', allow_pickle=True)
    valid_data = train_test_dataset(inputData)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True,  drop_last = False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle= False)
    
    model = GRU_D(input_size, hidden_size, 1)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(list(model.parameters()), lr = learning_rate) #0.01， 0.05, 0.1
    for epoch in range(1, epochs+1):
        # 模型训练状态
        model.train()
        train_loss = 0
    
        for delta, value, missing, label in train_loader:
            delta = delta.to(device)
            value = value.to(device)
            missing = missing.to(device)
            label = label.to(device)
            # 1.梯度置零
            optimizer.zero_grad()
            # 2.计算loss
            output = model(delta, value, missing).to(torch.float32)
            loss = criterion(output, label)
            train_loss += loss.item() * value.size(0)
            # loss.requires_grad_(True)
            # 3.反向传播
            loss.backward()
            # 4.梯度更新
            optimizer.step()
            
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f} ---------------------'.format(epoch, train_loss))
        test('valid', epoch, valid_loader, model, criterion)
    
    inputData = np.load('./test_data.npz', allow_pickle=True)
    test_data = train_test_dataset(inputData)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = False)
    # auc = test('test', epoch, test_loader, model, criterion)
    auc, auprc, acc, prec, recall, fscore = test('test', epoch, test_loader, model, criterion)

    # return auc
    return auc, auprc, acc, prec, recall, fscore



def selThrehold(y, y_scores):
    fpr, tpr, thersholds = roc_curve(y, y_scores)
    threshold = thersholds[np.argmax(tpr - fpr)]
    
    return threshold


def test(phase, epoch, test_loader, model, criterion):
    # 模型验证/测试状态
    model.eval()
    val_loss = 0
    
    gt_labels = []
    pred_scores = []
    pred_labels = []
    
    with torch.no_grad():
        for delta, value, missing, label in test_loader:
            delta = delta.to(device)
            value = value.to(device)
            missing = missing.to(device)
            label = label.to(device)

            gt_labels.append(label.to('cpu').detach().numpy())
            
            output = model(delta, value, missing).to(torch.float32)
            loss = criterion(output, label)
            val_loss += loss.item() * value.size(0)
            
            threshold = selThrehold(label.to('cpu').detach().numpy(), output.to('cpu').detach().numpy())
            pred_label = output.ge(threshold).float().to('cpu').detach().numpy()
            
            pred_scores.append(output.to('cpu').detach().numpy())
            pred_labels.append(pred_label)

    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_scores, pred_labels= np.concatenate(gt_labels), np.concatenate(pred_scores), np.concatenate(pred_labels)
    auc, auprc, acc, prec, recall, fscore = calculate_performance(phase, gt_labels, pred_scores, pred_labels)
    if phase == 'valid':
        print('Epoch: {} \tValidation Loss: {:.4f}, AUC: {:.4f}, AUPRC: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Recall: {:.4f}, F-Score: {:.4f}'.format(epoch, val_loss, auc, auprc, acc, prec, recall, fscore))
    else:
        print('Epoch: {} \tTest Loss: {:.4f}, AUC: {:.4f}, AUPRC: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Recall: {:.4f}, F-Score: {:.4f}'.format(epoch, val_loss, auc, auprc, acc, prec, recall, fscore))
    
    # return auc
    return auc, auprc, acc, prec, recall, fscore
            

if __name__ == '__main__':
    print('start')
    warnings.filterwarnings("ignore")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'epochs': (30, 60),
    #         'batch_size':(10, 60),
    #         'learning_rate':(0.01, 0.5),
    #         'hidden_size':(32, 64)
    #         }
    #     )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)
    
# {'target': 0.9612794612794613, 'params': {'batch_size': 59.967246723136924, 'epochs': 30.518450055636283, 'hidden_size': 63.989896286713176, 'learning_rate': 0.3387380140204286}}
# auc, auprc, acc, prec, recall, fscore
    auc_all = []
    auprc_all = []
    acc_all = []
    prec_all = []
    recall_all = []
    fscore_all = []
    
    for i in range(30):
        # train(epochs, batch_size, learning_rate, hidden_size)
        auc, auprc, acc, prec, recall, fscore = train(30.518450055636283, 59.967246723136924, 0.3387380140204286, 63.989896286713176)

        auc_all.append(auc)
        auprc_all.append(auprc)
        acc_all.append(acc)
        prec_all.append(prec)
        recall_all.append(recall)
        fscore_all.append(fscore)
    
    print("epoch---{}---AUC_ave: {:.4f}, AUC_std: {:.4f} -- AUPRC_ave: {:.4f}, AUPRC_std: {:.4f} -- Acc_ave: {:.4f}, Acc_std: {:.4f} -- Prec_ave: {:.4f}, Prec_std: {:.4f} -- Recall_ave: {:.4f}, Recall_std: {:.4f} -- Fscore_ave: {:.4f}, Fscore_std: {:.4f}".format(i, 
    np.mean(auc_all), np.std(auc_all), np.mean(auprc_all), np.std(auprc_all), np.mean(acc_all), np.std(acc_all), np.mean(prec_all), np.std(prec_all), np.mean(recall_all), np.std(recall_all), np.mean(fscore_all), np.std(fscore_all),))


    

