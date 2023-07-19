#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from GNN_models import *
from anj_util import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import collections
from sklearn.metrics import precision_recall_fscore_support

def get_class_wise_train_idx(idx_train, class_labels):
    class_wise_train_idx_dic = {}
    for node in idx_train:
        s = int(class_labels[node].cpu().detach().numpy())
        if s not in class_wise_train_idx_dic:
            class_wise_train_idx_dic[s] = [node]
        else:
            class_wise_train_idx_dic[s].append(node)
    return class_wise_train_idx_dic

def RunExp(args, dataset, data, Net, percls_trn, val_lb, seed, file1):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[idx_train]#[data.train_mask]
        if args.iplevel in ['pwkk', 'dwkk', 'ransample']:
            nll = F.nll_loss(out, changed_labels[idx_train])
        else:
            nll = F.nll_loss(out, data.y[idx_train])#[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)


    idx = np.arange(len(data.y))
    no_class = dataset.num_classes
    train_size = [2 for i in range(no_class)]
    np.random.shuffle(idx)
    idx_train = []
    count = [0 for i in range(no_class)]
    next = 0
    for i in idx:
        if count == train_size:
            break
        next += 1
        if i not in idx_train and count[data.y[i]] < train_size[data.y[i]]:
            idx_train.append(i)
            count[data.y[i]] += 1

    #print(idx_train)
    if args.iplevel:
        class_wise_train_idx_dic = get_class_wise_train_idx(idx_train, data.y)
        #print(collections.Counter(data.y.cpu().detach().numpy()))
        if args.iplevel == 'pwkk':
            idx_train, changed_labels = ipLevelIntervention_pwkk(dataset, data, class_wise_train_idx_dic, args)
        elif args.iplevel == 'dwkk':
            idx_train, changed_labels = ipLevelIntervention_dwkk(dataset, data, class_wise_train_idx_dic, args)
        elif args.iplevel == 'ransample':
            idx_train, changed_labels = ipLevelIntervention_ransample(dataset, data, class_wise_train_idx_dic, args)
        elif args.iplevel == 'metaPath':
            idx_train, changed_labels = ipLevelIntervention_metaPath(dataset, data, class_wise_train_idx_dic, args)

    #print(idx_train)

    test_size = None
    test_mask = idx[next:]
    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)
        '''
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
        '''
    #return test_acc, best_val_acc, Gamma_0

    prediction = model(data)

    if args.oplevel:
        if args.oplevel == 'pwkk':
            prediction, max_acc = olrw_parwalk(prediction, data, test_mask)
        elif args.oplevel == 'dwkk':
            prediction, max_acc = olrw_deepwalk(prediction, data, test_mask)
    #print(prediction)
    _,pred = prediction.max(dim=1)
    _correct1 = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = _correct1/len(test_mask)
    pre, rec, fsc, _ = precision_recall_fscore_support(pred[test_mask], data.y[test_mask], average='macro', zero_division=0)
    file1.write(str(seed)+","+str(np.around(pre, 2))+","+str(np.around(rec, 2))+","+str(np.around(fsc, 2))+","+str(np.around(acc, 2))+"\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'SAGE'],
                        default='SAGE')
    parser.add_argument('--iplevel', type = str, default = None, help='input level')
    parser.add_argument('--oplevel', type = str, default = None, help='output level intervention')

    args = parser.parse_args()

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'SAGE':
        Net = SAGE_Net

    dname = args.dataset
    dataset, data = DataLoader(dname)

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []

    '''
    for RP in tqdm(range(RPMAX)):

        test_acc, best_val_acc, Gamma_0 = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc, Gamma_0])

    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
    '''

    file1 = open("Dump/"+args.dataset+".csv", "a")
    seeds = [65482, 10737, 95648, 100000, 75642, 9564, 458715, 422548, 65856, 8545]
    for RP in tqdm(range(1)):
        RunExp(args, dataset, data, Net, percls_trn, val_lb,  seeds[RP], file1)
