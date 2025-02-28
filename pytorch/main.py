#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNetDataset
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNetDataset(root=args.dataset_path, split='trainval', npoints=args.num_points), num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNetDataset(root=args.dataset_path, split='test', npoints=args.num_points, data_augmentation=False), num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        if (epoch % 5 == 0):
            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                test_loss*1.0/count,
                                                                                test_acc,
                                                                                avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
    
    # Printing each class accuracy after training.
    from collections import defaultdict
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Initialize per-class accuracy tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Set model to evaluation mode
    model.eval()
    
    total_correct = 0
    total_testset = 0
    test_pred = []
    test_true = []

    # Initialize loss tracking
    count = 0
    test_loss = 0.0

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            # Ensure label shape is correct before indexing
            if label.dim() > 1:  # If label is 2D, extract column 0
                label = label[:, 0]

            # Transpose input data (ensure it's required)
            data = data.transpose(2, 1)

            # Move data to CUDA
            data, label = data.cuda(), label.cuda()

            # Forward pass
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]

            # Update loss tracking
            batch_size = data.size(0)
            count += batch_size
            test_loss += loss.item() * batch_size

            # Store predictions and true labels for later analysis
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            # Compute correct predictions for overall accuracy
            correct = preds.eq(label).cpu().sum()
            total_correct += correct.item()
            total_testset += batch_size

            # Compute per-class accuracy
            for t, p in zip(label.cpu().numpy(), preds.cpu().numpy()):
                if t == p:
                    class_correct[t] += 1
                class_total[t] += 1

    # Concatenate all predictions and labels after the loop
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    # Print overall accuracy
    overall_accuracy = 100 * total_correct / float(total_testset)
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy))

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for class_idx in sorted(class_correct.keys()):
        acc = 100 * class_correct[class_idx] / class_total[class_idx]
        print("Class {}: {:.2f}%".format(class_idx, acc))

    # Calculate precision, recall, and F1-score for class 7
    class_label = 7
    precision = precision_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)
    recall = recall_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)
    f1 = f1_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)

    print("\nMetrics for Class 7:")
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))


def test_model(args, io):
    test_loader = DataLoader(ModelNetDataset(root=args.dataset_path, split='test', npoints=args.num_points, data_augmentation=False), num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    
    # Printing each class accuracy after training.
    from collections import defaultdict
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Initialize per-class accuracy tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    total_correct = 0
    total_testset = 0
    test_pred = []
    test_true = []

    # Initialize loss tracking
    count = 0

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            # Ensure label shape is correct before indexing
            if label.dim() > 1:  # If label is 2D, extract column 0
                label = label[:, 0]

            # Transpose input data (ensure it's required)
            data = data.transpose(2, 1)

            # Move data to CUDA
            data, label = data.cuda(), label.cuda()

            # Forward pass
            logits = model(data)
            preds = logits.max(dim=1)[1]

            # Update loss tracking
            batch_size = data.size(0)

            # Store predictions and true labels for later analysis
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            # Compute correct predictions for overall accuracy
            correct = preds.eq(label).cpu().sum()
            total_correct += correct.item()
            total_testset += batch_size

            # Compute per-class accuracy
            for t, p in zip(label.cpu().numpy(), preds.cpu().numpy()):
                if t == p:
                    class_correct[t] += 1
                class_total[t] += 1

    # Concatenate all predictions and labels after the loop
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    # Print overall accuracy
    overall_accuracy = 100 * total_correct / float(total_testset)
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy))

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for class_idx in sorted(class_correct.keys()):
        acc = 100 * class_correct[class_idx] / class_total[class_idx]
        print("Class {}: {:.2f}%".format(class_idx, acc))

    # Calculate precision, recall, and F1-score for class 7
    class_label = 7
    precision = precision_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)
    recall = recall_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)
    f1 = f1_score(test_true, test_pred, labels=[class_label], average='macro', zero_division=0)

    print("\nMetrics for Class 7:")
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))

def test(args, io):
    test_loader = DataLoader(ModelNetDataset(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--dataset_path', type=str, required=True, help="dataset path")
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test_model(args, io)
