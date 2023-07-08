#!/usr/bin/env python
# coding:utf-8

import time
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn import metrics
from rdkit import Chem
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm
import GPy
import GPyOpt 

#-------------------------------------------------------------
# featurevectorのサイズ
atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo
#-------------------------------------------------------------

START = time.time()

#--------------------------
parser = argparse.ArgumentParser(description='CNN fingerprint')    
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
parser.add_argument('--validation', '-v', type=int, default= 5, help='N-fold cross validation')
parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of sweeps over the dataset to train')
parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--output', '-o', required=True, help='Directory to output the result')
parser.add_argument('--input', '-i', required=True, help='Input SDFs Dataset')
parser.add_argument('--atomsize', '-a', type=int, default=400, help='max length of smiles')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--protein', '-p', required=True, help='Name of protein (subdataset)')
parser.add_argument('--boost', type=int, default=1, help='Positive sample boost')
parser.add_argument('--k1', type=int, default=5, help='window-size of first convolution layer')
parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer')
parser.add_argument('--f1', type=int, default=960, help='No. of filters of first convolution layer')
parser.add_argument('--k2', type=int, default=19, help='window-size of first pooling layer')
parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer')
parser.add_argument('--k3', type=int, default=49, help='window-size of second convolution layer')
parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer')
parser.add_argument('--f3', type=int, default=480, help='No. of filters of second convolution layer')
parser.add_argument('--k4', type=int, default=33, help='window-size of second pooling layer')
parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer')
parser.add_argument('--n_hid', type=int, default=160, help='No. of hidden perceptron')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
args = parser.parse_args()

#-------------------------------
# hypterparameter
f = open(args.output+'/'+args.protein+'/CV_log.txt', 'w')
print(args.protein)
f.write('{0}\n'.format(args.protein))

#-------------------------------
# GPU check
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

#-------------------------------
# Loading SMILEs
print('Data loading...')
file = args.input + '/' + args.protein + '_wholetraining.smiles'
f.write('Loading TOX21smiles: {0}\n'.format(file))
smi = Chem.SmilesMolSupplier(file, delimiter=' ', titleLine=False)
mols = [mol for mol in smi if mol is not None]

# Make Feature Matrix
f.write('Make FeatureMatrix...\n')
F_list, T_list = [], []
for mol in mols:
    if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
        f.write("too long mol was ignored\n")
    else:
        F_list.append(mol_to_feature(mol, -1, args.atomsize))
        T_list.append(int(mol.GetProp('_Name')))

#-------------------------------
# Setting Dataset to model
f.write("Reshape the Dataset...\n")
Mf.random_list(F_list)
Mf.random_list(T_list)

data_t = torch.tensor(T_list, dtype=torch.int32).reshape(-1, 1)
data_f = torch.tensor(F_list, dtype=torch.float32).reshape(-1, 1, args.atomsize, lensize)
f.write('{0}\t{1}\n'.format(data_t.shape, data_f.shape))

f.write('Validate the Dataset...k ={0}\n'.format(args.validation))
dataset = torch.utils.data.TensorDataset(data_f, data_t)
print(dataset[0])
if args.validation > 1:
    dataset = torch.utils.data.random_split(dataset, [1/args.validation for i in range(args.validation)])

#-------------------------------
# Reset memory
del mol, mols, data_f, data_t, F_list, T_list
gc.collect()
#-------------------------------
# 5-fold
print('Training...')
f.write('Convolutional neural network is running...\n')
v = 1
while v <= args.validation:
    print('...{0}'.format(v))
    f.write('Cross-Validation : {0}\n'.format(v))

    # Set up a neural network to train
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,
                   args.k4, args.s4, args.n_hid, args.n_out)

    # Move the model to the device
    model.to(device)

    # Set up an optimizer
    f.write('Optimizer is setting up...\n')
    optimizer = optim.Adam(model.parameters())

    # Set up a loss function
    criterion = nn.CrossEntropyLoss()

    #-------------------------------
    # Set up a trainer
    f.write('Trainer is setting up...\n')

    train_loader = DataLoader(dataset[v - 1][0], batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset[v - 1][1], batch_size=args.batchsize, shuffle=False)

    # Train the model
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Evaluate the model
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze())

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        test_loss = running_loss / len(test_loader)
        test_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{args.epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    v += 1

END = time.time()
f.write('Nice, your Learning Job is done.\n')
f.write("Total time is {} sec.\n".format(END - START))
f.close()
