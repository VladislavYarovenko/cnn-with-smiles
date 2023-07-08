#!/usr/bin/env python
# coding:utf-8

import time
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm

# featurevector size
atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo

class TupleDataset(Dataset):
    def __init__(self, *data):
        assert all(len(d) == len(data[0]) for d in data), "All data should have the same length."
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(torch.tensor(d[index]) for d in self.data)

def train_step(batch):
    model.train()
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    output = model(inputs)
    loss = F.cross_entropy(output, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
    
def test_step(batch):
    model.eval()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    output = model(inputs)
    loss = F.cross_entropy(output, targets)
    accuracy = torch.mean((torch.argmax(output, dim=1) == targets).float())
    return loss.item(), accuracy.item()

def main():
    START = time.time()

    #--------------------------
    parser = argparse.ArgumentParser(description='SMILES CNN fingerprint')    
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch. Default = 32')
    parser.add_argument('--epoch', '-e', type=int, default= 500, help='Number of sweeps over the dataset to train. Default = 500')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot. Defalt = 1')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (-1 indicates CPU). Default = -1')
    parser.add_argument('--output', '-o',  required=True, help='PATH to output')
    parser.add_argument('--input', '-i',  required=True, help='PATH to TOX21 data')
    parser.add_argument('--atomsize', '-a', type=int, default=400, help='Max length of smiles, SMILES which length is larger than this value will be skipped. Default = 400')
    parser.add_argument('--protein', '-p', required=True, help='Name of protein (subdataset)')
    parser.add_argument('--boost', type=int, default=-1, help='Augmentation rate (-1 indicates OFF). Default = -1')
    parser.add_argument('--k1', type=int, default=11, help='window-size of first convolution layer. Default = 11')
    parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer. Default = 1')
    parser.add_argument('--f1', type=int, default=128, help='No. of filters of first convolution layer. Default = 128')
    parser.add_argument('--k2', type=int, default=5, help='window-size of first pooling layer. Default = 5')
    parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer. Default = 1')
    parser.add_argument('--k3', type=int, default=11, help='window-size of second convolution layer. Default = 11')
    parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer. Default = 1')
    parser.add_argument('--f3', type=int, default=64, help='No. of filters of second convolution layer. Default = 64')
    parser.add_argument('--k4', type=int, default=5, help='window-size of second pooling layer. Default = 5')
    parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer. Default = 1')
    parser.add_argument('--n_hid', type=int, default=96, help='No. of hidden perceptron. Default = 96')
    parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class). Default = 1')

    args = parser.parse_args()
    
    print('GPU: ', args.gpu)
    print('# Minibatch-size: ', args.batchsize)
    print('# epoch: {}'.format(args.epoch))
    print('# 1st convolution: ',args.k1, args.s1, args.f1)
    print('# max-pooling: ',args.k2, args.s2)
    print('# 2nd convolution: ',args.k3, args.s3, args.f3)
    print('# max-pooling: ',args.k4, args.s4)
    print('')
    
    #-------------------------------
    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #-------------------------------
    # Loading SMILES
    print('Making Training Dataset...')
    file = os.path.join(args.input, args.protein + '_wholetraining.smiles')
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter=' ', titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    
    F_list, T_list = [], []
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, args.atomsize))
            T_list.append(int(mol.GetProp('_Name')))
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    
    data_t = torch.tensor(T_list, dtype=torch.int32).reshape(-1, args.n_out).to(device)
    data_f = torch.tensor(F_list, dtype=torch.float32).reshape(-1, args.n_out, args.atomsize, lensize).to(device)
    print(data_t.shape, data_f.shape)
    train_dataset = TupleDataset(data_f, data_t)
    
    print('Making Scoring Dataset...')
    file = os.path.join(args.input, args.protein + '_score.smiles')
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter='\t', titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    
    F_list, T_list = [], []
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
            print("SMILES is too long. This mol will be ignored.")
        else:
            F_list.append(mol_to_feature(mol, -1, args.atomsize))
            T_list.append(mol.GetProp('_Name'))            
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    data_t = torch.tensor(T_list, dtype=torch.int32).reshape(-1, 1).to(device)
    data_f = torch.tensor(F_list, dtype=torch.float32).reshape(-1, 1, args.atomsize, lensize).to(device)
    print(data_t.shape, data_f.shape)
    test_dataset = TupleDataset(data_f, data_t)
    
    #-------------------------------
    # Reset memory
    del mol, mols, data_f, data_t, F_list, T_list
    torch.cuda.empty_cache()
    
    #-------------------------------
    # Set up a neural network to train
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3, args.k4, args.s4, args.n_hid, args.n_out).to(device)
    
    #-------------------------------
    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    #-------------------------------
    # Set up a trainer
    print('Trainer is setting up...')
    
    output_dir = os.path.join(args.input, args.protein)
    os.makedirs(output_dir, exist_ok=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)
    
    
    for epoch in range(args.epoch):
        train_loss = 0.0
        for batch in train_loader:
            loss = train_step(batch)
            train_loss += loss
        
        train_loss /= len(train_loader)
        
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for batch in test_loader:
                loss, accuracy = test_step(batch)
                test_loss += loss
                test_accuracy += accuracy
        
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        
        print(f"Epoch [{epoch+1}/{args.epoch}] - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    END = time.time()
    print('Nice, your Learning Job is done. Total time is {} sec.'.format(END - START))

#-------------------------------
# Model Figure

#-------------------------------
if __name__ == '__main__':
    main()
