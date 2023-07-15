import argparse
import gc

import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from rdkit import Chem
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm

#-------------------------------------------------------------
# featurevectorのサイズ
atomInfo = 21
structInfo = 21
lensize= atomInfo + structInfo

#------------------------------------------------------------- 
def main():
    
    #引数管理
    parser = argparse.ArgumentParser(description='CNN fingerprint')    
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of max iteration to evaluate')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot')
    parser.add_argument('--validation', '-v', type=int, default= 5, help='Cross validation No.')
    parser.add_argument('--model', '-m', required=True, help='Directory to Model to evaluate')
    parser.add_argument('--data', '-s', required=True, help='Input Smiles Dataset')
    parser.add_argument('--protein', '-p', default='NR-AR', help='Name of protain what you are choose')
    parser.add_argument('--atomsize', '-a', type=int, default=400, help='max length of smiles')
    parser.add_argument('--k1', type=int, default=1, help='window-size of first convolution layer')
    parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer')
    parser.add_argument('--f1', type=int, default=1, help='No. of filters of first convolution layer')
    parser.add_argument('--k2', type=int, default=1, help='window-size of first pooling layer')
    parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer')
    parser.add_argument('--k3', type=int, default=1, help='window-size of second convolution layer')
    parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer')
    parser.add_argument('--f3', type=int, default=1, help='No. of filters of second convolution layer')
    parser.add_argument('--k4', type=int, default=1, help='window-size of second pooling layer')
    parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer')
    parser.add_argument('--n_hid', type=int, default=1, help='No. of hidden perceptron')
    parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
    
    args = parser.parse_args()
    
    file = args.data + '/' + args.protein + '_wholetraining.smiles'
    print('Loading TOX21smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter=' ', titleLine=False)
    mols = [mol for mol in smi if mol is not None]

    print('Make FeatureMatrix...')
    F_list, T_list = [], []
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, args.atomsize))
            T_list.append(mol.GetProp('_Name'))

    # Setting Dataset to model
    print("Reshape the Dataset...")
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    data_t = torch.tensor(T_list, dtype=torch.int32).reshape(-1, 1)
    data_f = torch.tensor(F_list, dtype=torch.float32).reshape(-1, 1, args.atomsize, lensize)
    print(data_t.shape, data_f.shape)
    borders = [len(data_t) * i // args.validation for i in range(args.validation + 1)]
    borders.reverse()
    print('')

    del mol, mols, F_list, T_list
    gc.collect()

    print('Evaluator is running...')

    # Set up a neural network to evaluate
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3, args.k4, args.s4, args.n_hid, args.n_out)
    model.compute_accuracy = False
    model.to(args.device)

    f = open(args.model + '/' + args.protein + '/evaluation_epoch_each.csv', 'w')

    print("epoch", "validation", "TP", "FN", "FP", "TN", "Loss", "Accuracy", "B_accuracy", "Specificity", "Precision", "Recall", "F-measure", "AUC", sep="\t")
    f.write("epoch,validation,TP,FN,FP,TN,Loss,Accuracy,B_accuracy,Specificity,Precision,Recall,F-measure,AUC\n")

    for v in range(args.validation):
        for epoch in range(args.frequency, args.epoch + 1, args.frequency):

            model.load_state_dict(torch.load(args.model + '/' + args.protein + '/model_' + str(v + 1) + '_snapshot_' + str(epoch)))
            model.eval()

            x = data_f[borders[v + 1]:borders[v]]
            y = data_t[borders[v + 1]:borders[v]]
            
            dataloader = DataLoader(TensorDataset(x, y), batch_size=200, shuffle=False)

            pred_score_tmp, loss_tmp = [], []
            count_TP, count_FP, count_FN, count_TN = 0, 0, 0, 0

            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                    pred_tmp, sr = model.predict(Variable(batch_x))
                    pred_tmp = torch.sigmoid(pred_tmp.data)
                    loss_tmp.append(model(Variable(batch_x), Variable(batch_y)).data.item())
                    pred_score_tmp.extend(pred_tmp.data.reshape(-1).tolist())

                    pred = (pred_tmp >= 0.5).int()
                    count_TP += torch.sum((batch_y == pred) & (pred == 1)).item()
                    count_FP += torch.sum((batch_y != pred) & (pred == 1)).item()
                    count_FN += torch.sum((batch_y != pred) & (pred == 0)).item()
                    count_TN += torch.sum((batch_y == pred) & (pred == 0)).item()

            loss = np.mean(loss_tmp)
            pred_score = torch.tensor(pred_score_tmp).reshape(-1, 1).cpu().numpy()
            y = y.cpu().numpy()

            Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
            Specificity = count_TN / (count_TN + count_FP)
            Precision = count_TP / (count_TP + count_FP)
            Recall = count_TP / (count_TP + count_FN)
            Fmeasure = 2 * Recall * Precision / (Recall + Precision)
            B_accuracy = (Specificity + Recall) / 2
            AUC = metrics.roc_auc_score(y, pred_score, average='weighted')

            print(epoch, v + 1, count_TP, count_FN, count_FP, count_TN, loss, Accuracy, B_accuracy, Specificity, Precision, Recall, Fmeasure, AUC, sep="\t")
            text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(
                epoch, v + 1, count_TP, count_FN, count_FP, count_TN, loss, Accuracy, B_accuracy, Specificity, Precision,
                Recall, Fmeasure, AUC)
            f.write(text)

    f.close()

#------------------------------- 
if __name__ == '__main__':
    main()
