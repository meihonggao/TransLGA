from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import parameter_parser,load_data, metrics,Auc,get_edge_index
from models import TransLGA

def train(model,args,dataset):
    optimizer = torch.optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)                        
    model.train()
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        Lnc_output = model.forward(dataset)
        output = Lnc_output
        loss = torch.nn.MSELoss(reduction='mean')
        loss_train = loss(output[dataset['idx_train']], dataset['labels'][dataset['idx_train']])
        auc_train = Auc(output[dataset['idx_train']], dataset['labels'][dataset['idx_train']])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            Lnc_output = model.forward(dataset)            
            output = Lnc_output
        loss_val = loss(output[dataset['idx_val']], dataset['labels'][dataset['idx_val']])
        auc_val = Auc(output[dataset['idx_val']], dataset['labels'][dataset['idx_val']])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.8f}'.format(loss_train.data.item()),
              #'auc_train: {:.4f}'.format(auc_train),
              'loss_val: {:.8f}'.format(loss_val.data.item()),
              #'auc_val: {:.4f}'.format(auc_val),
              #'time: {:.4f}s'.format(time.time() - t)
              )    
        '''
        loss_values.append(loss_val.data.item())
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        #if bad_counter == args.patience:
        #    break
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    '''
  
def compute_test(model,dataset):
    model.eval()
    Lnc_output = model.forward(dataset)
    output = Lnc_output
    loss = torch.nn.MSELoss(reduction='mean')
    loss_test = loss(output[dataset['idx_test']], dataset['labels'][dataset['idx_test']])
    auc_train,aupr_train,acc_train,F1_train,Pre_train,Rec_train = metrics(output[dataset['idx_train']], dataset['labels'][dataset['idx_train']])
    auc_val,aupr_val,acc_val,F1_val,Pre_val,Rec_val = metrics(output[dataset['idx_val']], dataset['labels'][dataset['idx_val']])
    auc_test,aupr_test,acc_test,F1_test,Pre_test,Rec_test = metrics(output[dataset['idx_test']], dataset['labels'][dataset['idx_test']])
    print("Train set results:",
          "auc_train= {:.4f}".format(auc_train),
          "aupr_train= {:.4f}".format(aupr_train),
          "acc_train= {:.4f}".format(acc_train),
          "F1_train= {:.4f}".format(F1_train),
          "Pre_train= {:.4f}".format(Pre_train),
          "Rec_train= {:.4f}".format(Rec_train)
          )
    print("Val set results:",
          "auc_val= {:.4f}".format(auc_val),
          "aupr_val= {:.4f}".format(aupr_val),
          "acc_val= {:.4f}".format(acc_val),
          "F1_val= {:.4f}".format(F1_val),
          "Pre_val= {:.4f}".format(Pre_val),
          "Rec_val= {:.4f}".format(Rec_val)
          )
    print("Test set results:",
          #"loss_test= {:.4f}".format(loss_test.data.item()),
          "auc_test= {:.4f}".format(auc_test),
          "aupr_test= {:.4f}".format(aupr_test),
          "acc_test= {:.4f}".format(acc_test),
          "F1_test= {:.4f}".format(F1_test),
          "Pre_test= {:.4f}".format(Pre_test),
          "Rec_test= {:.4f}".format(Rec_test)
          )

def main():
    # Training settings
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load data
    dataset = load_data()
    # Model and optimizer
    args.l_f_nfeat=dataset['Lnc_f_features'].shape[1]
    args.g_f_nfeat=dataset['Gene_f_features'].shape[1]
    args.l_m_nfeat=dataset['Lnc_m_features'].shape[1]
    args.g_m_nfeat=dataset['Gene_m_features'].shape[1]
    args.d_model_l=dataset['Lnc_f_features'].shape[0]
    args.d_model_g=dataset['Gene_f_features'].shape[0]
    #args.nclass=dataset['Gene_f_features'].shape[1]
    model = TransLGA(args)
        
    if args.cuda:
        model.cuda()
        dataset['weight_matrix'] = dataset['weight_matrix'].cuda()
        dataset['Lnc_f_edge_index'] = dataset['Lnc_f_edge_index'].cuda()
        dataset['Lnc_f_adj'] = dataset['Lnc_f_adj'].cuda()
        dataset['Lnc_f_features'] = dataset['Lnc_f_features'].cuda()
        dataset['Gene_f_edge_index'] = dataset['Gene_f_edge_index'].cuda()
        dataset['Gene_f_adj'] = dataset['Gene_f_adj'].cuda()
        dataset['Gene_f_features'] = dataset['Gene_f_features'].cuda()
        dataset['Lnc_m_edge_index'] = dataset['Lnc_m_edge_index'].cuda()
        dataset['Lnc_m_adj'] = dataset['Lnc_m_adj'].cuda()
        dataset['Lnc_m_features'] = dataset['Lnc_m_features'].cuda()
        dataset['Gene_m_edge_index'] = dataset['Gene_m_edge_index'].cuda()
        dataset['Gene_m_adj'] = dataset['Gene_m_adj'].cuda()
        dataset['Gene_m_features'] = dataset['Gene_m_features'].cuda()
        dataset['labels'] = dataset['labels'].cuda()
        dataset['idx_train'] = dataset['idx_train'].cuda()
        dataset['idx_val'] = dataset['idx_val'].cuda()
        dataset['idx_test'] = dataset['idx_test'].cuda()
        #dataset['Lnc_f_edge_index'], dataset['Lnc_f_adj'], dataset['Lnc_f_features'], dataset['Gene_f_edge_index'], dataset['Gene_f_adj'], dataset['Gene_f_features'], dataset['labels'] = Variable(dataset['Lnc_f_edge_index']), Variable(dataset['Lnc_f_adj']), Variable(dataset['Lnc_f_features']), Variable(dataset['Gene_f_edge_index']), Variable(dataset['Gene_f_adj']), Variable(dataset['Gene_f_features']), Variable(dataset['labels'])
    # Train model
    train(model,args,dataset)
    # Testing
    compute_test(model,dataset)

if __name__ == "__main__":
    main()