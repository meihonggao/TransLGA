import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GATLayer,GCNLayer,CNNLayer,TransformerLayer
from utils import get_adj

class TransLGA(nn.Module):
    def __init__(self, args):
        """Dense version of TransLGA."""
        super(TransLGA, self).__init__()
        self.dropout = args.dropout
        self.args = args
        
        #GAT layer
        self.attentions = [GATLayer(1*args.l_f_nfeat, args.nhid, dropout=args.dropout, alpha=args.alpha, concat=True) for _ in range(args.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GATLayer(args.nhid * args.nheads, args.nclass, dropout=args.dropout, alpha=args.alpha, concat=True)
        
        self.attentions2 = [GATLayer(1*args.g_f_nfeat, args.nhid, dropout=args.dropout, alpha=args.alpha, concat=True) for _ in range(args.nheads)]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)
        self.out_att2 = GATLayer(args.nhid * args.nheads, args.nclass, dropout=args.dropout, alpha=args.alpha, concat=False)

        #GCN layer
        self.GCN=GCNLayer(args)
        
        #CNN layer
        self.CNN=CNNLayer(args)  

        #Transformer layer
        self.transformer=TransformerLayer(args)         

        
    def forward(self, dataset):
        l_feat, g_feat = self.transformer(dataset)
        
        #l_f_feat, g_f_feat, l_m_feat, g_m_feat = self.GCN(dataset)   
        
        
        #l_feat = torch.cat((l_f_feat, l_m_feat), 1)
        #g_feat = torch.cat((g_f_feat, g_m_feat), 1)
        
        
        #print(l_feat.shape)
        #print(g_feat.shape)
        
        '''
        l_feat=dataset['Lnc_f_features']
        g_feat=dataset['Gene_f_features']


        x = F.dropout(l_feat, self.dropout, training=self.training)
        x = torch.cat([att(x, dataset['Lnc_f_adj']) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, dataset['Lnc_f_adj']))
        
        y = F.dropout(g_feat, self.dropout, training=self.training)
        y = torch.cat([att2(y, dataset['Gene_m_adj']) for att2 in self.attentions2], dim=1)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.elu(self.out_att2(y, dataset['Gene_m_adj']))
        '''
        #print(l_feat.shape)
        #print(g_feat.shape)
        
        
        
        l_feat = l_feat.view(l_feat.shape[1],l_feat.shape[2])
        g_feat = g_feat.view(g_feat.shape[1],g_feat.shape[2])
        A1 = torch.mm(l_feat, dataset['weight_matrix'])
        A2 = torch.mm(A1,g_feat)
        A = torch.sigmoid(A2)
        return A
        
    def decoder(self, z1,z2):
        #W = torch.randn(200,200)
        #A = torch.mm(z1, W, torch.t(z2))
        A = torch.mm(z1, torch.t(z2))
        A = torch.sigmoid(A)
        return A

