import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math

class GATLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCNLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.args = args
        self.conv1_lnc_f = GCNConv(args.l_f_nfeat, args.nhid)
        self.act1_lnc_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_lnc_f = GCNConv(args.nhid, args.nhid)
        self.act2_lnc_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_lnc_f = GCNConv(args.nhid, args.nclass)
        
        self.conv1_gene_f = GCNConv(args.g_f_nfeat, args.nhid)
        self.act1_gene_f=nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_gene_f = GCNConv(args.nhid, args.nhid)
        self.act2_gene_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_gene_f = GCNConv(args.nhid, args.nclass)
        
        
        self.conv1_lnc_m = GCNConv(args.l_f_nfeat, args.nhid)
        self.act1_lnc_m = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_lnc_m = GCNConv(args.nhid, args.nhid)
        self.act2_lnc_m = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_lnc_m = GCNConv(args.nhid, args.nclass)
        
        self.conv1_gene_m = GCNConv(args.g_m_nfeat, args.nhid)
        self.act1_gene_m=nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_gene_m = GCNConv(args.nhid, args.nhid)
        self.act2_gene_m = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_gene_m = GCNConv(args.nhid, args.nclass)
        
    def forward(self, dataset):
        x = self.act1_lnc_f(self.conv1_lnc_f(dataset['Lnc_f_features'], dataset['Lnc_f_edge_index']))
        x = self.act2_lnc_f(self.conv2_lnc_f(x, dataset['Lnc_f_edge_index']))
        x = self.conv3_lnc_f(x, dataset['Lnc_f_edge_index'])
        
        y = self.act1_gene_f(self.conv1_gene_f(dataset['Gene_f_features'], dataset['Gene_f_edge_index']))
        y = self.act2_gene_f(self.conv2_gene_f(y, dataset['Gene_f_edge_index']))
        y = self.conv3_gene_f(y, dataset['Gene_f_edge_index'])

        
        x2 = self.act1_lnc_m(self.conv1_lnc_m(dataset['Lnc_m_features'], dataset['Lnc_f_edge_index']))
        x2 = self.act2_lnc_m(self.conv2_lnc_m(x2, dataset['Lnc_m_edge_index']))
        x2 = self.conv3_lnc_m(x2, dataset['Lnc_m_edge_index'])
        
        y2 = self.act1_gene_m(self.conv1_gene_f(dataset['Gene_m_features'], dataset['Gene_f_edge_index']))
        y2 = self.act2_gene_m(self.conv2_gene_f(y2, dataset['Gene_m_edge_index']))
        y2 = self.conv3_gene_m(y2, dataset['Gene_m_edge_index'])
        
        #print(x.shape)
        #print(y.shape)
        #print(x2.shape)
        #print(y2.shape)
        return x, y, x2, y2

    
class CNNLayer(nn.Module):
    """
    Simple CNN layer
    """
    def __init__(self, args):
        super(CNNLayer, self).__init__()
        self.args = args
        self.cnn_x = nn.Conv1d(in_channels=args.l_f_nfeat,
                               out_channels=200,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=args.g_f_nfeat,
                               out_channels=200,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.globalAvgPool_x = nn.AvgPool2d((3, 3), (1, 1),padding=1)
        self.globalAvgPool_y = nn.AvgPool2d((3, 3), (1, 1),padding=1)
        
        self.fc_x = nn.Linear(in_features=200,
                             out_features=args.nclass)
        self.fc_y = nn.Linear(in_features=200,
                             out_features=args.nclass)

    def forward(self, x, x2):
        x=x.t()#print(x.shape)
        x = torch.relu(self.cnn_x(x.view(1, x.shape[0], x.shape[1])))#print(x.shape)
        x = torch.relu(torch.relu(self.globalAvgPool_x(x.view(1, x.shape[0], x.shape[1], x.shape[2]))))#print(x.shape)
        x = self.fc_x(x.view(x.shape[2], x.shape[3]).t())#print(x.shape)
   
        x2=x2.t()#print(x2.shape)
        x2 = torch.relu(self.cnn_y(x2.view(1, x2.shape[0], x2.shape[1])))#print(x2.shape)
        x2 = torch.relu(self.globalAvgPool_y(x2.view(1, x2.shape[0], x2.shape[1],x2.shape[2])))#print(x2.shape)
        x2 = self.fc_y(x2.view(x2.shape[2], x2.shape[3]).t())#print(x2.shape)
        return x, x2 
 
 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, lnc/pcg_no, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
                
 
class TransformerLayer(nn.Module): 
    """
    Simple Transformer layer
    """    
    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.args = args
        self.positionalEncodingLnc = PositionalEncoding(d_model=self.args.d_model_l, dropout=args.dropout)
        self.positionalEncodingPcg = PositionalEncoding(d_model=self.args.d_model_g, dropout=args.dropout)
        self.encoder_layer_lnc = nn.TransformerEncoderLayer(
            d_model=self.args.d_model_l, dim_feedforward=3*args.d_model_l, nhead=2, dropout=args.dropout
        )
        self.encoder_layer_pcg = nn.TransformerEncoderLayer(
            d_model=self.args.d_model_g, dim_feedforward=3*args.d_model_g, nhead=2, dropout=args.dropout
        )
        self.decoder_layer_lnc = nn.TransformerDecoderLayer(
            d_model=self.args.d_model_l, dim_feedforward=3*args.d_model_l, nhead=2, dropout=args.dropout
        )
        self.decoder_layer_pcg = nn.TransformerDecoderLayer(
            d_model=self.args.d_model_g, dim_feedforward=3*args.d_model_g, nhead=2, dropout=args.dropout
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 2),
        )

    def forward(self, dataset):
        x=dataset['Lnc_f_features']
        #print("Lnc_f_features shape--------------------")
        #print(x.shape)
        #print(x)
        
        x = x.view(-1,x.shape[0],x.shape[1])
        #print("Lnc_f_features shape after reshape--------------------")
        #print(x.shape)
        
        x=self.positionalEncodingLnc(x)
        #print("Lnc_f_features shape after positionalEncodingLnc--------------------")
        #print(x.shape)
        
        x=self.encoder_layer_lnc(x)
        #print("Lnc_f_features shape after encoder_layer_lnc--------------------")
        #print(x.shape)
        
        tgt = torch.rand(1, 308, 308)
        #if self.args.cuda:
        ##    tgt = tgt.cuda()
        #x=self.decoder_layer_lnc(tgt,x)
        #print("Lnc_f_features shape after decoder_layer_lnc--------------------")
        #print(x.shape)
        #print("*****************************************************************************************************************")
        #print("*****************************************************************************************************************")
        
        
        x2=dataset['Gene_f_features']
        #print("Gene_f_features shape--------------------")
        #print(x2.shape)
        
        x2 = x2.view(-1,x2.shape[0],x2.shape[1])
        #print("Gene_f_features shape after permute--------------------")
        #print(x2.shape)
        
        x2=self.positionalEncodingPcg(x2)
        #print("Gene_f_features shape after positionalEncoding--------------------")
        #print(x2.shape)
        
        x2=self.encoder_layer_pcg(x2)
        #print("Gene_f_features shape after encoder--------------------")
        #print(x2.shape)
        
        #tgt2 = torch.rand(1, 256, 256)
        #if self.args.cuda:
        #    tgt2 = tgt2.cuda()
        #x2=self.decoder_layer_pcg(tgt2,x2)
        #print("Gene_f_features shape after decoder--------------------")
        #print(x2.shape)
        return x, x2     
    
        
