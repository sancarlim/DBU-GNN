import dgl
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn

gcn_msg=fn.u_mul_e('h', 'w', 'm') #elemnt-wise (broadcast)
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(GCNLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear = nn.Linear(in_feats, out_feats)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = 0.
        
    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': h}

    def forward(self, g, feature,e_w, snorm_n, snorm_e):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            
            if self.dropout:
                feature = self.dropout(feature)
            
            g.ndata['h_s']=self.linear_self(feature)
            
            #normalization
            degs = g.out_degrees().float().clamp(min=1)
            norm=torch.pow(degs,-0.5)
            shp = norm.shape + (1,)*(feature.dim() -1)
            norm = torch.reshape(norm,shp)
            feature = feature*norm
            
            #aggregate
            g.edata['w'] = e_w
            g.ndata['h'] = feature
            g.update_all(gcn_msg, self.reduce_func)
            
            #mult W and normalization
            h = self.linear(g.ndata['h'])
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feature.dim() - 1)
            norm = torch.reshape(norm, shp)
            h = h * norm
            
            h = g.ndata['h_s'] + h #Vx6xout_feats
            
            #h = h * (torch.ones_like(h)*snorm_n)  # normalize activation w.r.t. graph node size
            #e_w =  e_w * (torch.ones_like(e_w)*snorm_e)  # normalize activation w.r.t. graph edge size
            e_w =  e_w
            
            return h, e_w

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super().__init__()
        self.embedding_h = nn.Linear(in_feats, hid_feats)
        self.conv1 = GCNLayer(in_feats=hid_feats, out_feats=hid_feats)
        self.conv2 = GCNLayer(in_feats=hid_feats, out_feats=hid_feats)
        self.fc= nn.Linear(hid_feats,out_feats)
        self.dropout = dropout
        if dropout:
            self.linear_dropout = nn.Dropout(dropout)
        else:
            self.linear_dropout = 0.
    def forward(self, graph, inputs,e_w,snorm_n, snorm_e):
        # input embedding
        h = self.embedding_h(inputs)
        h,_ = self.conv1(graph, h,e_w,snorm_n, snorm_e,self.dropout) #Vx6x4 -> Vx6x32  
        h = F.relu(h)
        h,_ = self.conv2(graph,h,e_w,snorm_n, snorm_e,self.dropout)  #Vx6x2 
        h = F.relu(h)
        h = self.linear_dropout(h)
        y = self.fc(h)
        return y