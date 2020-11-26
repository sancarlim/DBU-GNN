import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn

class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bn=True, feat_drop=0., attn_drop=0., att_ew=False):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        if att_ew:
            self.attention_func = nn.Linear(3 * out_feats, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)

        self.feat_drop_l = nn.Dropout(feat_drop)
        self.attn_drop_l = nn.Dropout(attn_drop)
        self.bn = bn
        
        self.bn_node_h = nn.BatchNorm1d(out_feats)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_self.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_func.weight, gain=gain)
        nn.init.xavier_normal_(self.attention_func.weight, gain=gain)
    
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1) #(n_edg,hid)||(n_edg,hid) -> (n_edg,2*hid) 
        
        if self.att_ew:
           concat_z = torch.cat([edges.src['z'], edges.dst['z'], edges.data['w']], dim=-1) 
        
        src_e = self.attention_func(concat_z)  #(n_edg, 1) att logit
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        h_s = nodes.data['h_s']
        
        #ATTN DROPOUT
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h):
        with g.local_scope():
            
            #feat dropout
            h=self.feat_drop_l(h)
            
            h_in = h
            g.ndata['h']  = h 
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) #(18) -> (18) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h'] # result of graph convolution
            #h = h * snorm_n # normalize activation w.r.t. graph node size
            if self.bn:
                h = self.bn_node_h(h) # batch normalization 
            
            h = torch.relu(h) # non-linear activation
            h = h_in + h # residual connection
            
            return h #graph.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat', bn=True, feat_drop=0., attn_drop=0., att_ew=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats, bn=bn, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs))

    
class My_GAT(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, bn=True, bn_gat=True, feat_drop=0., attn_drop=0., heads=1,att_ew=False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.heads = heads
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop, bn_gat,att_ew)
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0., False,att_ew)
            self.linear1 = nn.Linear(hidden_dim, output_dim)
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=heads, bn=bn_gat, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)
            self.embedding_e2 = nn.Linear(1, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads, hidden_dim*heads, num_heads=1, bn=False, feat_drop=0., attn_drop=0., att_ew=att_ew)
            self.batch_norm = nn.BatchNorm1d(hidden_dim*heads)
            self.linear1 = nn.Linear(hidden_dim*heads, output_dim)
            
        #self.linear2 = nn.Linear( int(hidden_dim/2),  output_dim)
        
        if dropout:
            self.dropout_l = nn.Dropout(dropout)
        else:
            self.dropout_l = nn.Dropout(0.)
        self.bn = bn
        
    def forward(self, g, h,e_w,snorm_n,snorm_e):
        
        # input embedding
        h = self.embedding_h(h)  #input (N, 24)- (N,hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e
        # gat layers
        h = self.gat_1(g, h)
        if self.heads > 1:
            e = self.embedding_e2(e_w)
            g.edata['w']=e
        h = self.gat_2(g, h)  #RELU DENTRO DE LA GAT_LAYER
        
        h = self.dropout_l(h)
        if self.bn:
            h = self.batch_norm(h)
        y = self.linear1(h)  # (6,32) -> (6,2)
        #y = self.linear2(torch.relu(y))
        return y
    