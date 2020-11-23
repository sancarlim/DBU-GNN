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
    def __init__(self, in_feats, out_feats, feat_drop=0., attn_drop=0.):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.bn_node = nn.BatchNorm1d(out_feats)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_self.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_func.weight, gain=gain)
        nn.init.xavier_normal_(self.attention_func.weight, gain=gain)
    
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1) #(n_edg,6*64)||(n_edg,6*64) -> (n_edg,2*6*64) 
        src_e = self.attention_func(concat_z)  #(n_edg, 1) att logit
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        h_s = nodes.data['h_s']
        
        #ATTN DROPOUT
        a = self.attn_drop(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h):
        with g.local_scope():
            
            #feat dropout
            h=self.feat_drop(h)
            
            h_in = h
            g.ndata['h']  = h 
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) #(18) -> (18) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h'] # result of graph convolution
            #h = h * snorm_n # normalize activation w.r.t. graph node size
            #h = self.bn_node(h) # batch normalization 
            
            h = torch.relu(h) # non-linear activation
            h = h_in + h # residual connection
            
            return h #graph.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs))

class MLP_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, input_dim/2) 
        self.layer2 = nn.Linear(input_dim/2, input_dim/4) 
        
    def forward(self, x):
        y = x
        y = self.layer1(y)
        y = torch.relu(y)
        y = self.layer2(y)
        return y
    
    
class My_GAT(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, feat_drop=0., attn_drop=0., heads=4):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop)
        self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop)
        #self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, heads)
        #self.gat_2 = MultiHeadGATLayer(hidden_dim*heads, hidden_dim*heads, 1)
        
        self.linear1 = nn.Linear(hidden_dim, output_dim) #hidden*heads para multihead
        #self.linear2 = nn.Linear( int(hidden_dim/2),  output_dim)
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = 0.
        
    def forward(self, g, h,e_w,snorm_n,snorm_e):
        
        # input embedding
        h = self.embedding_h(h)  #input (70, 6,4) - (70, 6,32) checked
        # gat layers
        h = self.gat_1(g, h,snorm_n)
        h = self.gat_2(g, h,snorm_n)  #RELU DENTRO DE LA GAT_LAYER
        
        h = self.dropout(h)
        y = self.linear1(h)  # (6,32) -> (6,2)
        #y = self.linear2(torch.relu(y))
        return y
    