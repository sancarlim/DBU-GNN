import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch.conv.gatconv import edge_softmax, Identity, expand_as_pair
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader

def collate_batch(samples):
    graphs, masks, feats, gt = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()

    #masks = masks.view(masks.shape[0],-1)
    #masks= masks.view(masks.shape[0]*masks.shape[1],masks.shape[2],masks.shape[3])#.squeeze(0) para TAMAÑO FIJO
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt



class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                                'output for those nodes will be invalid. '
                                'This is harmful for some applications, '
                                'causing silent performance regression. '
                                'Adding self-loop on the input graph by '
                                'calling `g = dgl.add_self_loop(g)` will resolve '
                                'the issue. Setting ``allow_zero_in_degree`` '
                                'to be `True` when constructing this module will '
                                'suppress the check and let the code run.')

            
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats,  relu=True, feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        self.relu = relu
        if att_ew:
            self.attention_func = nn.Linear(3 * out_feats, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.feat_drop_l = nn.Dropout(feat_drop, inplace=False)
        self.attn_drop_l = nn.Dropout(attn_drop, inplace=False)   
        self.res_con = res_connection
        self.reset_parameters()
      
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        
        nn.init.kaiming_normal_(self.linear_self.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_func.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.attention_func.weight, a=0.01, nonlinearity='leaky_relu')
        
    
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
        #Attention score
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h,snorm_n):
        with g.local_scope():
            h_in = h.clone()
            g.ndata['h']  = h 
            #feat dropout
            h=self.feat_drop_l(h)
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h =  g.ndata['h'] #+g.ndata['h_s'] 
            #h = h * snorm_n # normalize activation w.r.t. graph node size
            if self.relu:
                h = torch.relu(h) # non-linear activation
            if self.res_con:
                h = h_in + h # residual connection           
            return h #graph.ndata.pop('h') - another option to g.local_scope()


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, relu=True, merge='cat',  feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection))
        self.merge = merge

    def forward(self, g, h, snorm_n):
        head_outs = [attn_head(g, h,snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs))

    
class My_GAT(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, bn=True, feat_drop=0., attn_drop=0., heads=1,att_ew=False, res_weight=True, res_connection=True, ew_type=False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.embedding_e = nn.Linear(2, hidden_dim) if  ew_type else nn.Linear(1, hidden_dim)
        self.heads = heads
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop,att_ew, res_weight=res_weight, res_connection=res_connection ) #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu) 
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0.,att_ew, res_weight=res_weight, res_connection=res_connection )  #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu)
            self.linear1 = nn.Linear(hidden_dim, output_dim)          
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, res_weight=res_weight, merge='average', res_connection=res_connection , num_heads=heads,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew) #GATConv(hidden_dim, hidden_dim, heads,feat_drop, attn_drop,residual=True, activation='relu')
            self.embedding_e2 = nn.Linear(2, hidden_dim*heads) if ew_type else nn.Linear(1, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads, hidden_dim*heads, merge='cat', res_weight=res_weight, res_connection=res_connection ,num_heads=1, feat_drop=0., attn_drop=0., att_ew=att_ew) #GATConv(hidden_dim*heads, hidden_dim*heads, heads,feat_drop, attn_drop,residual=True, activation='relu')
            self.linear1 = nn.Linear(hidden_dim*heads, output_dim)

        if dropout:
            self.dropout_l = nn.Dropout(dropout, inplace=False)
        else:
            self.dropout_l = nn.Dropout(0.)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.embedding_e.weight)       
        if self.heads > 1:
            nn.init.xavier_normal_(self.embedding_e2.weight)
    
    def forward(self, g, feats,e_w,snorm_n,snorm_e):
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.contiguous().view(feats.shape[0],-1)
        # input embedding
        h = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e
        # gat layers
        h = self.gat_1(g, h,snorm_n) 
        if self.heads > 1:
            e = self.embedding_e2(e_w)
            g.edata['w']=e
        h = self.gat_2(g, h, snorm_n)  #BN Y RELU DENTRO DE LA GAT_LAYER
        h = self.dropout_l(h)
        y = self.linear1(h)
        return y
    
if __name__ == '__main__':

    history_frames = 3
    future_frames = 3
    hidden_dims = 256
    heads = 1

    test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type='gat')  #1754
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
     
    input_dim = 5*history_frames
    output_dim = 2*future_frames 

    hidden_dims = round(hidden_dims / heads) 
    model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, heads=heads, dropout=0.1, bn=True, feat_drop=0., attn_drop=0., att_ew=True, res_connection=False, res_weight=False)
    
    iter_dataloader = iter(test_dataloader)
    graph, masks, snorm_n, snorm_e, feats, labels = next(iter_dataloader)
    edge_mask=graph.edata['w'].view(graph.edata['w'].shape[0],1)
    out = model(graph,feats,  edge_mask,snorm_n,snorm_e)
