import dgl
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn
import matplotlib.pyplot as plt

gcn_msg=fn.u_mul_e('h', 'w', 'm') #elemnt-wise (broadcast)
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, bn):
        super(GCNLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear = nn.Linear(in_feats, out_feats)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Dropout(0.)
        self.bn = bn
        self.bn_node_h = nn.BatchNorm1d(out_feats)
        
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
            '''
            #VISUALIZE
            A = g.adjacency_matrix(scipy_fmt='coo').toarray().astype('uint8')
            A=(A*255/np.max(A))
            plt.imshow(A,cmap='hot')
            plt.show()
            feat = feature.detach().cpu().numpy().astype('uint8')
            feat=(feat*255/np.max(feat))
            h0=g.ndata['h'].detach().cpu().numpy().astype('uint8')-feature.detach().cpu().numpy().astype('uint8')
            h0=(h0*255/np.max(h0))
            '''

            #mult W and normalization
            h = self.linear(g.ndata['h'])
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feature.dim() - 1)
            norm = torch.reshape(norm, shp)
            h = h * norm

            #VISUALIZE
            #h1=h.detach().cpu().numpy().astype('uint8')
            #h1=(h1*255/np.max(h1))
            
            h = g.ndata['h_s'] + h
            '''
            #VISUALIZE
            h2=h.detach().cpu().numpy().astype('uint8')
            h2=(h2*255/np.max(h2))
            fig,ax=plt.subplots(2,2)
            im1=ax[0,0].imshow(feat,cmap='hot')
            ax[0,0].set_title('X',fontsize=8)
            im4=ax[0,1].imshow(h0,cmap='hot')
            ax[0,1].set_title('M-X',fontsize=8)
            im2=ax[1,0].imshow(h1,cmap='hot')
            ax[1,0].set_title('M*W',fontsize=8)
            im3=ax[1,1].imshow(h2,cmap='hot')
            ax[1,1].set_title('H_lin=M*W + X*W_s',fontsize=8)
            fig.colorbar(im1,ax=ax[0,0])
            fig.colorbar(im2,ax=ax[1,0])
            fig.colorbar(im2,ax=ax[1,1])
            plt.show()
            '''
            if self.bn:
                self.bn_node_h(h)
            #h = h * (torch.ones_like(h)*snorm_n)  # normalize activation w.r.t. graph node size
            #e_w =  e_w * (torch.ones_like(e_w)*snorm_e)  # normalize activation w.r.t. graph edge size
            e_w =  e_w
            
            return h, e_w

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout, gcn_drop, bn,gcn_bn, embedding=True):
        super().__init__()
        self.embedding_h = nn.Linear(in_feats, hid_feats)
        hid1 = hid_feats if embedding else in_feats
        self.conv1 = GCNLayer(in_feats=hid1, out_feats=hid_feats, dropout=gcn_drop, bn=gcn_bn)
        self.conv2 = GCNLayer(in_feats=hid_feats, out_feats=hid_feats, dropout=False, bn=False)
        self.fc= nn.Linear(hid_feats,out_feats)
        self.gcn_drop = gcn_drop
        self.linear_dropout = nn.Dropout(dropout)

        self.batch_norm = nn.BatchNorm1d(hid_feats)
        self.bn = bn
        self.embedding = embedding

    def forward(self, graph, inputs,e_w,snorm_n, snorm_e):

        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        h = inputs.view(inputs.shape[0],-1)

        # input embedding
        if self.embedding:
            h = self.embedding_h(h)

        #Graph Conv
        h,_ = self.conv1(graph, h,e_w,snorm_n, snorm_e) 
        h = F.relu(h)
        h,_ = self.conv2(graph,h,e_w,snorm_n, snorm_e) 
        h = F.relu(h)

        h = self.linear_dropout(h)
        if self.bn:
            h = self.batch_norm(h)
        #Last linear layer    
        y = self.fc(h)
        '''
        h=h.detach().cpu().numpy().astype('uint8')
        h=(h*255/np.max(h))
        y=y.detach().cpu().numpy().astype('uint8')
        y=(y*255/np.max(y))
        fig,ax=plt.subplots(1,2)
        im1=ax[0].imshow(h,cmap='hot')
        ax[0].set_title('H',fontsize=8)
        im2=ax[1].imshow(y,cmap='hot')
        ax[1].set_title('Output',fontsize=8)
        fig.colorbar(im1,ax=ax[0])
        fig.colorbar(im2,ax=ax[1])
        plt.show()
        '''
        return y