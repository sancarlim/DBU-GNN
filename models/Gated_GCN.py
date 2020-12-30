
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import matplotlib.pyplot as plt
import dgl.function as fn


class GatedGCN_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh'] #n_e,256
        # e_ij = Ce_ij + Dhi + Ehj   N*B,256
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh'] #n_e,256
        edges.data['e'] = e_ij
        #VISUALIZE E
        '''
        e_ijvis=e_ij.detach().cpu().numpy().astype('uint8')
        e_ijvis=(e_ijvis*255/np.max(e_ijvis))
        plt.imshow(e_ijvis, cmap='hot')
        plt.show()
        '''
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        # sigma_ij = sigmoid(e_ij)
        torch.clamp(e.sigmoid_(), min=1e-4, max=1-1e-4) 
        sigma_ij = torch.sigmoid(e)
        # hi = Ahi + sum_j eta_ij * Bhj   
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / torch.sum(sigma_ij, dim=1)  #shape n_nodes*256
        
        #VISUALIZE M AND H SIN RESIDUAL CONNECTION, PUERTA ETA
        '''
        h0=h.detach().cpu().numpy().astype('uint8')
        h0=(h0*255/np.max(h0))
        M = torch.sum(sigma_ij * Bh_j, dim=1) / torch.sum(sigma_ij, dim=1)
        M=M.detach().cpu().numpy().astype('uint8')
        M=(M*255/np.max(M))
        fig,ax=plt.subplots(1,2)
        im1=ax[0].imshow(h0,cmap='hot',aspect='auto')
        ax[0].set_title('h',fontsize=8)
        im2=ax[1].imshow(M,cmap='hot',aspect='auto')
        ax[1].set_title('Aggregated Message',fontsize=8)
        fig.colorbar(im1,ax=ax[0])
        fig.colorbar(im2,ax=ax[1])
        plt.show()
        '''
        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        
        h_in = h # residual connection
        e_in = e # residual connection
        
        
        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution

        
        h = h * snorm_n # normalize activation w.r.t. graph node size
        e = e * snorm_e # normalize activation w.r.t. graph edge size
        
        h = self.bn_node_h(h) # batch normalization  
        e = self.bn_node_e(e) # batch normalization  
        
        h = torch.relu(h) # non-linear activation
        e = torch.relu(e) # non-linear activation
        
        h = h_in + h # residual connection
        e = e_in + e # residual connection

        #VISUALIZE E AND H
        '''
        hvis=h.detach().cpu().numpy().astype('uint8')
        hvis=(hvis*255/np.max(hvis))
        evis=e.detach().cpu().numpy().astype('uint8')
        evis=(evis*255/np.max(evis))
        fig,ax=plt.subplots(1,2)
        im1=ax[0].imshow(hvis,cmap='hot')
        ax[0].set_title('H_l+1',fontsize=8)
        im2=ax[1].imshow(evis,cmap='hot')
        ax[1].set_title('Edges_l+1',fontsize=8)
        fig.colorbar(im1,ax=ax[0])
        fig.colorbar(im2,ax=ax[1])
        plt.show()
        '''
        return h, e


class GatedGCN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, bn):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN1 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.GatedGCN2 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

        if dropout:
            self.linear_dropout = nn.Dropout(dropout)
        else:
            self.linear_dropout =  nn.Dropout(0.)

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.bn = bn
        
    def forward(self, g, inputs, e, snorm_n, snorm_e):

        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        inputs = inputs.view(inputs.shape[0],-1)

        # input embedding
        h = self.embedding_h(inputs)
        e = self.embedding_e(e)
        # graph convnet layers
        h, e = self.GatedGCN1(g, h, e, snorm_n, snorm_e)
        h, e = self.GatedGCN2(g, h, e, snorm_n, snorm_e)
        # MLP 
        h = self.linear_dropout(h)
        if self.bn:
            h = self.batch_norm(h)
        y = self.linear1(h)
        
        return y