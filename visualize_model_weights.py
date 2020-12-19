import pickle
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import scipy.sparse as spp
from scipy import spatial
from dgl.data import DGLDataset
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from models.GCN import GCN 
from models.My_GAT import My_GAT
from models.Gated_GCN import GatedGCN
from models.gnn_rnn import Model_GNN_RNN
from tqdm import tqdm
import pandas as pd
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from LIT_system import LitGNN
from PIL import Image


if __name__ == "__main__":

    hidden_dims = 64
    heads = 1
    model_type = 'gcn'
    history_frames = 3
    future_frames= 3


    input_dim = 5*history_frames
    output_dim = 2*future_frames

    if model_type == 'gat':
        hidden_dims = round(hidden_dims/heads)
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, heads=heads, output_dim=output_dim,dropout=0.1, bn=False, bn_gat=False, feat_drop=0, attn_drop=0, att_ew=True)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=hidden_dims, out_feats=output_dim, dropout=0, gcn_drop=0, bn=False, gcn_bn=False)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn= False)
    

    LitGCN_sys = LitGNN(model=model, lr=1e-3, model_type=model_type,wd=0.1, history_frames=history_frames, future_frames=future_frames)
    LitGCN_sys = LitGCN_sys.load_from_checkpoint(checkpoint_path='/home/sandra/PROGRAMAS/DBU_Graph/logs/dbu_graph/dfc1ejbx/checkpoints/'+'epoch=53.ckpt',model=LitGCN_sys.model)

    #visualize weights
    
    w0_s = (LitGCN_sys.model.conv1.linear_self.weight.data).detach().cpu().numpy()
    w0_s = (w0_s*255/np.max(w0_s)).astype('uint8')
    w1_s = (LitGCN_sys.model.conv1.linear.weight.data).detach().cpu().numpy()
    w1_s = (w1_s*255/np.max(w1_s)).astype('uint8')
    #plt.imshow(w0_s,cmap='hot')
    #plt.colorbar()
    #plt.show()
    
    w0 = (LitGCN_sys.model.embedding_h.weight.data).detach().cpu().numpy()
    w0 = (w0*255/np.max(w0)).astype('uint8')
    w1 = (LitGCN_sys.model.fc.weight.data).detach().cpu().numpy()
    w1 = (w1*255/np.max(w1)).astype('uint8')
    #bias0 = (LitGCN_sys.model.conv1.linear.bias.data).detach().cpu().numpy()
    #bias0 = (bias0*255/np.max(bias0)).astype('uint8')

    
    fig,ax=plt.subplots(2,2)
    im1=ax[0,0].imshow(w0,cmap='hot')
    ax[0,0].set_title('Embedding',fontsize=8)
    im2=ax[0,1].imshow(w1,cmap='hot')
    ax[0,1].set_title('FC output',fontsize=8)
    im3=ax[1,0].imshow(w0_s,cmap='hot')
    ax[1,0].set_title('w0s',fontsize=8)
    im4=ax[1,1].imshow(w1_s,cmap='hot')
    ax[1,1].set_title('w0',fontsize=8)
    fig.colorbar(im1,ax=ax[0,0])
    fig.colorbar(im2,ax=ax[0,1])
    fig.colorbar(im3,ax=ax[1,0])
    fig.colorbar(im4,ax=ax[1,1])
    plt.show()
