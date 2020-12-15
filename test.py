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


dataset = 'ind'  #'apollo'
history_frames = 5
future_frames= 3
total_frames = history_frames + future_frames


def collate_test(samples):
    graphs, masks, track_info, mean_xy = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    track_info = torch.vstack(track_info)
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, track_info.detach().cpu().numpy(),mean_xy[0]

if __name__ == "__main__":


    if dataset.lower() == 'apollo':
        test_dataset = ApolloScape_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames)  #230
    elif dataset.lower() == 'ind':
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, test=True)  #1935
        print(len(test_dataset))
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)  

    input_dim = 5*history_frames
    output_dim = 2*future_frames

    hidden_dims = 256
    heads = 3
    model_type = 'gated'

    if model_type == 'gat':
        hidden_dims = round(hidden_dims/heads)
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, heads=heads, output_dim=output_dim,dropout=0.1, bn=False, bn_gat=False, feat_drop=0, attn_drop=0, att_ew=True)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=hidden_dims, out_feats=output_dim, dropout=0, gcn_drop=0, bn=False, gcn_bn=False)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn= False)
    

    LitGCN_sys = LitGNN(model=model, lr=1e-3, model_type='gat',wd=0.1)
    LitGCN_sys = LitGCN_sys.load_from_checkpoint(checkpoint_path='/home/sandra/PROGRAMAS/DBU_Graph/models_checkpoints/faithful-sweep-12.ckpt',model=LitGCN_sys.model)
    trainer = pl.Trainer(gpus=1, profiler=True)

    trainer.test(LitGCN_sys, test_dataloaders=test_dataloader)
    