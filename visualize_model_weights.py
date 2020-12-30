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
from main_pylightning import LitGNN
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance, LayerIntegratedGradients
from captum.attr import NeuronConductance

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

    hidden_dims = 256
    heads = 3
    model_type = 'gated'
    history_frames = 3
    future_frames= 3


    input_dim = 5*history_frames
    output_dim = 2*future_frames

    if model_type == 'gat':
        hidden_dims = round(hidden_dims/heads)
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, heads=heads, output_dim=output_dim,dropout=0.1, bn=False, bn_gat=False, feat_drop=0, attn_drop=0, att_ew=True)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=hidden_dims, out_feats=output_dim, dropout=0, gcn_drop=0, bn=False, gcn_bn=False, embedding=False)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn= False)
    

    LitGCN_sys = LitGNN( model=model, lr=1e-3, model_type=model_type,wd=0.1, alfa=0)
    LitGCN_sys = LitGCN_sys.load_from_checkpoint(checkpoint_path='/home/sandra/PROGRAMAS/DBU_Graph/logs/dbu_graph/9e9gbk63/checkpoints/'+'epoch=44.ckpt',model=LitGCN_sys.model)

    #visualize weights
    
    #a = (LitGCN_sys.model.embedding_h.weight.data).detach().cpu().numpy()
    #a= (a*255/np.max(a)).astype('uint8').T
    #b = (LitGCN_sys.model.conv1.linear_self.weight.data).detach().cpu().numpy()
    #b = (b*255/np.max(b)).astype('uint8').T
    #plt.imshow(w0_s,cmap='hot')
    #plt.colorbar()
    #plt.show()
    
    #c = (LitGCN_sys.model.conv2.linear_self.weight.data).detach().cpu().numpy()
    #c = (c*255/np.max(c)).astype('uint8').T
    #d = (LitGCN_sys.model.conv2.linear.weight.data).detach().cpu().numpy()
    #d = (d*255/np.max(d)).astype('uint8').T
    #bias0 = (LitGCN_sys.model.conv1.linear.bias.data).detach().cpu().numpy()
    #bias0 = (bias0*255/np.max(bias0)).astype('uint8')

    '''
    fig,ax=plt.subplots(2,2)
    im1=ax[0,0].imshow(a,cmap='hot')
    ax[0,0].set_title('W0s',fontsize=8)
    im2=ax[0,1].imshow(b,cmap='hot')
    ax[0,1].set_title('W0 ',fontsize=8)
    im3=ax[1,0].imshow(c,cmap='hot')
    ax[1,0].set_title('w1s',fontsize=8)
    im4=ax[1,1].imshow(d,cmap='hot')
    ax[1,1].set_title('w1',fontsize=8)
    fig.colorbar(im1,ax=ax[0,0])
    fig.colorbar(im2,ax=ax[0,1])
    fig.colorbar(im3,ax=ax[1,0])
    fig.colorbar(im4,ax=ax[1,1])
    plt.show()
    '''
    #FORWARD
    test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, test=True, grip_model=False)  
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)
    iter_dataloader = iter(test_dataloader)
    graph, output_mask,snorm_n, snorm_e,track_info,mean_xy = next(iter_dataloader)

    while (track_info[0,0,0]!=8 or track_info[0,2,1]<650):
        graph, output_mask,snorm_n, snorm_e,track_info,mean_xy = next(iter_dataloader)
    print('Rec: {} Actual Frame: {}'.format(track_info[0,0,0],track_info[0,2,1]))

    LitGCN_sys.model.eval()
    model= LitGCN_sys.model
    #if model_type != 'gcn':
    #    graph.edata['w']=graph.edata['w'].float().view(graph.edata['w'].shape[0],1)
    #model(graph, graph.ndata['x'].float(),graph.edata['w'],snorm_n,snorm_e)
    
    
    #CAPTUM
    # Helper method to print importances and visualize distribution
    def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
        print(title)
        for i in range(len(feature_names)):
            print(feature_names[i], ": ", '%.3f'%(importances[i]))
        x_pos = (np.arange(len(feature_names)))
        if plot:
            plt.figure(figsize=(12,6))
            plt.bar(x_pos, importances, align='center')
            plt.xticks(x_pos, feature_names, wrap=True)
            plt.xlabel(axis_title)
            plt.title(title)

    def model_forward_ig(edge_mask, graph, snorm_n, snorm_e):
        if model_type != 'gcn':
            edge_mask=edge_mask.view(edge_mask.shape[0],1)
        out = model(graph,graph.ndata['x'].float(),  edge_mask,snorm_n,snorm_e)
        return out

    def explain(data, target=0):
        input_mask = data.edata['w'].requires_grad_(True)
        ig = IntegratedGradients(model_forward_ig)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,snorm_n,snorm_e,),
                            internal_batch_size=data.edata['w'].shape[0])

        edge_mask = np.abs(mask.cpu().detach().numpy())
        if edge_mask.max() > 0:  # avoid division by zero
            edge_mask = edge_mask / edge_mask.max()
        return edge_mask
    
    def draw_graph(g, xy, track_info, edge_mask=None, draw_edge_labels=False):
        g = dgl.to_networkx(g, edge_attrs=['w'])
        node_labels = {}
        pos={}
        for u in g.nodes():
            node_labels[u] = track_info[u,2,2] #track_id
            pos[u] = xy[u].tolist()

        #pos = nx.planar_layout(g)
        #pos = nx.planar_layout(g, pos=pos)  #En pos meter dict {u: (x,y)}
        if edge_mask is None:
            edge_color = 'black'
            widths = None
        else:
            edge_color = [edge_mask[i] for i,(u, v) in enumerate(g.edges())]
            widths = [x * 10 for x in edge_color]
        nx.draw(g, pos=pos, labels=node_labels, width=widths,
                edge_color=edge_color, edge_cmap=plt.cm.Blues,
                node_color='azure')
        
        if draw_edge_labels and edge_mask is not None:
            edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}    
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                        font_color='red')
        plt.show()

    #edge_mask = explain(graph, target=5)
    #graph.edata['w'] = torch.from_numpy(edge_mask)
    #draw_graph(graph, graph.ndata['x'].float()[:,2,:2],track_info, edge_mask, draw_edge_labels=False)
    
    #2. Layer Attributions
    def model_forward(input_mask, graph, snorm_n, snorm_e):
        if model_type != 'gcn':
            edge_mask=graph.edata['w'].view(graph.edata['w'].shape[0],1)
        out = model(graph,input_mask,  edge_mask,snorm_n,snorm_e)
        return out
    input_mask = graph.ndata['x'].requires_grad_(True)
    cond = LayerIntegratedGradients(model_forward, model.GatedGCN1)
    cond_vals = cond.attribute(input_mask, target=5,
                            additional_forward_args=(graph,snorm_n,snorm_e,))
    cond_vals = cond_vals.detach().numpy()
    visualize_importances(range(64),np.mean(cond_vals, axis=0),title="Average Neuron Importances", axis_title="Neurons")
    
    '''
    #PCA
    S=np.cov(b.T) #asume obs en las columnas
    autovalores, autovectores =np.linalg.eigh(S)
    #assert sum(autovalores) == np.trace(S)
    prop = (autovalores / np.trace(S)) * 100
    y = sorted(prop, reverse=True)
    plt.plot(y, marker='x')
    plt.title('grafico de sedimentacion')
    plt.xlabel('componentes principales (Z)')
    plt.ylabel('porcentaje de variacion')
    plt.show()


    Sb=np.cov(b)
    autovaloresb, autovectoresb =np.linalg.eigh(Sb)
    #assert sum(autovalores) == np.trace(S)
    propb = (autovaloresb / np.trace(Sb)) * 100
    yb = sorted(propb, reverse=True)
    plt.plot(yb, marker='x')
    plt.title('grafico de sedimentacion')
    plt.xlabel('componentes principales (Z)')
    plt.ylabel('porcentaje de variacion')
    plt.show()

    autovaloresr=autovalores[[4,3,2]]
    autovectores=autovectores[:,[4,3,2]]
    #all = autovactores
    '''