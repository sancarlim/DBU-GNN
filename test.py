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
from models.rnn_baseline import RNN_baseline
from models.RGCN import RGCN
from models.Gated_GCN import GatedGCN
from models.gnn_rnn import Model_GNN_RNN
from tqdm import tqdm
import pandas as pd
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from LIT_system import LitGNN
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance, LayerIntegratedGradients
from captum.attr import NeuronConductance
import argparse

dataset = 'ind'  #'apollo'
parser = argparse.ArgumentParser()
parser.add_argument('--goal', type=str , default='test' ,help='metrics / visualize model weights')
parser.add_argument('--recording', type=int , default=18 )
parser.add_argument('--frame', type=int , default=425 )
parser.add_argument('--target', type=int , default=5, help='Output to be predicted' )
args = parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(0)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

        
def collate_test(samples):
    graphs, masks, track_info, mean_xy, feats,gt, obj_class = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()
    track_info = torch.vstack(track_info)
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, track_info.detach().cpu().numpy(), mean_xy[0], feats, gt, obj_class[0].numpy()

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

def model_forward_ig(edge_mask, graph, feats, snorm_n, snorm_e):
    if model_type != 'gcn':
        edge_mask=edge_mask.view(edge_mask.shape[0],1)
    
    if model_type == 'rgcn':
        rel_type = graph.edata['rel_type'].long()
        norm = graph.edata['norm']
        out = model(graph, feats,edge_mask, rel_type,norm)
    else:
        out = model(graph, feats,edge_mask,snorm_n,snorm_e)

    return out

def explain(data, feats, snorm_n, snorm_e, target=1):
    input_mask = data.edata['w'].requires_grad_(True)
    ig = IntegratedGradients(model_forward_ig)
    mask = ig.attribute(input_mask, target=target,
                        additional_forward_args=(data,feats, snorm_n,snorm_e,),
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


#2. Layer Attributions
def model_forward(input_mask, graph, snorm_n, snorm_e):
    if model_type != 'gcn':
        edge_mask=graph.edata['w'].view(graph.edata['w'].shape[0],1)
    out = model(graph,input_mask,  edge_mask,snorm_n,snorm_e)
    return out

def visualize(LitGCN_sys,test_dataloader):
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
    iter_dataloader = iter(test_dataloader)
    graph, masks, snorm_n, snorm_e, track_info, mean_xy, feats, labels, obj_class = next(iter_dataloader)

    while (track_info[0,0,0]!=args.recording or track_info[0,2,1]<args.frame):
        graph, masks, snorm_n, snorm_e,track_info, mean_xy, feats, labels, obj_class = next(iter_dataloader)
    print('Rec: {} Actual Frame: {}'.format(track_info[0,0,0],track_info[0,2,1]))

    LitGCN_sys.model.eval()
    model= LitGCN_sys.model
    model_type = LitGCN_sys.model_type
    
    '''
    input_mask = feats.requires_grad_(True)
    cond = LayerIntegratedGradients(model_forward, model.GatedGCN1)
    cond_vals = cond.attribute(input_mask, target=5,
                            additional_forward_args=(graph,snorm_n,snorm_e,))
    cond_vals = cond_vals.detach().numpy()
    visualize_importances(range(64),np.mean(cond_vals, axis=0),title="Average Neuron Importances", axis_title="Neurons")
    '''
    edge_mask = explain(graph, feats, snorm_n, snorm_e, target=args.target)
    graph.edata['w'] = torch.from_numpy(edge_mask)
    draw_graph(graph, feats.float()[:,2,:2],track_info, edge_mask, draw_edge_labels=False)

    
class LitGNN(pl.LightningModule):
    def __init__(self, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gat', wd: float = 1e-1, dataset: str = 'ind', history_frames: int=3, future_frames: int=5):
        super().__init__()
        self.model= model
        self.lr = lr
        self.model_type = model_type
        self.batch_size = batch_size
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.wd = wd
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        
        self.overall_loss_time_list_vru=[]
        self.overall_long_err_list_vru=[]
        self.overall_lat_err_list_vru=[]

        #For visualization purposes
        self.pred_x_list = []
        self.pred_y_list = []
        self.gt_x_list = []
        self.gt_y_list = []
        self.feats_x_list = []
        self.feats_y_list = []
        self.track_info_list = []
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt
    
    def compute_RMSE_batch(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) # x^2+y^2 BV,T
        overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return overall_sum_time, overall_num, x2y2_error

    def compute_long_lat_error(self,pred,gt,mask):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        lateral_error = pred[:,:,0]-gt[:,:,0]
        long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return lateral_error, long_error, overall_num

    def compute_change_pos(self, feats,gt):
        gt_vel = gt.detach().clone()
        feats_vel = feats[:,:,:2].detach().clone()
        new_mask_feats = (feats_vel[:, 1:,:2]!=0) * (feats_vel[:, :-1, :2]!=0) 
        new_mask_gt = (gt_vel[:, 1:,:2]!=0) * (gt_vel[:, :-1, :2]!=0) 

        gt_vel[:, 1:,:2] = (gt_vel[:, 1:,:2] - gt_vel[:, :-1, :2]).float() * new_mask_gt.float()
        gt_vel[:, :1, :2] = (gt_vel[:, 0:1,:2] - feats_vel[:, -1:, :2]).float()
        feats_vel[:, 1:,:2] = (feats_vel[:, 1:,:2] - feats_vel[:, :-1, :2]).float() * new_mask_feats.float()
        feats_vel[:, 0, :2] = 0
        
        return feats_vel.float(), gt_vel.float()

    def training_step(self, train_batch, batch_idx):
        '''needs to return a loss from a single batch'''

        batched_graph, output_masks,snorm_n, snorm_e = train_batch
        feats = batched_graph.ndata['x'].float()
        labels= batched_graph.ndata['gt'][:,:,:2].float()
        last_loc = feats[:,-1:,:2]
        if dataset.lower() == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel, labels_vel = self.compute_change_pos(feats,labels)
            #Input pos + heading + vel
            feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)
        
        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.view(e_w.shape[0],1)

        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        overall_sum_time, overall_num, _ = self.compute_RMSE_batch(pred, labels, output_masks[:,history_frames:,:])  #(B,6)
        total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))

        # Log metrics
        #self.logger.agg_and_log_metrics({"Train/loss": total_loss.data.item()}, step=self.current_epoch)
        self.log("Sweep/train_loss",  total_loss.data.item(), on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e = val_batch
        feats = batched_graph.ndata['x'].float()
        labels= batched_graph.ndata['gt'][:,:,:2].float()
        last_loc = feats[:,-1:,:2]
        if dataset.lower() == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel,_ = self.compute_change_pos(feats,labels)
            #Input pos + heading + vel
            feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.view(e_w.shape[0],1)

        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        '''
        # Compute predicted trajs.
        for i in range(1,feats.shape[1]):
            pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-1) #BV,6,2  
        pred += last_loc
        '''
        _ , overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks[:,history_frames:,:])
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(), axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0)#T
        self.log( "Sweep/val_loss", np.sum(overall_loss_time) )
        
        mse_overall_loss_time =np.sum(np.sum(x2y2_error.detach().cpu().numpy(), axis=0)) / np.sum(np.sum(overall_num.detach().cpu().numpy(), axis=0)) 
        #self.logger.agg_and_log_metrics({'val/Loss':mse_overall_loss_time}, step= self.current_epoch) #aggregate loss for epochs


    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, track_info, mean_xy, feats, labels, obj_class = test_batch
        last_loc = feats[:,-1:,:2]
        if dataset.lower() == 'apollo':
            feats_vel,_ = self.compute_change_pos(feats,labels)
            feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats,e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        
        #For visualization purposes
        self.gt_x_list.append((labels[:,:,0].detach().cpu().numpy().reshape(-1)+mean_xy[0])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))
        self.gt_y_list.append((labels[:,:,1].detach().cpu().numpy().reshape(-1)+mean_xy[1])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))
        self.pred_x_list.append((pred[:,:,0].detach().cpu().numpy().reshape(-1)+mean_xy[0])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))  #Lista donde cada elemento array (V*T_pred) (V1,V2,V3...)
        self.pred_y_list.append((pred[:,:,1].detach().cpu().numpy().reshape(-1)+mean_xy[1])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))  
        self.track_info_list.append(track_info[:,self.history_frames:self.total_frames,:].reshape(-1,track_info.shape[-1])) # V*T_pred, 6 (recording_id,frame,id, l,w,class)
        
        _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
        long_err, lat_err, _ = self.compute_long_lat_error(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_loss_time[np.isnan(overall_loss_time)]=0
        overall_long_err = np.sum(long_err.detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_lat_err = np.sum(lat_err.detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_long_err[np.isnan(overall_long_err)]=0
        overall_lat_err[np.isnan(overall_lat_err)]=0
        #print('per sec loss:{}, Sum{}'.format(overall_loss_time, np.sum(overall_loss_time)))
        #print('per sec long_err:{}, Sum{}'.format(overall_long_err, np.sum(overall_long_err)))
        #print('per sec lat_err:{}, Sum{}'.format(overall_lat_err, np.sum(overall_lat_err)))
        for i in obj_class:
            if i == 1:
                self.overall_loss_time_list.append(overall_loss_time)
                self.overall_long_err_list.append(overall_long_err)
                self.overall_lat_err_list.append(overall_lat_err)
            elif i == 2:
                self.overall_loss_time_list_vru.append(overall_loss_time)
                self.overall_long_err_list_vru.append(overall_long_err)
                self.overall_lat_err_list_vru.append(overall_lat_err)
        self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[:1]), "test/loss_2": torch.tensor(overall_loss_time[1:2]), "test/loss_3": torch.tensor(overall_loss_time[2:]) })
        if self.future_frames == 5:
            self.log_dict({ "test/loss_4": torch.tensor(overall_loss_time[3:4]), "test/loss_5": torch.tensor(overall_loss_time[4:])})


    def on_test_epoch_end(self):
        overall_loss_time = np.array(self.overall_loss_time_list)
        avg = [sum(overall_loss_time[:,i])/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        var = [sum((overall_loss_time[:,i]-avg[i])**2)/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        print('CAR Loss avg: ',avg)
        print('CAR Loss variance: ',var)
        overall_loss_time = np.array(self.overall_loss_time_list_vru)
        avg = [sum(overall_loss_time[:,i])/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        var = [sum((overall_loss_time[:,i]-avg[i])**2)/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        print('VRU Loss avg: ',avg)
        print('VRU Loss variance: ',var)

        overall_long_err = np.array(self.overall_long_err_list)
        avg_long = [sum(overall_long_err[:,i])/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        var_long = [sum((overall_long_err[:,i]-avg[i])**2)/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        
        overall_lat_err = np.array(self.overall_lat_err_list)
        avg_lat = [sum(overall_lat_err[:,i])/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        var_lat = [sum((overall_lat_err[:,i]-avg[i])**2)/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        print('\n'.join('CAR Long avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_long, var_long))))
        print('\n'.join('CAR Lat avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_lat, var_lat))))

        overall_long_err = np.array(self.overall_long_err_list_vru)
        avg_long = [sum(overall_long_err[:,i])/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        var_long = [sum((overall_long_err[:,i]-avg[i])**2)/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        
        overall_lat_err = np.array(self.overall_lat_err_list_vru)
        avg_lat = [sum(overall_lat_err[:,i])/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        var_lat = [sum((overall_lat_err[:,i]-avg[i])**2)/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        print('\n'.join('VRU Long avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_long, var_long))))
        print('\n'.join('VRU Lat avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_lat, var_lat))))

        #For visualization purposes
        recording_id_list = [self.track_info_list[i][:,0] for i in range(len(self.track_info_list))]  # lista de 1935[n_seq] arrays(V_seq,T_pred) Recording is the same for all V, but can change in the sequence (5 frames from 2 dif recordings)
        frame_id_list =  [self.track_info_list[i][:,1] for i in range(len(self.track_info_list))] #(V_seqxT_pred)
        obj_id_list = [self.track_info_list[i][:,2] for i in range(len(self.track_info_list))] #V,T
        track_vis_dict = {
            'pred_x': np.concatenate(self.pred_x_list,axis=0),
            'pred_y': np.concatenate(self.pred_y_list,axis=0),
            'gt_x': np.concatenate(self.gt_x_list,axis=0),
            'gt_y': np.concatenate(self.gt_y_list,axis=0),
            'recording_id': np.concatenate(recording_id_list,axis=0),
            'frame_id': np.concatenate(frame_id_list,axis=0),
            'obj_id': np.concatenate(obj_id_list,axis=0)
        }

        
        df_vis = pd.DataFrame.from_dict(track_vis_dict)   #1935(xVxTpred)x5
        raw_preds = df_vis.groupby(['recording_id'])
        for csv in raw_preds:
            csv[1].to_csv('/home/sandra/PROGRAMAS/raw_data/inD/val_test_data/'+str(int(csv[0]))+'_vrus_preds.csv')

        


if __name__ == "__main__":

    hidden_dims = 256
    heads = 3
    model_type = 'gat'
    history_frames = 3
    future_frames= 3

    if dataset.lower() == 'apollo':
        test_dataset = ApolloScape_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames)  #230
    elif dataset.lower() == 'ind':
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type=model_type,  test=True)  #1935
        print(len(test_dataset))
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)  

    input_dim = 5*history_frames
    output_dim = 2*future_frames

    if model_type == 'gat':
        hidden_dims = round(hidden_dims/heads)
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, heads=heads, output_dim=output_dim,dropout=0.1, bn=True, bn_gat=False, feat_drop=0, attn_drop=0, att_ew=True)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=hidden_dims, out_feats=output_dim, dropout=0, gcn_drop=0, bn=False, gcn_bn=False)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn= False)
    elif model_type == 'rgcn':
        model = RGCN(in_dim=input_dim, h_dim=hidden_dims, out_dim=output_dim, num_rels=3, num_bases=-1, num_hidden_layers=2, embedding=True, bn=False, dropout=0.1)
    

    LitGCN_sys = LitGNN(model=model, lr=1e-3, model_type=model_type,wd=0.1, history_frames=history_frames, future_frames=future_frames)
    LitGCN_sys = LitGCN_sys.load_from_checkpoint(checkpoint_path='/home/sandra/PROGRAMAS/DBU_Graph/logs/dbu_graph/pcsb0h53/checkpoints/'+'epoch=74.ckpt',model=LitGCN_sys.model)
    LitGCN_sys.model_type = model_type

    if args.goal  == 'test':
        trainer = pl.Trainer(gpus=1, profiler=True)
        trainer.test(LitGCN_sys, test_dataloaders=test_dataloader)
    elif args.goal == 'vis':
        visualize(LitGCN_sys, test_dataloader)


    