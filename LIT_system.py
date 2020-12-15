
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
from models.GCN import GCN 
from models.My_GAT import My_GAT
from models.Gated_GCN import GatedGCN
from tqdm import tqdm
import random
import wandb
import pandas as pd
import pytorch_lightning as pl
import statistics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

dataset = 'ind'
history_frames = 5
future_frames= 5
total_frames = history_frames + future_frames


class LitGNN(pl.LightningModule):
    def __init__(self, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gat', wd: float = 1e-1, dataset: str = 'ind'):
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
        batched_graph, output_masks,snorm_n, snorm_e, track_info, mean_xy = test_batch
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

        #For visualization purposes
        self.gt_x_list.append((labels[:,:,0].detach().cpu().numpy().reshape(-1)+mean_xy[0])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))
        self.gt_y_list.append((labels[:,:,1].detach().cpu().numpy().reshape(-1)+mean_xy[1])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))
        self.pred_x_list.append((pred[:,:,0].detach().cpu().numpy().reshape(-1)+mean_xy[0])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))  #Lista donde cada elemento array (V*T_pred) (V1,V2,V3...)
        self.pred_y_list.append((pred[:,:,1].detach().cpu().numpy().reshape(-1)+mean_xy[1])* output_masks[:,self.history_frames:self.total_frames,:].detach().cpu().numpy().reshape(-1))  
        self.track_info_list.append(track_info[:,history_frames:total_frames,:].reshape(-1,track_info.shape[-1])) # V*T_pred, 6 (recording_id,frame,id, l,w,class)
        
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
        self.overall_loss_time_list.append(overall_loss_time)
        self.overall_long_err_list.append(overall_long_err)
        self.overall_lat_err_list.append(overall_lat_err)
        self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[:1]), "test/loss_2": torch.tensor(overall_loss_time[1:2]), "test/loss_3": torch.tensor(overall_loss_time[2:]) })
        if self.future_frames == 5:
            self.log_dict({ "test/loss_4": torch.tensor(overall_loss_time[3:4]), "test/loss_5": torch.tensor(overall_loss_time[4:])})


    def on_test_epoch_end(self):
        overall_loss_time = np.array(self.overall_loss_time_list)
        avg = [sum(overall_loss_time[:,i])/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        var = [sum((overall_loss_time[:,i]-avg[i])**2)/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
        print('Loss avg: ',avg)
        print('Loss variance: ',var)

        overall_long_err = np.array(self.overall_long_err_list)
        avg_long = [sum(overall_long_err[:,i])/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        var_long = [sum((overall_long_err[:,i]-avg[i])**2)/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        
        overall_lat_err = np.array(self.overall_lat_err_list)
        avg_lat = [sum(overall_lat_err[:,i])/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        var_lat = [sum((overall_lat_err[:,i]-avg[i])**2)/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        print('\n'.join('Long avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_long, var_long))))
        print('\n'.join('Lat avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_lat, var_lat))))

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
        print(df_vis.head())
        #df_vis.to_csv('/home/sandra/PROGRAMAS/raw_data/inD/val_test_data/22_preds_sinsolape.csv')