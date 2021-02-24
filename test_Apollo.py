import pickle
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
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
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from roundD_Dataset import roundD_DGLDataset
from models.GCN import GCN 
from models.My_GAT_visualize import My_GAT_vis
from models.My_GAT import My_GAT
from models.rnn_baseline import RNN_baseline
from models.RGCN import RGCN
from models.Gated_MDN import Gated_MDN
from models.Gated_GCN import GatedGCN
from models.gnn_rnn import Model_GNN_RNN
from tqdm import tqdm
import pandas as pd
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from LIT_system import LitGNN


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
    graphs, feats , track_info, obj_class, mean_xy, masks = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = np.vstack(masks)
    feats = torch.vstack(feats)
    track_info = np.vstack(track_info)
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, snorm_n, snorm_e, track_info, feats, obj_class[0], mean_xy[0], masks



class LitGNN(pl.LightningModule):
    def __init__(self,model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gat', wd: float = 1e-1, dataset: str = 'ind', history_frames: int=6, future_frames: int=6 ):
        super().__init__()
        self.model= model
        self.lr = lr
        self.model_type = model_type
        self.batch_size = batch_size
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.wd = wd
        self.overall_loss_car=[]
        self.overall_long_err_car=[]
        self.overall_lat_err_car=[]
        
        self.overall_loss_ped=[]
        self.overall_long_err_ped=[]
        self.overall_lat_err_ped=[]

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
    
    def compute_RMSE_batch(self,pred, gt, mask, car_ids, ped_ids): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        
        x2y2_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) # x^2+y^2 BV,T
        x2y2_error_car = x2y2_error[car_ids]
        x2y2_error_ped = x2y2_error[ped_ids]
        x2y2_list = [x2y2_error, x2y2_error_car, x2y2_error_ped]

        overall_sum_all = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
        overall_sum_car = x2y2_error_car.sum(dim=-2)
        overall_sum_ped = x2y2_error_ped.sum(dim=-2)
        overall_sum_list = [overall_sum_all, overall_sum_car, overall_sum_ped]

        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        overall_num_car = mask[car_ids].sum(dim=-1).type(torch.int)
        overall_num_ped = mask[ped_ids].sum(dim=-1).type(torch.int)
        overall_num_list = [overall_num, overall_num_car, overall_num_ped]

        return overall_sum_list, overall_num_list, x2y2_list

    def compute_long_lat_error(self,pred,gt,mask, car_ids, ped_ids):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        lateral_error = pred[:,:,0]-gt[:,:,0]
        long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
        lateral_error = [lateral_error, lateral_error[car_ids], lateral_error[ped_ids]]
        long_error = [long_error, long_error[car_ids], long_error[ped_ids]]
        return lateral_error, long_error

    def compute_change_pos(self, feats):
        feats_vel = feats[:,:,:2].clone()
        new_mask_feats = (feats_vel[:, 1:,:2]!=0) * (feats_vel[:, :-1, :2]!=0) 
        feats_vel[:, 1:,:2] = (feats_vel[:, 1:,:2] - feats_vel[:, :-1, :2]).float() * new_mask_feats.float()
        feats_vel[:, 0, :2] = 0
        
        return feats_vel.float()
    
    def sample(self,pred):
        """Draw samples from a MoG.
            For Apollo Challenge we sample the most-likely 
        """
        pi, sigma, mu = pred
        categorical = Categorical(pi)
        pis = [torch.argmax(pi_n) for pi_n in pi] # pis = list(categorical.sample().data)  #B
        #Create a tensor of the same type as sigma and size B,O (Gaussian noise sample ~ N(0,I))
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_()) #B,12
        for i, idx in enumerate(pis):
            #sample from one of the modes for each agent in the batch z=mu(idx)+sigma(idx)*N(0,I) , where idx ~ pi
            sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx]) #sigma[i,idx].add(mu[i,idx]) 
        return sample


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
        
        batched_graph, snorm_n, snorm_e, track_info, feats , obj_class, mean_xy, output_masks = test_batch

        last_loc = feats[:,-1:,:2]
        feats_vel= self.compute_change_pos(feats)
        #Inputs
        #feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)
        #Inputs vel + heading + obj
        feats = torch.cat([feats_vel, feats[:,:,2:input_feat]], dim=-1)[:,1:,:]

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats,e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)


        if probabilistic:
            pred=self.sample(pred)  #N,12

        pred=pred.view(pred.shape[0],6,-1)
        
        #Compute predicted trajs
        for i in range(1,pred.shape[1]):
            pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-2) #BV,6,2 
        pred += last_loc

        #For visualization purposes
        pred_x=pred[:,:,0].detach().cpu().numpy()+mean_xy[0] #TxV
        pred_y=pred[:,:,1].detach().cpu().numpy()+mean_xy[1]  
        #self.track_info_list.append(track_info.T.reshape(3,-1)) # V*T_pred, 3 (frame,id,class)
        frame_now = track_info[0,-1,0]
        frames = [frame_now+1,frame_now+2,frame_now+3,frame_now+4,frame_now+5,frame_now+6]
        obj_ids = track_info[:,-1,1]
        obj_types = track_info[:,-1,2]
        with open('/home/sandra/PROGRAMAS/DBU_Graph/Apollo_Challenge/prediction_result.txt', 'a') as f:
            for i, frame in enumerate(frames):
                for id, type, pred_x_t,pred_y_t in zip(obj_ids, obj_types, pred_x[:,i], pred_y[:,i]):
                    print('{}'.format(' '.join(['{}'.format(x) for x in [int(frame), int(id), int(type), round(pred_x_t,3), round(pred_y_t,3)]])), sep=' ', end='\n',file=f)
                    

if __name__ == "__main__":

    hidden_dims = 512
    heads = 3
    input_feat = 5
    model_type = 'gated_mdn'
    history_frames = 6
    future_frames= 6
    probabilistic = True

    test_dataset = ApolloScape_DGLDataset(train_val='test',test=True)  #230
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test, num_workers=12)  
    print(len(test_dataloader))
    input_dim = input_feat*5  #feats vel+head+obj+mask * 5frames (remove vel[0])  #6*history_frames
    output_dim = 2*future_frames

    if model_type == 'gat':
        hidden_dims = round(hidden_dims/heads)
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, heads=heads, output_dim=output_dim,dropout=0.1, bn=False, feat_drop=0, attn_drop=0, att_ew=True)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=hidden_dims, out_feats=output_dim, dropout=0, gcn_drop=0, bn=False, gcn_bn=False)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn= True)
    elif model_type == 'gated_mdn':
        model = Gated_MDN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, dropout=0.1, bn=True)
    elif model_type == 'rgcn':
        model = RGCN(in_dim=input_dim, h_dim=hidden_dims, out_dim=output_dim, num_rels=3, num_bases=-1, num_hidden_layers=2, embedding=True, bn=False, dropout=0.1) 

    LitGCN_sys = LitGNN(model=model, lr=1e-3, model_type=model_type,wd=0.1, history_frames=history_frames, future_frames=future_frames)
    LitGCN_sys = LitGCN_sys.load_from_checkpoint(checkpoint_path= '/home/sandra/PROGRAMAS/DBU_Graph/logs/MDN/quiet-sweep-1/epoch=164-step=10394.ckpt',model=LitGCN_sys.model)
    LitGCN_sys.model_type = model_type 
    LitGCN_sys.history_frames = history_frames
    LitGCN_sys.future_frames = future_frames


    trainer = pl.Trainer(gpus=0, profiler=True)
    trainer.test(LitGCN_sys, test_dataloaders=test_dataloader)

    