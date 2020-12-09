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
from models.rnn_baseline import RNN_baseline
from tqdm import tqdm
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

#from LIT_system import LitGNN

dataset = 'ind'  #'apollo'
#history_frames = 3
#future_frames= 3
#total_frames = history_frames + future_frames

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()


def collate_batch(samples):
    graphs, masks = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)

    #masks = masks.view(masks.shape[0],-1)
    #masks= masks.view(masks.shape[0]*masks.shape[1],masks.shape[2],masks.shape[3])#.squeeze(0) para TAMAÑO FIJO
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e

class LitGNN(pl.LightningModule):
    def __init__(self, history_frames: int, future_frames: int, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gcn', wd: float = 1e-1):
        super().__init__()
        self.model= model
        self.lr = lr
        self.model_type = model_type
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        wandb.watch(self.model, log="all")
        
    
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

    def compute_long_lat_error(self,pred,gt,mask):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        lateral_error = pred[:,:,0]-gt[:,:,0]
        long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return lateral_error, long_error, overall_num

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
        overall_sum_time, overall_num, _ = self.compute_RMSE_batch(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])  #(B,6)
        total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))

        # Log metrics
        #self.logger.agg_and_log_metrics({"Train/loss": total_loss.data.item()}, step=self.current_epoch)
        self.log("Sweep/train_loss",  total_loss.data.item(), on_step=False, on_epoch=True)
        return total_loss
    '''
    def training_epoch_end(self, total_loss):
        self.epoch += 1
        print('|{}| Train_loss: {}'.format(datetime.now(), np.sum(self.overall_loss_train)/len(self.overall_loss_train)))
        #self.log("Train/Loss", np.sum(self.overall_loss_train)/len(self.overall_loss_train) )
        self.logger.log_metrics({"Train/loss": np.sum(self.overall_loss_train)/len(self.overall_loss_train)}, step=self.epoch)
        #wandb.log({"Train/loss": np.sum(self.overall_loss_train)/len(self.overall_loss_train)}, step=self.epoch)#, step=epoch)      
        self.overall_loss_train=[]
    '''
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
        _ , overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(), axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0)#T
        self.log( "Sweep/val_loss", np.sum(overall_loss_time) )
        
        mse_overall_loss_time =np.sum(np.sum(x2y2_error.detach().cpu().numpy(), axis=0)) / np.sum(np.sum(overall_num.detach().cpu().numpy(), axis=0)) 
        #self.logger.agg_and_log_metrics({'val/Loss':mse_overall_loss_time}, step= self.current_epoch) #aggregate loss for epochs

    '''    
    def validation_epoch_end(self, val_results):
        overall_sum_time=np.sum(self.overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
        overall_num_time =np.sum(self.overall_num_list, axis=0)
        overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame
        self.overall_num_list=[]
        self.overall_x2y2_list=[]
        
        #self.logger.log_metrics({'val/Loss':np.sum(overall_loss_time)}, step= self.epoch)
        #wandb.log({"Val/Loss": np.sum(overall_loss_time)}, step= self.epoch)
    '''
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e = test_batch
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
        # Compute predicted trajs.
        '''
        for i in range(1,feats.shape[1]):
            pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-1) #BV,6,2 
        pred += last_loc
        '''
        _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
        long_err, lat_err, _ = self.compute_long_lat_error(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_loss_time[np.isnan(overall_loss_time)]=0
        overall_long_err = np.sum(long_err.detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_lat_err = np.sum(lat_err.detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_long_err[np.isnan(overall_long_err)]=0
        overall_lat_err[np.isnan(overall_lat_err)]=0
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
        print('Loss variance: ' , var)

        overall_long_err = np.array(self.overall_long_err_list)
        avg_long = [sum(overall_long_err[:,i])/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        var_long = [sum((overall_long_err[:,i]-avg[i])**2)/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
        
        overall_lat_err = np.array(self.overall_lat_err_list)
        avg_lat = [sum(overall_lat_err[:,i])/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        var_lat = [sum((overall_lat_err[:,i]-avg[i])**2)/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
        print('\n'.join('Long avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_long, var_long))))
        print('\n'.join('Lat avg error in sec {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_lat, var_lat))))

        self.log_dict({ "test/var_s1": torch.tensor(var[0]), "test/var_s2": torch.tensor(var[1]),"test/var_s3": torch.tensor(var[2])})
        if self.future_frames == 5:
            self.log_dict({"var_s4": torch.tensor(var[3]), "var_s5": torch.tensor(var[4])})




def sweep_train():
    wandb.init() 
    config = wandb.config
    print('config: ', dict(config))
    wandb_logger = pl_loggers.WandbLogger(save_dir='./logs/')  #name=
    train_dataloader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    val_dataloader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
     
    input_dim = 5*config.history_frames
    output_dim = 2*config.future_frames

    if config.model_type == 'gat':
        hidden_dims = round(config.hidden_dims / config.heads) 
        model = My_GAT(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, heads=config.heads, dropout=config.dropout, bn=config.bn, bn_gat=config.bn_gat, feat_drop=config.feat_drop, attn_drop=config.attn_drop, att_ew=config.att_ew)
    elif config.model_type == 'gcn':
        model = model = GCN(in_feats=input_dim, hid_feats=config.hidden_dims, out_feats=output_dim, dropout=config.dropout, gcn_drop=config.gcn_drop, bn=config.bn, gcn_bn=config.gcn_bn)
    elif config.model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=config.hidden_dims, output_dim=output_dim, dropout=config.dropout, bn=config.bn)
    elif config.model_type == 'rnn':
        model = Model_GNN_RNN(input_dim=5, hidden_dim=config.hidden_dims, output_dim=output_dim, pred_length=config.future_frames, dropout=config.dropout, bn=config.bn, bn_gat=config.bn_gat, feat_drop=config.feat_drop, attn_drop=config.attn_drop, att_ew=config.att_ew)
    elif config.model_type == 'baseline':
        model = RNN_baseline(input_dim=5, hidden_dim=config.hidden_dims, output_dim=output_dim, pred_length=config.future_frames, dropout=config.dropout, bn=config.bn)

    LitGNN_sys = LitGNN(model=model, lr=config.learning_rate, model_type= config.model_type, wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    #checkpoint_callback = ModelCheckpoint(monitor='val/Loss', mode='min')
    early_stop_callback = EarlyStopping('Sweep/val_loss')
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=wandb_logger, precision=16, default_root_dir='./models_checkpoints/', callbacks=[early_stop_callback], profiler=True)  #precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20, 
    
    print("############### TRAIN ####################")
    trainer.fit(LitGNN_sys, train_dataloader, val_dataloader)   

    print("############### TEST ####################")
    trainer.test(test_dataloaders=test_dataloader)




if __name__ == '__main__':

    history_frames=5
    future_frames=5

    if dataset.lower() == 'apollo':
        train_dataset = ApolloScape_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames)  #230
    elif dataset.lower() == 'ind':
        train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames) #12281
        val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames)  #3509
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames)  #1754
        
    #train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    '''
    if model_type == 'gat':
        model = My_GAT(input_dim=18, hidden_dim=hidden_dims, output_dim=12)
    elif model_type == 'gcn':
        model = GCN(in_feats=18, hid_feats=hidden_dims, out_feats=12)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=18, hidden_dim=hidden_dims, output_dim=12)
    '''
    
    #wandb.init(project="dbu_graph", config= hyperparameters_default) 
    #config = wandb.config

    sweep_config = {
    "name": "inD baseline 5s/5s",
    "method": "grid",
    "metric": {
        'name': 'Sweep/val_loss',
        'goal': 'minimize'   
    },
    "early_terminate": {
        'type': 'hyperband',
        'min_iter': 3
    },
    "parameters": {
            "history_frames":{
                "values": [history_frames]
            },
            "future_frames":{
                "values": [future_frames]
            },
            "learning_rate":{
                #"distribution": 'uniform',
                #"max": 1e-1,
                #"min": 1e-5,
                "values": [1e-3, 1e-4]
            },
            "batch_size": {
                #"distribution": 'int_uniform',
                #"max": 512,
                #"min": 64,
                "values": [64]
            },
            "hidden_dims": {
                #"distribution": 'int_uniform',
                #"max": 512,
                #"min": 64,
                "values": [64]
            },
            "model_type": {
                "values": ['rnn', 'baseline']
            },
            "dropout": {
                #"distribution": 'uniform',
                #"min": 0.1,
                #"max": 0.5
                "values": [0.1]
            },
            "feat_drop": {
                "values": [0.]
            },
            "attn_drop": {
                "values": [0.]
            },
            "bn": {
                "distribution": 'categorical',
                "values": [False]
            },
            "bn_gat": {
                "distribution": 'categorical',
                "values": [False]
            },
            "wd": {
                #"distribution": 'uniform',
                #"max": 1,
                #"min": 0.001,
                "values": [0.1]
            },
            "heads": {
                "values": [1, 3]
            },
            "att_ew": {
                "distribution": 'categorical',
                "values": [ False]
            },               
            "gcn_drop": {
                "values": [0.]
            },
            "gcn_bn": {
                "distribution": 'categorical',
                "values": [True]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="dbu_graph")
    #init model
    #LitGNN_sys = LitGNN(model=model, lr=lr, model_type= model_type)

    wandb.agent(sweep_id, sweep_train)
    #print("############### TRAIN ####################")
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    # using only half the training data and checking validation every quarter of a training epoch
    
    #trainer = pl.Trainer(gpus=1, max_epochs=total_epoch, progress_bar_refresh_rate=20, precision=16, limit_train_batches=0.5, val_check_interval=0.25,profiler=True)  #val_check_interval=0.25
    #trainer.fit(LitGCN_sys, train_dataloader, val_dataloader)   

    #print("############### TEST ####################")
    #trainer.test(test_dataloaders=test_dataloader)


    #tensorboard
    #tensorboard --logdir lightning_logs/