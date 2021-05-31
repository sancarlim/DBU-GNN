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
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

#from LIT_system import LitGNN

batch_size=64
total_epoch = 50
learning_rate =1e-3
hidden_dims=64
learning_rate=1e-3


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
    masks = np.vstack(masks)
    masks = torch.tensor(masks)#+torch.zeros(2)
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
    def __init__(self, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gcn', wd: float = 1e-1):
        super().__init__()
        self.model= model
        self.lr = lr
        self.model_type = model_type
        self.batch_size = batch_size
        self.wd = wd
        wandb.watch(self.model, log="all")
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.wd)
        return opt
    
    def compute_RMSE_batch(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) # x^2+y^2 BV,T
        overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los V agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return overall_sum_time, overall_num, x2y2_error

    def compute_change_pos(self, data):
        '''
        gt_vel = gt.detach().clone()
        feats_vel = feats.detach().clone()
        new_mask_feats = (feats_vel[:, 1:,:2]!=0) * (feats_vel[:, :-1, :2]!=0) 
        new_mask_gt = (gt_vel[:, 1:,:2]!=0) * (gt_vel[:, :-1, :2]!=0) 

        gt_vel[:, 1:,:2] = (gt_vel[:, 1:,:2] - gt_vel[:, :-1, :2]).float() * new_mask_gt.float()
        gt_vel[:, :1, :2] = (gt_vel[:, 0:1,:2] - feats_vel[:, -1:, :2]).float()
        feats_vel[:, 1:,:2] = (feats_vel[:, 1:,:2] - feats_vel[:, :-1, :2]).float() * new_mask_feats.float()
        feats_vel[:, 0, :2] = 0
        '''
        new_data = data.detach().clone()
        new_mask = (new_data[:, 1:,:2]!=0) * (new_data[:, :-1, :2]!=0)
        new_data[:, 1:,:2] = (new_data[:, 1:,:2] - new_data[:, :-1, :2]).float() * new_mask.float() 
        new_data[:, 0, :2] = 0

        return new_data.float()

    def training_step(self, train_batch, batch_idx):
        '''needs to return a loss from a single batch'''

        batched_graph, output_masks,snorm_n, snorm_e = train_batch
        feats = batched_graph.ndata['x'].float()
        labels= batched_graph.ndata['gt'].float()   #AHORA LABELS C=4  Y MASK T=12
        #USE CHANGE IN POS AS INPUT
        data = torch.cat(feats,labels,dim=1)   #BV,12,4
        new_data = self.compute_change_pos(data)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.view(e_w.shape[0],1)
        
        for now_history_frames in range(1, feats.shape[1]):
            input_data = new_data[:,:now_history_frames,:]
            output_gt = labels[:,now_history_frames:now_history_frames+feats.shape[1],:2]
            output_masks =

        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        overall_sum_time, overall_num, _ = self.compute_RMSE_batch(pred, labels, output_masks)  #(B,6)
        total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))
        # Log metrics
        self.logger.agg_and_log_metrics({"Train/loss": total_loss.data.item()}, step=self.current_epoch)
        #self.log("Train/loss",  total_loss.data.item(), on_step=False, on_epoch=True)
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
        labels= batched_graph.ndata['gt'].float()
        #USE CHANGE IN POS AS INPUT
        feats_vel,_ = self.compute_change_pos(feats,labels)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.view(e_w.shape[0],1)

        pred = self.model(batched_graph, feats_vel,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        # Compute predicted trajs.
        last_loc = feats[:,-1:,:2]
        
        for i in range(1,feats.shape[1]):
            pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-1) #BV,6,2  
        pred += last_loc

        _ , overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks)
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(), axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0)#T
        self.logger.agg_and_log_metrics({ "val/rmse_loss": np.sum(overall_loss_time) }, step= self.current_epoch)
        
        mse_overall_loss_time =np.sum(np.sum(x2y2_error.detach().cpu().numpy(), axis=0)) / np.sum(np.sum(overall_num.detach().cpu().numpy(), axis=0)) 
        self.logger.agg_and_log_metrics({'val/Loss':mse_overall_loss_time}, step= self.current_epoch) #aggregate loss for epochs
        #self.log('val/Loss', np.sum(overall_loss_time) )
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
        labels= batched_graph.ndata['gt'].float()
        #USE CHANGE IN POS AS INPUT
        feats_vel,_ = self.compute_change_pos(feats,labels)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.view(e_w.shape[0],1)

        pred = self.model(batched_graph, feats_vel,e_w,snorm_n,snorm_e)
        pred=pred.view(pred.shape[0],labels.shape[1],-1)
        # Compute predicted trajs.
        last_loc = feats[:,-1:,:2]
        for i in range(1,feats.shape[1]):
            pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-1) #BV,6,2 
        pred += last_loc

        _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks)
        #self.test_overall_num_list.extend(overall_num.detach().cpu().numpy())#BV,T
        #self.test_overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  #BV,T
        overall_loss_time = np.sum((x2y2_error**0.5).detach().cpu().numpy(),axis=0) / np.sum(overall_num.detach().cpu().numpy(), axis=0) #T
        overall_loss_time[np.isnan(overall_loss_time)]=0
        
        self.log_dict({'test/loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[:2]), "test/loss_2": torch.tensor(overall_loss_time[2:4]), "test/loss_3": torch.tensor(overall_loss_time[4:]) })
 

    '''             
    def test_epoch_end(self,test_results):
        overall_sum_time=np.sum(self.test_overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
        overall_num_time =np.sum(self.test_overall_num_list, axis=0)
        overall_loss_time=(overall_sum_time / overall_num_time)
        self.log('test/loss', np.sum(overall_loss_time))
        #self.log('test/loss_per_sec',' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time)]))
        self.log('test/loss_per_sec', overall_loss_time)    

        #wandb.log({'test/loss': np.sum(overall_loss_time)})    
    '''

def sweep_train():
    wandb.init() 
    config = wandb.config
    print('config: ', dict(config))
    wandb_logger = pl_loggers.WandbLogger(save_dir='./logs/')  #name=
    train_dataloader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    val_dataloader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=12,collate_fn=collate_batch)


    print(config.model_type)
    if config.model_type == 'gat':
        hidden_dims = round(config.hidden_dims / config.heads) 
        model = My_GAT(input_dim=24, hidden_dim=hidden_dims, output_dim=12, heads=config.heads, dropout=config.dropout, bn=config.bn, bn_gat=config.bn_gat, feat_drop=config.feat_drop, attn_drop=config.attn_drop, att_ew=config.att_ew)
    elif config.model_type == 'gcn':
        model = model = GCN(in_feats=24, hid_feats=config.hidden_dims, out_feats=12, dropout=config.dropout, gcn_drop=config.gcn_drop, bn=config.bn, gcn_bn=config.gcn_bn)
    elif config.model_type == 'gated':
        model = GatedGCN(input_dim=24, hidden_dim=config.hidden_dims, output_dim=12)

    LitGNN_sys = LitGNN(model=model, lr=learning_rate, model_type= config.model_type, wd=config.wd)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    #checkpoint_callback = ModelCheckpoint(monitor='val/Loss', mode='min')
    trainer = pl.Trainer(gpus=1, max_epochs=40,logger=wandb_logger, precision=16, default_root_dir='./models_checkpoints/', profiler=True)  #precision=16, callbacks=[early_stop_callback],limit_train_batches=0.5, progress_bar_refresh_rate=20, 
    
    print("############### TRAIN ####################")
    trainer.fit(LitGNN_sys, train_dataloader, val_dataloader)   

    print("############### TEST ####################")
    trainer.test(test_dataloaders=test_dataloader)




if __name__ == '__main__':

    train_dataset = ApolloScape_DGLDataset(train_val='train') #3447
    val_dataset = ApolloScape_DGLDataset(train_val='val')  #919
    test_dataset = ApolloScape_DGLDataset(train_val='test')  #230
    
    #train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=12, collate_fn=collate_batch)  
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
    "name": "Sweep gat drop att_ew hids val/loss=mse ",
    "method": "grid",
    "metric": {
        'name': 'val/Loss',
        'goal': 'minimize'   
    },
    "early_terminate": {
        'type': 'hyperband',
        'min_iter': 3
    },
    "parameters": {
            "batch_size": {
                #"distribution": 'int_uniform',
                #"max": 512,
                #"min": 64,
                "values": [128]
            },
            "hidden_dims": {
                #"distribution": 'int_uniform',
                #"max": 512,
                #"min": 64,
                "values": [256,511]
            },
            "model_type": {
                "values": ['gat']
            },
            "dropout": {
                #"distribution": 'uniform',
                #"min": 0.1,
                #"max": 0.5
                "values": [0.1,0.25]
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
                "values": [1]
            },
            "att_ew": {
                "values": [ True, False]
            }               
            #"gcn_drop": {
            #    "values": [0, 0.2]
            #},
            #"gcn_bn": {
            #    "distribution": 'categorical',
            #    "values": [True, False]
            #}
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