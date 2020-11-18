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


batch_size=64
total_epoch = 30
lr =1e-3
hidden_dims=128
model_type= 'gated'



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


class LitGCN(pl.LightningModule):
    def __init__(self, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64):
        super().__init__()
        self.model= model
        self.lr = lr
        self.batch_size = batch_size
        self.epoch=0
        
        self.overall_loss_train=[]
        
        self.overall_num_list=[]
        self.overall_x2y2_list=[]
        
        self.test_overall_num_list=[]
        self.test_overall_x2y2_list=[]
        
        wandb.watch(self.model, log="all")
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt
    
    def compute_RMSE_batch(self,pred, gt, mask): 
        pred=pred.view(pred.shape[0],mask.shape[1],-1)
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) # x^2+y^2 BV,T
        overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los V agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(T)] - num de agentes (Y CON DATOS) en cada frame
        return overall_sum_time, overall_num, x2y2_error

    
    def training_step(self, train_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e = train_batch
        feats = batched_graph.ndata['x'].float()
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.view(feats.shape[0],-1)
        e_w = batched_graph.edata['w'].float()
        labels= batched_graph.ndata['gt'].float()
        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        overall_sum_time, overall_num, _ = self.compute_RMSE_batch(pred, labels, output_masks)  #(B,6)
        total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))
        self.overall_loss_train.extend([total_loss.data.item()])
        #self.log('train_loss',total_loss, prog_bar=True, logger=True, on_epoch=True)
        return total_loss
    
    def training_epoch_end(self, train_results):
        self.epoch += 1
        print('|{}| Train_loss: {}'.format(datetime.now(), np.sum(self.overall_loss_train)/len(self.overall_loss_train)))
        wandb.log({"Train/loss": np.sum(self.overall_loss_train)/len(self.overall_loss_train)}, step=self.epoch)#, step=epoch)      
        self.overall_loss_train=[]

    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e = val_batch
        feats = batched_graph.ndata['x'].float()
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.view(feats.shape[0],-1)
        e_w = batched_graph.edata['w'].float()
        labels= batched_graph.ndata['gt'].float()
        pred = model(batched_graph, feats,e_w,snorm_n,snorm_e)
        _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks)
        self.overall_num_list.extend(overall_num.detach().cpu().numpy())
        self.overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  
        
    def validation_epoch_end(self, val_results):
        overall_sum_time=np.sum(self.overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
        overall_num_time =np.sum(self.overall_num_list, axis=0)
        overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame
        
        self.overall_num_list=[]
        self.overall_x2y2_list=[]
        #self.log('val_loss',val_loss_sum,prog_bar=True, logger=True, on_epoch=True)
        wandb.log({"Val/loss": np.sum(overall_loss_time)}, step= self.epoch)
    
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e = test_batch
        feats = batched_graph.ndata['x'].float()
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.view(feats.shape[0],-1)
        e_w = batched_graph.edata['w'].float()
        #for GatedGCN
        #e_w= e_w.view(e_w.shape[0],1)
        labels= batched_graph.ndata['gt'].float()
        pred = model(batched_graph, feats,e_w,snorm_n,snorm_e)
        _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks)
        self.test_overall_num_list.extend(overall_num.detach().cpu().numpy())
        self.test_overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())
                  
    def test_step_end(self,test_results):
        #self.log('test_loss',np.sum(overall_x2y2_list,axis=0)/np.sum(overall_num_list,axis=0),prog_bar=True, logger=True, on_epoch=True)
        wandb.log({'test/loss': np.sum(self.test_overall_x2y2_list,axis=0)/np.sum(self.test_overall_num_list,axis=0)})    

            


if __name__ == '__main__':

    train_dataset = ApolloScape_DGLDataset(train_val='train') #3447
    val_dataset = ApolloScape_DGLDataset(train_val='val')  #919
    test_dataset = ApolloScape_DGLDataset(train_val='test')  #230

    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    val_dataloader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True,  num_workers=12, collate_fn=collate_batch)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=12, collate_fn=collate_batch)  
    
    if model_type == 'gat':
        model = My_GAT(input_dim=18, hidden_dim=hidden_dims, output_dim=12)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=18, hid_feats=hidden_dims, out_feats=12)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=18, hidden_dim=hidden_dims, output_dim=12)

    
    wandb.init(project="dbu_graph", config={"epochs": total_epoch, "batch_size": batch_size, "learning_rate": lr,"model_architecture": model, "hidden_dims": hidden_dims})
    #init model
    LitGCN = LitGCN(model=model, lr=lr)
    
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='max'
    )

    print("############### TRAIN ####################")
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    # using only half the training data and checking validation every quarter of a training epoch
    
    trainer = pl.Trainer(gpus=1, max_epochs=total_epoch, progress_bar_refresh_rate=20, precision=16, limit_train_batches=0.5, val_check_interval=0.25,profiler=True)  #val_check_interval=0.25
    trainer.fit(LitGCN, train_dataloader, val_dataloader)   

    print("############### TEST ####################")
    trainer.test(test_dataloaders=test_dataloader)


    #tensorboard
    #tensorboard --logdir lightning_logs/