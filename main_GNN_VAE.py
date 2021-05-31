import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from roundD_Dataset import roundD_DGLDataset
from models.VAE_GNN import VAE_GNN
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import math
from torch.distributions.kl import kl_divergence


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str , default='gated_mdn' ,help='model type')
parser.add_argument('--name', type=str , default='DEBUG' ,help='sweep name')
parser.add_argument('--count', type=int , default=16 ,help='sweep number of runs')
parser.add_argument('--history_frames', type=int , default=6,help='Temporal size of the history sequence.')
parser.add_argument('--future_frames', type=int , default=6 ,help='Temporal size of the predicted sequence.')
parser.add_argument('--gpus', type=int , nargs='+',default=0)
parser.add_argument('--dataset', type=str , default='apollo')
parser.add_argument('--apollo_vel', type=bool , default=True)
parser.add_argument('--norm', type=bool , default=True)
parser.add_argument('--res_connection', type=bool , default=True)
parser.add_argument('--res_weight', type=bool , default=True)  #False  = ' '
args = parser.parse_args()
dataset = args.dataset



def collate_batch(samples):
    graphs, masks, feats, gt = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt



class LitGNN(pl.LightningModule):
    def __init__(self, model, history_frames: int=3, future_frames: int=3, input_dim: int=2, lr: float = 1e-3, batch_size: int = 64, wd: float = 1e-1, alfa: float = 2, beta: float = 0., delta: float = 1.):
        super().__init__()
        self.model= model
        self.lr = lr
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.alfa = alfa
        self.beta = beta
        self.delta = delta
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt
    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    
    def val_dataloader(self):
        return  DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
    
    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
    def compute_RMSE(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum((pred-gt)**2,dim=-1) # x^2+y^2 BV,T  PROBABILISTIC -> gt[:,:,:2]
        overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return overall_sum_time, overall_num, x2y2_error

    def huber_loss(self, pred, gt, mask, delta):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        #error = torch.sum(torch.where(torch.abs(gt-pred) < delta , 0.5*((gt-pred)**2)*(1/delta), torch.abs(gt - pred) - 0.5*delta), dim=-1)   # BV,T
        error = torch.sum(torch.where(torch.abs(gt-pred) < delta , (0.5*(gt-pred)**2), torch.abs(gt - pred)*delta - 0.5*(delta**2)), dim=-1)
        overall_sum_time = error.sum(dim=-2) #T - suma de los errores de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int) 
        return overall_sum_time, overall_num

    
    def vae_loss(self, pred, gt, mask, mu, log_var, beta=1):
        #overall_sum_time, overall_num, _ = self.compute_RMSE(pred, gt, mask) #T
        overall_sum_time, overall_num = self.huber_loss(pred, gt, mask, self.delta)  #T
        recons_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2)) #T -> 1
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        std = torch.exp(log_var / 2)
        kld_loss = kl_divergence(
        torch.distributions.Normal(mu, std), torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        ).sum(-1)
        loss = recons_loss + beta * kld_loss.mean()
        return loss, {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KL':kld_loss}
  

    def check_overlap(self, preds):
        intersect=[]
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].detach().cpu().numpy()-preds[j,:,1].detach().cpu().numpy()))).size > 0  for i in range(len(preds)-1) for j in range(i+1,len(preds))]
        #x_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,0].detach().cpu().numpy()-preds[j,:,0].detach().cpu().numpy()[::-1]))).size > 0 for i in range(len(preds)-1)  for j in range(i+1,len(preds))]
        #intersect = [True if y and x else False for y,x in zip(y_intersect,x_intersect)]
        '''
        for i in range(len(preds)-1):
            for j in range(i+1,len(preds)):
                y_intersect=(torch.sign(preds[i,:,1]-preds[j,:,1])-torch.sign(preds[i,:,1]-preds[j,:,1])[0]).bool().any()  #True if non all-zero
                x_intersect=(torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))-torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))[0]).bool().any()
                intersect.append(True if y_intersect and x_intersect else False)
        '''
        y_sub = torch.cat([torch.sign(preds[i:-1,:,1]-preds[i+1:,:,1]) for i in range(len(preds)-1)])  #N(all combinations),6
        y_intersect=( y_sub - y_sub[:,0].view(len(y_sub),-1)).bool().any(dim=1) #True if non all-zero (change sign)
        x_sub = torch.cat([torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0])) for i in range(len(preds)-1)])
        x_intersect = (x_sub -x_sub[:,0].view(len(x_sub),-1)).bool().any(dim=1)
        #x_intersect=torch.cat([(torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))-torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))[0]).bool().any(dim=1) for i in range(len(preds)-1)])
        intersect = torch.logical_and(y_intersect,x_intersect) #[torch.count_nonzero(torch.logical_and(y,x))/len(x) for y,x in zip(y_intersect,x_intersect)] #to intersect, both True
        return torch.count_nonzero(intersect)/len(intersect) #percentage of intersections between all combinations
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].cpu()-preds[j,:,1].cpu()))).size > 0 for j in range(i+1,len(preds)) for i in range(len(preds)-1)]

    def compute_change_pos(self, feats,gt):
        gt_vel = gt.clone()  #.detach().clone()
        feats_vel = feats[:,:,:2].clone()
        new_mask_feats = (feats_vel[:, 1:]!=0) * (feats_vel[:, :-1]!=0) 
        new_mask_gt = (gt_vel[:, 1:]!=0) * (gt_vel[:, :-1]!=0) 

        gt_vel[:, 1:] = (gt_vel[:, 1:] - gt_vel[:, :-1]) * new_mask_gt
        gt_vel[:, :1] = (gt_vel[:, :1] - feats_vel[:, -1:]) * new_mask_gt[:,0:1]
        feats_vel[:, 1:] = (feats_vel[:, 1:] - feats_vel[:, :-1]) * new_mask_feats
        feats_vel[:, 0] = 0
        
        return feats_vel, gt_vel

    def compute_long_lat_error(self,pred,gt,mask):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        lateral_error = pred[:,:,0]-gt[:,:,0]
        long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return lateral_error, long_error, overall_num

    
    
    def training_step(self, train_batch, batch_idx):
        '''returns a loss from a single batch'''
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels = train_batch
        if dataset.lower() == 'apollo' and args.apollo_vel:
            #USE CHANGE IN POS AS INPUT
            feats_vel, labels = self.compute_change_pos(feats,labels)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] # torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels)

        e_w = batched_graph.edata['w'].float()
        #if not self.rel_types:
        #    e_w= e_w.unsqueeze(1)

        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=1)

        self.log_dict({f"Sweep/train_{k}": v for k,v in logs.items()}, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = val_batch
        last_loc = feats[:,-1:,:2]
        if dataset.lower() == 'apollo' and args.apollo_vel:
            #USE CHANGE IN POS AS INPUT
            feats_vel,labels = self.compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels)

        e_w = batched_graph.edata['w'].float()
        #if not self.rel_types:
        #    e_w= e_w.unsqueeze(1)
        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e,labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=1)

        self.log_dict({"Sweep/val_loss": logs['loss'], "Sweep/val_recons_loss": logs['Reconstruction_Loss'], "Sweep/Val_KL": logs['KL']})

         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = test_batch
        last_loc = feats[:,-1:,:2].detach().clone()        
        e_w = batched_graph.edata['w'].float()
        #if not self.rel_types:
        #    e_w= e_w.unsqueeze(1)
        ade = []
        fde = []         
        #En test batch=1 secuencia con n agentes
        #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
        for i in range(10): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
            #Model predicts relative_positions
            pred = self.model.inference(batched_graph, feats[:,:,:self.input_dim],e_w,snorm_n,snorm_e)
            preds=preds.view(preds.shape[0],self.future_frames,-1)
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc
            #Compute error for this sample
            _ , overall_num, x2y2_error = self.compute_RMSE_batch(preds[:,:self.future_frames,:], labels_pos[:,:self.future_frames,:], output_masks)
            ade_ts = torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)   
            ade_s = torch.sum(ade_ts)/ self.future_frames  #T ->1
            fde_s = torch.sum((x2y2_error**0.5), dim=0)[-1] / torch.sum(overall_num, dim=0)[-1]
            if torch.isnan(fde_s):  #visible pero no tiene datos para los siguientes 6 frames
                print('stop')
                fde_s[np.isnan(fde_s)]=0
                ade_ts[np.isnan(ade_ts)]=0
                for j in range(self.future_frames-2,-1,-1):
                    if ade_ts[j] != 0:
                            fde_s =  torch.sum((x2y2_error**0.5), dim=0)[j] / torch.sum(overall_num, dim=0)[j]  #compute FDE with the last frame with data
                            ade_s = torch.sum(ade_ts)/ (j+1) #compute ADE dividing by number of frames with data
                            break
            ade.append(ade_s) #S samples
            fde.append(fde_s)
    
        self.log_dict({'test/ade': min(ade), "test/fde": min(fde)}) #, sync_dist=True
    
        
   
def sweep_train():

    #wandb.init()
    
    run=wandb.init(project="dbu_graph", config=default_config)
    config = wandb.config

    print('config: ', dict(config))
    wandb_logger = pl_loggers.WandbLogger()  #name=
    
    input_dim = config.input_dim*5 if dataset=='apollo' else config.input_dim*config.history_frames
    output_dim = 2*config.future_frames if config.probabilistic == False else 5*config.future_frames

    model = VAE_GNN(input_dim, config.hidden_dims//config.heads, config.z_dims, output_dim, fc=False, dropout=config.dropout, feat_drop=config.feat_drop, attn_drop=config.attn_drop, heads=config.heads, att_ew=config.att_ew, ew_dims=config.ew_dims)

    LitGNN_sys = LitGNN(model=model, input_dim=input_dim, lr=config.learning_rate,  wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames, alfa= config.alfa, beta = config.beta, delta=config.delta)
    wandb_logger.watch(LitGNN_sys.model)  #log='all' for params & grads
    
    checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath='./logs/'+run.name, save_top_k=0)
    early_stop_callback = EarlyStopping('Sweep/val_loss', patience=6)
    trainer = pl.Trainer(weights_summary='full', gpus=1,  logger=wandb_logger,  profiler=True, callbacks=[early_stop_callback,checkpoint_callback] ) # precision=16,precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    #resume_from_checkpoint=path, 
    
    print('Best lr: ', LitGNN_sys.lr)
    print("############### TRAIN ####################")
    trainer.fit(LitGNN_sys)
    #wandb.save(trainer.checkpoint_callback.best_model_path)
    
    print("############### TEST ####################")
    if dataset.lower() != 'apollo':
        trainer.test()




if __name__ == '__main__':

    history_frames = args.history_frames
    future_frames = args.future_frames
    print(args.norm)
    if dataset.lower() == 'apollo':
        train_dataset = ApolloScape_DGLDataset(train_val='train',  test=False, ew_dims=2) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', test=False, ew_dims=2)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='val', test=False, ew_dims=2)  #230
        print(len(train_dataset), len(val_dataset))
        input_dim = [5] 
    elif dataset.lower() == 'ind':
        train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,2,3,4)) #12281
        val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,2,3,4))  #3509
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,2,3,4))  #1754
        print(len(train_dataset), len(val_dataset), len(test_dataset))
        input_dim = [6]
    elif dataset.lower() == 'round':
        train_dataset = roundD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,3,5,6,7,8)) #12281
        val_dataset = roundD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,3,5,6,7,8))  #3509
        test_dataset = roundD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type=args.model, classes=(1,3,5,6,7,8))  #1754
        print(len(train_dataset), len(val_dataset), len(test_dataset))
        input_dim = [4]

    
    if args.model == 'gat':
        bn = [False]
        heads = [3]
        att_ew = [True]
        bs = [ 512]
        lr=[ 1e-5,1e-4,5e-4]
        alfa = [0]
        beta = [0]
        delta = [1]
        attn_drop = [0.2]
        hidden_dims = [256,1024]
    else:
        heads = [1]
        attn_drop = [0.]
        att_ew = [False]
        beta = [0]
        delta = [0]
        bs = [512]
        lr = [1e-4, 3e-4, 3e-5] if args.model == 'rgcn' else [1e-6]
        hidden_dims = [512, 1024] if args.model == 'rgcn' else [512]
        alfa = [0,1,2,4] if args.model == 'rgcn' else [0]
        bn = [True]


    sweep_config = {
    "name": args.name,
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
            "probabilistic":{
                "values": [True]
            },
            "input_dim":{
                "values": input_dim #velxy+heading+obj+mask
            },
            "embedding":{
                "values": [True]
            },
            "history_frames":{
                "values": [history_frames]
            },
            "history_frames":{
                "values": [history_frames]
            },
            "future_frames":{
                "values": [future_frames]
            },
            "learning_rate":{
                #"distribution": 'log_uniform',
                #"max": -2,
                #"min": -6
                "values": lr
            },
            "batch_size": {
                #"distribution": 'int_uniform',
                #"max": 512,
                #"min": 64,
                "values": bs
            },
            "hidden_dims": {
                #"distribution": 'int_uniform',
                #"max": 1024,
                #"min": 512
                "values": hidden_dims
            },
            "model_type": {
                "values": [args.model]
            },
            "dropout": {
                #"distribution": 'uniform',
                #"min": 0.01,
                #"max": 0.5
                "values": [0.]
            },
            "alfa": {
                #"distribution": 'uniform',
                #"min": 1,
                #"max": 4
                "values": alfa
            },
            "beta": {
                #"distribution": 'uniform',
                #"min": 0,
                #"max": 1
                "values": beta
            },
            "delta": {
                #"distribution": 'uniform',
                #"min": 0.01,
                #"max": 10
                "values": delta
            },
            "feat_drop": {
                "values": [0.]
            },
            "attn_drop": {
                "values": attn_drop
            },
            "bn": {
                #"distribution": 'categorical',
                "values": bn
            },
            "wd": {
                #"distribution": 'log_uniform',
                #"max": -1,
                #"min": -3
                "values": [0.01]
            },
            "heads": {
                "values": heads
            },
            "att_ew": {
                "distribution": 'categorical',
                "values": att_ew
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

    
    default_config = {
            "probabilistic": True,
            "input_dim": input_dim[0],
            "history_frames":history_frames,
            "future_frames":future_frames,
            "learning_rate":1e-6,
            "batch_size": 1,
            "hidden_dims": 512,
            "z_dims": 128,
            "model_type": args.model,
            "dropout": 0.1,
            "alfa": 0,
            "beta": 0,
            "delta": 0.1,
            "feat_drop": 0.,
            "attn_drop":0.,
            "bn":False,
            "wd": 0.01,
            "heads": 2,
            "att_ew": True,               
            "gcn_drop": 0.,
            "gcn_bn": True,
            'embedding':True,
            'ew_dims': 2
        }
    #sweep_id = wandb.sweep(sweep_config, project="dbu_graph", entity='sandracl72')
    #wandb.agent(sweep_id, sweep_train, count=args.count)
    sweep_train()
