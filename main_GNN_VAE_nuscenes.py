import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from nuscenes_Dataset import nuscenes_Dataset
from models.VAE_GNN import VAE_GNN
from models.VAE_GATED import VAE_GATED
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser, Namespace
import math
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction
input_dim_model = history_frames*2 + 7 #Input features to the model: x,y-global (zero-centralized), heading, history relative_positions (4fx2), vel, accel, heading_rate, type 
output_dim = future_frames*2

default_config = {
            "ew_dims":2,
            "input_dim": 6,
            "history_frames":history_frames,
            "future_frames":future_frames,
            "learning_rate":1e-5,
            "batch_size": 1,
            "hidden_dims": 512,
            "z_dims": 128,
            "dropout": 0.1,
            "alfa": 0,
            "beta": 1,
            "delta": 1,
            "feat_drop": 0.,
            "attn_drop":0.25,
            "bn":False,
            "wd": 0.01,
            "heads": 2,
            "att_ew": True,               
            "gcn_drop": 0.,
            "gcn_bn": True,
            'embedding':True
        }

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
    def __init__(self, model,  train_dataset, val_dataset, test_dataset, dataset,  history_frames: int=3, future_frames: int=3, input_dim: int=2, lr: float = 1e-3, batch_size: int = 64, wd: float = 1e-1, alfa: float = 2, beta: float = 0., delta: float = 1., rel_types: bool = False, scale_factor=1):
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
        self.min_val_loss = 100
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    
    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=512, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
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

    
    def vae_loss(self, pred, gt, mask, mu, log_var, beta=1, reconstruction_loss='huber'):
        #Train with Huber . Validate with MSE
        if reconstruction_loss == 'huber':
            overall_sum_time, overall_num = self.huber_loss(pred, gt, mask, self.delta)  #T
        else:
            overall_sum_time, overall_num,_ = self.compute_RMSE(pred, gt, mask)  #T

        recons_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2)) #T -> 1
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        std = torch.exp(log_var / 2)
        kld_loss = kl_divergence(
        Normal(mu, std), Normal(torch.zeros_like(mu), torch.ones_like(std))
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

    def compute_change_pos(self, feats,gt, scale_factor):
        gt_vel = gt.clone()  #.detach().clone()
        feats_vel = feats[:,:,:2].clone()
        new_mask_feats = (feats_vel[:, 1:]!=0) * (feats_vel[:, :-1]!=0) 
        new_mask_gt = (gt_vel[:, 1:]!=0) * (gt_vel[:, :-1]!=0) 
            
        rescale_xy=torch.ones((1,1,2), device=self.device)*scale_factor

        gt_vel[:, 1:] = (gt_vel[:, 1:] - gt_vel[:, :-1]) * new_mask_gt
        gt_vel[:, :1] = (gt_vel[:, :1] - feats_vel[:, -1:]*rescale_xy) * new_mask_gt[:,0:1]
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
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = train_batch
        
        if self.dataset  == 'apollo':
            #Use relative positions
            feats_rel, labels = self.compute_change_pos(feats,labels_pos,1)
            #Input pos + heading + vel
            feats = torch.cat([feats_rel, feats[:,:,2:]], dim=-1)[:,1:,:] # torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels_pos, self.scale_factor)

        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        labels = labels.contiguous().view(labels.shape[0],-1)

        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)

        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        labels=labels.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=self.beta)

        self.log_dict({f"Sweep/train_{k}": v for k,v in logs.items()}, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = val_batch
        
        if self.dataset == 'apollo':
            #Use relative positions
            feats_rel,labels = self.compute_change_pos(feats,labels_pos,1)
            #Input pos + heading + vel
            feats = torch.cat([feats_rel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels_pos, self.scale_factor)

        
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        labels = labels.contiguous().view(labels.shape[0],-1)

        e_w = batched_graph.edata['w']
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e,labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        labels=labels.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=self.beta, reconstruction_loss='mse')

        self.log_dict({"Sweep/val_loss": logs['loss'], "Sweep/val_mse_loss": logs['Reconstruction_Loss'], "Sweep/Val_KL": logs['KL']})
        return total_loss

    def validation_epoch_end(self, val_loss_over_batches):
        #log best val loss
        if torch.mean(torch.tensor(val_loss_over_batches,device=self.device)) < self.min_val_loss:            
            self.min_val_loss =  torch.mean(torch.tensor(val_loss_over_batches,device=self.device))
            self.logger.experiment.summary["best_val_loss"] = self.min_val_loss
    
         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = test_batch
        rescale_xy=torch.ones((1,1,2), device=self.device)*10
        last_loc = feats[:,-1:,:2].detach().clone() 
        last_loc = last_loc*rescale_xy       
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        
        ade = []
        fde = []         
        #En test batch=1 secuencia con n agentes
        #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
        for i in range(10): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
            #Model predicts relative_positions
            preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e)
            preds=preds.view(preds.shape[0],self.future_frames,-1)
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc
            #Compute error for this sample
            _ , overall_num, x2y2_error = self.compute_RMSE(preds, labels_pos, output_masks)
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
    
        self.log_dict({'test/ade': min(ade), "test/fde": fde[ade.index(min(ade))]})
        
   
def main(args: Namespace):
    # keep track of parameters in logs
    print(args)
    seed=seed_everything(np.random.randint(1000000))

    train_dataset = nuscenes_Dataset(train_val_test='train',  rel_types=args.ew_dims>1) #3447
    val_dataset = nuscenes_Dataset(train_val_test='val',  rel_types=args.ew_dims>1)  #919
    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1)  #230

    if args.model_type == 'vae_gated':
        model = VAE_GATED(input_dim_model, args.hidden_dims, z_dim=args.z_dims, output_dim=output_dim, fc=False, dropout=args.dropout,  ew_dims=args.ew_dims)
    else:
        model = VAE_GNN(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, feat_drop=args.feat_drop, attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, ew_dims=args.ew_dims)

    LitGNN_sys = LitGNN(model=model, input_dim=input_dim, lr=args.learning_rate,  wd=args.wd, history_frames=args.history_frames, future_frames= args.future_frames, alfa= args.alfa, beta = args.beta, delta=args.delta,
    dataset=args.dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)
    
    
    early_stop_callback = EarlyStopping('Sweep/val_loss', patience=3)

    if not args.nowandb:
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath=os.path.join('/media/14TBDISK/sandra/logs/',os.environ.get('WANDB_SWEEP_ID'),run.name))
        run=wandb.init(project="dbu_graph", entity="sandracl72", job_type="training")  
        wandb_logger = pl_loggers.WandbLogger() 
        wandb_logger.experiment.log({'seed': seed}) 
        #wandb_logger.watch(LitGNN_sys.model)  #log='all' for params & grads
        trainer = pl.Trainer( weights_summary='full', gpus=1, deterministic=True, precision=16, logger=wandb_logger, callbacks=[early_stop_callback,checkpoint_callback], profiler=True)  # resume_from_checkpoint=config.path, precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    else:
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath=os.path.join('/media/14TBDISK/sandra/logs/',run.name))
        trainer = pl.Trainer( weights_summary='full', gpus=1, deterministic=True, precision=16, callbacks=[early_stop_callback,checkpoint_callback], profiler=True) 

    
    print('Best lr: ', LitGNN_sys.lr)
    print('GPU Nº: ', device)
    print("############### TRAIN ####################")
    trainer.fit(LitGNN_sys)
    print('Model checkpoint path:',trainer.checkpoint_callback.best_model_path)
    
    print("############### TEST ####################")
    trainer.test(ckpt_path='best')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam: learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Adam: weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="Size of the batches")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=512)
    parser.add_argument("--model_type", type=str, default='vae_gat', help="Choose aggregation function between GAT or GATED",
                        choices=['vae_gat', 'vae_gated'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--att_drop", type=float, default=0.25)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument("--beta", type=float, default=1.0, help='Weighting factor of the KL divergence loss term')
    parser.add_argument("--delta", type=float, default=1.0, help='Delta factor in Huber Loss (Reconstruction Loss)')
    parser.add_argument('--att_ew', action='store_true', help='use this flag to add edge features in attention function (GAT)')    
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')

    
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    hparams = parser.parse_args()

    main(hparams)

    main()    
