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
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from roundD_Dataset import roundD_DGLDataset
from models.GCN import GCN 
from models.My_GAT import My_GAT
from models.SCOUT_MDN import SCOUT_MDN
from models.Gated_MDN import Gated_MDN
from models.Gated_GCN import GatedGCN
from models.gnn_rnn import Model_GNN_RNN
from models.rnn_baseline import RNN_baseline
from models.RGCN import RGCN
from tqdm import tqdm
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_normalized_adj
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str , default='gated_mdn' ,help='model type')
parser.add_argument('--name', type=str , default='DEBUG' ,help='sweep name')
parser.add_argument('--count', type=int , default=16 ,help='sweep number of runs')
parser.add_argument('--history_frames', type=int , default=8,help='Temporal size of the history sequence.')
parser.add_argument('--future_frames', type=int , default=12 ,help='Temporal size of the predicted sequence.')
parser.add_argument('--gpus', type=int , nargs='+',default=0)
parser.add_argument('--dataset', type=str , default='ind')
parser.add_argument('--apollo_vel', type=bool , default=True)
parser.add_argument('--norm', type=bool , default=True)
parser.add_argument('--res_connection', type=bool , default=True)
parser.add_argument('--res_weight', type=bool , default=True)  #False  = ' '
args = parser.parse_args()
dataset = args.dataset

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(121958)

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
    def __init__(self, history_frames: int=3, future_frames: int=3, input_dim: int=2, model: nn.Module = GCN, lr: float = 1e-3, batch_size: int = 64, model_type: str = 'gcn', wd: float = 1e-1, alfa: float = 2, beta: float = 0., delta: float = 1., prob: bool = False):
        super().__init__()
        self.model= model
        self.lr = lr
        self.input_dim = input_dim
        self.model_type = model_type
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.alfa = alfa
        self.beta = beta
        self.delta = delta
        self.probabilistic = prob
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.probabilistic:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
            #opt=torch.optim.SGD(self.parameters(),lr=0.01)
        return opt
    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    
    def val_dataloader(self):
        return  DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
    
    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
    def bivariate_loss(self,pred,gt):
        #mux, muy, sx, sy, corr
        normx = gt[:,:,0]- pred[:,:,0]
        normy = gt[:,:,1]- pred[:,:,1]

        sx = torch.exp(pred[:,:,2]) #sx
        sy = torch.exp(pred[:,:,3]) #sy
        corr = torch.tanh(pred[:,:,4]) #corr
        
        sxsy = sx * sy

        z = torch.clamp((normx/sx)**2,max=1e15) + torch.clamp((normy/sy)**2,max=1e15) - 2*((corr*normx*normy)/sxsy) #normx/sx)**2 inf (1e38 max para tensor)
        negRho = torch.clamp(1 - corr**2,min=1e-15)

        # Numerator
        result = torch.clamp(torch.exp(-z/(2*negRho)),max=1e15)
        # Normalization factor
        denom = torch.clamp(2 * np.pi * (sxsy * torch.sqrt(negRho)),min=1e-15)

        # Final PDF calculation
        result = result / denom

        # Numerical stability
        epsilon = 1e-20

        #Loss: neg log-likelihood
        result = -torch.log(torch.clamp(result, min=epsilon))
        result = torch.mean(result)
        
        return result

    def gaussian_probability(self,sigma, mu, target):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
        
        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.unsqueeze(1).expand_as(sigma)
        ret = ONEOVERSQRT2PI * torch.clamp(torch.exp(-0.5 * ((target - mu) / sigma)**2), min=1e-15) / sigma   #B,G,O
        return torch.clamp(torch.prod(ret, 2),min=1e-30)  #B,G


    def mdn_loss(self,pred, target, mask):
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
            mask (B,G,12)  -  indicates wether there is data in that frame for each agent
        """
        pi, sigma, mu = pred  #B G 12, pi(B,G)
        mu = mu*mask
        target = target.contiguous().view(target.shape[0],-1)  
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))  #-log(prob(Bi)) #B  - suma las probabilidades de que label pertenezca a cada Gaussiana, cuanta mayor prob menor nll
        return torch.mean(nll)


    def sample(self,pred):
        """Draw samples from a MoG.
        """
        pi, sigma, mu = pred
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)  #B
        #Create a tensor of the same type as sigma and size B,O (Gaussian noise sample ~ N(0,I))
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_()) #B,12
        for i, idx in enumerate(pis):
            #sample from one of the modes for each agent in the batch z=mu(idx)+sigma(idx)*N(0,I) , where idx ~ pi
            sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
        return sample

    
    def compute_RMSE_batch(self,pred, gt, mask): 
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
        '''needs to return a loss from a single batch'''
        
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels = train_batch
        
        if dataset.lower() == 'apollo' and args.apollo_vel:
            #USE CHANGE IN POS AS INPUT
            feats_vel, labels = self.compute_change_pos(feats,labels)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] # torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        
        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats[:,:,:],e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        
        #Probabilistic vs. Deterministic output
        if self.probabilistic:
            #total_loss = self.bivariate_loss(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
            mask = output_masks[:,self.history_frames:self.total_frames,:].expand(output_masks.shape[0],self.future_frames, 2)  #expand mask (B,Tpred,1) -> (B,T_pred,2)
            total_loss = self.mdn_loss(pred, labels,mask.contiguous().view(mask.shape[0],-1).unsqueeze(1).expand_as(pred[1]))  
        else:
            #Socially consistent
            perc_overlap = self.check_overlap(pred*output_masks[:,self.history_frames:self.total_frames,:]) if self.alfa !=0 else 0
            pred=pred.view(labels.shape[0],self.future_frames,-1)
            overall_sum_time, overall_num = self.huber_loss(pred[:,:self.future_frames,:], labels[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:], self.delta)  #(B,6)
            #overall_sum_time , overall_num = self.compute_RMSE_batch(pred[:,:self.future_frames,:], labels[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:])
            total_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))*(1+self.alfa*perc_overlap) + self.beta*(overall_sum_time[-1]/overall_num.sum(dim=-2)[-1])

        # Log metrics
        self.log("Sweep/train_loss",  total_loss, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, val_batch, batch_idx):
        
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = val_batch
        #feats = batched_graph.ndata['x']
        #labels= batched_graph.ndata['gt'][:,:,:2].float()
        last_loc = feats[:,-1:,:2]
        if dataset.lower() == 'apollo' and args.apollo_vel:
            #USE CHANGE IN POS AS INPUT
            feats_vel,labels = self.compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats[:,:,:], e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e)
        
        if self.probabilistic:
            mask = output_masks[:,self.history_frames:self.total_frames,:].expand(output_masks.shape[0],self.future_frames, 2)  #expand mask (B,Tpred,1) -> (B,T_pred,2)
            total_loss = self.mdn_loss(pred, labels,mask.contiguous().view(mask.shape[0],-1).unsqueeze(1).expand_as(pred[1]))  
            ade = []
            fde = []         
            #En test batch=1 secuencia con n agentes
            #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
            for i in range(15): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
                preds=self.sample(pred)  #N,12
                preds=preds.view(preds.shape[0],self.future_frames,-1)
                if dataset.lower() == 'apollo':
                    for j in range(1,labels.shape[1]):
                        preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
                    preds += last_loc

                _ , overall_num, x2y2_error = self.compute_RMSE_batch(preds[:,:self.future_frames,:], labels_pos[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:])
                ade_s = torch.sum(torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)) / self.future_frames  #T ->1
                fde_s = torch.sum((x2y2_error**0.5), dim=0)[-1] / torch.sum(overall_num, dim=0)[-1]
                if torch.isnan(fde_s) or torch.isnan(ade_s):  #visible pero no tiene datos para los siguientes 6 frames
                    print('stop')
                ade.append(ade_s) #S samples
                fde.append(fde_s)
            print('\n',min(ade).item(),'\n', min(fde).item())
            self.log_dict({'test/ade': min(ade), "test/fde": min(fde)}) #, sync_dist=True
        
        else:
            pred=pred.view(labels.shape[0],self.future_frames,-1)
            
            if dataset.lower() == 'apollo':
                for i in range(1,labels.shape[1]):
                    pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-2) #BV,6,2 
                pred += last_loc
           
            _ , overall_num, x2y2_error = self.compute_RMSE_batch(pred[:,:self.future_frames,:], labels_pos[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:])
            total_loss = torch.sum(torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)) / self.future_frames #T->1

        self.log( "Sweep/val_loss", total_loss )   #torch.sum(overall_loss_time).clone().detach()

         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = test_batch
        last_loc = feats[:,-1:,:2].detach().clone()
        '''
        if self.dataset == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel,_ = self.compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        '''
        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn':
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats,e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats[:,:,:self.input_dim],e_w,snorm_n,snorm_e)
       
       
        if self.probabilistic:
            ade = []
            fde = []         
            #En test batch=1 secuencia con n agentes
            #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
            for i in range(10): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
                preds=self.sample(pred)  #N,12
                preds=preds.view(preds.shape[0],self.future_frames,-1)
                #if self.dataset == 'apollo':
                for j in range(1,labels_pos.shape[1]):
                    preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
                preds += last_loc

                _ , overall_num, x2y2_error = self.compute_RMSE_batch(preds[:,:self.future_frames,:], labels_pos[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:])
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
        
        else:
            pred=pred.view(pred.shape[0],labels.shape[1],-1)
            # Compute predicted trajs.
            if dataset.lower() == 'apollo':
                for i in range(1,labels.shape[1]):
                    pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-2) #BV,6,2 
                pred += last_loc
                
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
            
            if self.future_frames == 3:
                self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[:1]), "test/loss_2": torch.tensor(overall_loss_time[1:2]), "test/loss_3": torch.tensor(overall_loss_time[2:]) })
            elif self.future_frames == 6:
                self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[1:2]), "test/loss_2": torch.tensor(overall_loss_time[3:4]), "test/loss_3": torch.tensor(overall_loss_time[-1:]) })
            
            elif self.future_frames == 5:
                self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_1": torch.tensor(overall_loss_time[:1]), "test/loss_2": torch.tensor(overall_loss_time[1:2]), "test/loss_3": torch.tensor(overall_loss_time[2:]), "test/loss_4": torch.tensor(overall_loss_time[3:4]), "test/loss_5": torch.tensor(overall_loss_time[-1:]) })
            elif self.future_frames == 8:
                self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_0.8": torch.tensor(overall_loss_time[1:2]), "test/loss_2": torch.tensor(overall_loss_time[4:5]), "test/loss_2.8": torch.tensor(overall_loss_time[6:7]), "test/loss_3.2": torch.tensor(overall_loss_time[-1:]) })
            elif self.future_frames == 12:
                self.log_dict({'Sweep/test_loss': np.sum(overall_loss_time), "test/loss_0.8": torch.tensor(overall_loss_time[1:2]), "test/loss_2": torch.tensor(overall_loss_time[4:5]), "test/loss_2.8": torch.tensor(overall_loss_time[6:7]), "test/loss_4": torch.tensor(overall_loss_time[9:10]), "test/loss_4.8": torch.tensor(overall_loss_time[-1:]) }) #, sync_dist=True
            
    def on_test_epoch_end(self):
        #wandb_logger.experiment.save(run.name + '.ckpt')
        if not self.probabilistic:
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
                self.log_dict({"var_s4": torch.tensor(var[3]), "var_s5": torch.tensor(var[-1])})
            elif self.future_frames == 12:
                self.log_dict({"test/var_2": torch.tensor(var[4:5]), "test/var_2.8": torch.tensor(var[6:7]), "test/var_4": torch.tensor(var[9:10]), "test/var_4.8": torch.tensor(var[-1:]) }) #, sync_dist=True
            

        
        
def sweep_train():

    #wandb.init()
    
    run=wandb.init(project="dbu_graph", config=default_config)
    config = wandb.config

    print('config: ', dict(config))
    wandb_logger = pl_loggers.WandbLogger()  #name=
    
    input_dim = config.input_dim*5 if dataset=='apollo' else config.input_dim*config.history_frames
    output_dim = 2*config.future_frames #if config.probabilistic == False else 5*config.future_frames

    if config.model_type == 'gat':
        hidden_dims = round(config.hidden_dims / config.heads)
        model = SCOUT_MDN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, heads=config.heads, dropout=config.dropout, bn=config.bn, feat_drop=config.feat_drop, attn_drop=config.attn_drop, att_ew=config.att_ew, res_connection=args.res_connection , res_weight=args.res_weight)
        print('INPUT DIM: ',input_dim)
    elif config.model_type == 'gcn':
        model = GCN(in_feats=input_dim, hid_feats=config.hidden_dims, out_feats=output_dim, dropout=config.dropout, gcn_drop=config.gcn_drop, bn=config.bn, gcn_bn=config.gcn_bn, embedding=config.embedding)
    elif config.model_type == 'gated':
        model = GatedGCN(input_dim=input_dim, hidden_dim=config.hidden_dims, output_dim=output_dim, dropout=config.dropout, bn=config.bn)
    elif config.model_type == 'gated_mdn':
        model = Gated_MDN(input_dim=input_dim, hidden_dim=config.hidden_dims, output_dim=output_dim, dropout=config.dropout, bn=config.bn, types=True)
    elif config.model_type == 'baseline':
        model = RNN_baseline(input_dim=5, hidden_dim=config.hidden_dims, output_dim=output_dim, pred_length=config.future_frames, dropout=config.dropout, bn=config.bn)
    elif config.model_type == 'rgcn':
        model = RGCN(in_dim=input_dim, h_dim=config.hidden_dims, out_dim=output_dim, num_rels=3, num_bases=-1, num_hidden_layers=2, embedding=config.embedding, bn=config.bn, dropout=config.dropout)
    
    path = '/home/sandra/PROGRAMAS/DBU_Graph/logs/MDN/swept-sweep-2/epoch=160-step=58120.ckpt'#DGX/APOLLO/helpful-sweep-17/epoch=48-step=3282.ckpt'
    LitGNN_sys = LitGNN(model=model, input_dim=input_dim, lr=config.learning_rate, model_type= config.model_type, wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames, alfa= config.alfa, beta = config.beta, delta=config.delta, prob=config.probabilistic)
    

    #wandb_logger.watch(LitGNN_sys.model)  #log='all' for params & grads
    '''
    ########################### TEST APOLLO MDN ############################
    path = '/home/sandra/PROGRAMAS/DBU_Graph/logs/MDN/swept-sweep-2/epoch=160-step=58120.ckpt'#DGX/APOLLO/helpful-sweep-17/epoch=48-step=3282.ckpt'
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=path,model=LitGNN_sys.model, input_dim=input_dim, lr=config.learning_rate, model_type= config.model_type, wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames, alfa= config.alfa, beta = config.beta, delta=config.delta, prob=config.probabilistic)
    print('############ TEST APOLLO ##############')
    trainer = pl.Trainer(gpus=0, profiler=True)
    trainer.test(LitGNN_sys)
    #########################################################
    '''
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
        train_dataset = ApolloScape_DGLDataset(train_val='train',  test=False) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', test=False)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='val', test=False)  #230
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
            "model_type": args.model,
            "dropout": 0.1,
            "alfa": 0,
            "beta": 0,
            "delta": 0.,
            "feat_drop": 0.,
            "attn_drop":0.,
            "bn":False,
            "wd": 0.01,
            "heads": 2,
            "att_ew": True,               
            "gcn_drop": 0.,
            "gcn_bn": True,
            'embedding':True
        }
    #sweep_id = wandb.sweep(sweep_config, project="dbu_graph", entity='sandracl72')
    #wandb.agent(sweep_id, sweep_train, count=args.count)
    sweep_train()
