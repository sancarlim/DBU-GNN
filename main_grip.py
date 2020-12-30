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
from models.stgcn import STGCN
from models.grip_model import GRIPModel
from models.social_stgcn import social_stgcnn
from tqdm import tqdm
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_normalized_adj

#from LIT_system import LitGNN

dataset = 'ind'  #'apollo'

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


total_epoch = 150
learning_rate = 1e-3
hidden_dims = 128
model_type = 'gated' #gcn
batch_train=64
batch_val=64
dev = 'cuda:0'
work_dir = './models_checkpoints'


if not os.path.exists(work_dir):
	os.makedirs(work_dir)

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

def my_print(content):
	with open(log_file, 'a') as writer:
		print(content)
		writer.write(content+'\n')

def display_result(results, pra_pref='Train_epoch'):
	all_overall_sum_list, all_overall_num_list = pra_results
	overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0)
	overall_num_time = np.sum(all_overall_num_list, axis=0)
	overall_loss_time = (overall_sum_time / overall_num_time) 
	overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
	my_print(overall_log)
	return overall_loss_time
	

def my_save_model(model):
    path = '{}/grip_ep{:03}.pt'.format(work_dir, total_epoch)
    if os.path.exists(path):
        path= '.' + path.split('.')[1] + '_' + str(datetime.now().minute)+ '.pt'
    torch.save(model.state_dict(), path)
    print('Successfully saved to {}'.format(path))

def my_load_model(model, path):
    #checkpoint = torch.load(path)
    #model.load_state_dict(checkpoint['graph_model'])
    model.load_state_dict(torch.load(path))
    print('Successfull loaded from {}'.format(path))
    return model

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

def compute_RMSE(pred, gt, mask): 
    #output mask vale 0 si no visible o visible pero no hay datos en ese frame N V T C
    pred = pred*mask #Con esto ya quito del error aquellas filas donde no tengo datos.
    gt = gt*mask
    xy_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) #NVTC -> N,V,T 
    overall_sum_time = xy_error.sum(dim=1) #N,V,T  -> N,T - suma de los errores de los V agentes
    overall_num = mask.sum(dim=-1).sum(dim=1).type(torch.int) #NVTC -> N,T - num de agentes en cada frame
    return overall_sum_time, overall_num, xy_error

def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,:,0]- V_pred[:,:,:,0]
    normy = V_trgt[:,:,:,1]- V_pred[:,:,:,1]

    sx = torch.exp(V_pred[:,:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result
   

def train(model, model_type, train_dataloader, val_dataloader, opt):
    wandb.watch(model, log="all")
    train_loss_sum=[]
    val_loss_sum=[]
    val_loss_prev=0
    n_epochs=0
    for epoch in range(total_epoch):
    
        print("Epoch: ",epoch)
        overall_loss_train=[]
        model.train()
        n_epochs=epoch+1
        
        for feats, labels, A,_,masks in tqdm(train_dataloader):
            feats = feats.float().to('cuda') #N 2 3 70
            labels= labels[:,:,:,:2].float().to('cuda') #N 70 3 2
            last_loc = feats[:,-1:,:2] 
            if dataset.lower() == 'apollo':
                #USE CHANGE IN POS AS INPUT
                feats_vel, labels_vel = compute_change_pos(feats,labels)
                #Input pos + heading + vel
                feats = torch.cat([feats, feats_vel], dim=-1)
            if model_type == 'grip':
                pred = model(feats, A.to('cuda'),pra_pred_length=future_frames).permute(0,3,2,1)  #N 2 3 70 -> N, V, T, C
            else:
                pred,_ = model(feats[:,:2,:,:], A.to('cuda'))
                pred= pred[:,:2,:,:].permute(0,3,2,1) #64 5 3 70 -> N, V, T, C
            overall_sum_time, overall_num, _ = compute_RMSE(pred, labels, masks[:,:,history_frames:total_frames,:].to('cuda'))  #(B,6)
            total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num)
            opt.zero_grad() 
            total_loss.backward()
            #print(model.embedding_h.weight.grad) #model.GatedGCN1.A.weight.grad)
            opt.step()
            overall_loss_train.extend([total_loss.data.item()])
        #print('|{}| Train_loss: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_train) + [np.sum(overall_loss_train)]])))
        print('|{}| Train_loss: {}'.format(datetime.now(), np.sum(overall_loss_train)/len(overall_loss_train)))
        train_loss_sum.append(np.sum(overall_loss_train)/len(overall_loss_train))
        wandb.log({"Train/loss": train_loss_sum[-1]}, step=epoch)

        val(model,  model_type, val_dataloader, val_loss_sum, epoch)

        if val_loss_prev < val_loss_sum[-1] and epoch !=0:
            patience+=1
            val_loss_prev = val_loss_sum[-1]
        else:
            patience = 0
            val_loss_prev = val_loss_sum[-1]
        if patience > 2:
            print("Early stopping")
            break

    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    my_save_model(model)

    epochs = list(range(n_epochs))
    plt.subplot(1,2,1)
    plt.plot(epochs,train_loss_sum)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs,val_loss_sum)
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.show()

def val(model,  model_type, val_dataloader,val_loss_sum, epoch):
    model.eval()
    with torch.no_grad():
        overall_num_list=[] 
        overall_x2y2_list=[]
        for feats, labels, A,_,masks in tqdm(train_dataloader):
            feats = feats.float().to('cuda') #N 2 3 70
            labels= labels[:,:,:,:2].float().to('cuda') #N 2 3 70
            last_loc = feats[:,-1:,:2] 
            if dataset.lower() == 'apollo':
                #USE CHANGE IN POS AS INPUT
                feats_vel, labels_vel = compute_change_pos(feats,labels)
                #Input pos + heading + vel
                feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)

            if model_type == 'grip':
                pred = model(feats, A.to('cuda'),pra_pred_length=future_frames).permute(0,3,2,1)  #N 2 3 70 -> N, V, T, C
            else:
                pred,_ = model(feats[:,:2,:,:], A.to('cuda'))
                pred= pred[:,:2,:,:].permute(0,3,2,1) #64 5 3 70 -> N, V, T, C
            _ , overall_num, x2y2_error = compute_RMSE(pred, labels, masks[:,:,history_frames:total_frames,:].to(dev))
            
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy().sum(axis=1))  #RMSE para cada nodo en cada T
            
    overall_sum_time=np.sum(overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
    overall_num_time =np.sum(overall_num_list, axis=0)
    overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame

    print('|{}| Val_loss: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]])))
    val_loss_sum.append(np.sum(overall_loss_time))
    wandb.log({'Val/Loss': np.sum(overall_loss_time) }, step=epoch)

	
def test(model, model_type, test_dataloader):
    model.eval()
    with torch.no_grad():
        overall_num_list=[] 
        overall_x2y2_list=[]
        for feats, labels, A,_,masks in tqdm(train_dataloader):
            feats = feats.float().to('cuda') #N 2 3 70
            labels= labels[:,:,:,:2].float().to('cuda') #N 2 3 70
            last_loc = feats[:,-1:,:2] 
            if dataset.lower() == 'apollo':
                #USE CHANGE IN POS AS INPUT
                feats_vel, labels_vel = compute_change_pos(feats,labels)
                #Input pos + heading + vel
                feats = torch.cat([feats[:,:,:], feats_vel], dim=-1)

            if model_type == 'grip':
                pred = model(feats, A.to('cuda'),pra_pred_length=future_frames).permute(0,3,2,1)  #N 2 3 70 -> N, V, T, C
            else:
                pred,_ = model(feats[:,:2,:,:], A.to('cuda'))
                pred= pred[:,:2,:,:].permute(0,3,2,1) #64 2 3 70 -> N, V, T, C
            #pred=pred.view(pred.shape[0],labels.shape[1],-1)
            _ , overall_num, x2y2_error = compute_RMSE(pred, labels, masks[:,:,history_frames:total_frames,:].to(dev))
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy().sum(axis=1))  #RMSE para cada nodo en cada T
        
    overall_sum_time=np.sum(overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
    overall_num_time =np.sum(overall_num_list, axis=0)
    overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame

    print('|{}| Test_RMSE: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]])))
    wandb.log({'test/loss_per_sec': overall_loss_time }) 
    wandb.log({'test/loss': np.sum(overall_loss_time) }) 
    

if __name__ == '__main__':

    history_frames=3
    future_frames=3
    model_type = 'social'
    total_frames = history_frames+future_frames

    if dataset.lower() == 'apollo':
        train_dataset = ApolloScape_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, grip_model=True) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, grip_model=True)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, grip_model=True)  #230
    elif dataset.lower() == 'ind':
        train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, grip_model=True) #12281
        val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, grip_model=True)  #3509
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, grip_model=True)  #1754
        
    train_dataloader=DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=12)
    val_dataloader=DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12) 


    hyperparameters_default={
        'learning_rate': 1e-3,
        'history_frames': history_frames,
        'future_frames': future_frames
    }

    wandb.init(project="dbu_graph", config= hyperparameters_default) 
    config = wandb.config
    print('config: ', dict(config))
    #init model
    if model_type == 'grip':
        model = GRIPModel(in_channels=4, num_node=70, edge_importance_weighting=False).to(dev)
    else:
        model = social_stgcnn(input_feat=2,output_feat=2, seq_len=history_frames,pred_seq_len=future_frames).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("############### TRAIN ####################")
    train(model, model_type, train_dataloader,val_dataloader ,opt)
    print("############### TEST ####################")
    
    #my_load_model(model, './models_checkpoints/grip_1.pt')
    test(model,model_type, test_dataloader )

    '''
    wandb_logger = pl_loggers.WandbLogger(save_dir='./logs/')  
    LitGNN_sys = LitGNN(model=model, lr=config.learning_rate, model_type= 'grip', wd=0., history_frames=config.history_frames, future_frames= config.future_frames)

    print("############### TRAIN ####################")
    early_stop_callback = EarlyStopping('Sweep/val_loss')
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=wandb_logger, precision=16, default_root_dir='./models_checkpoints/', callbacks=[early_stop_callback], profiler=True)  #precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20, 
    trainer.fit(LitGCN_sys, train_dataloader, val_dataloader)   

    print("############### TEST ####################")
    trainer.test(test_dataloaders=test_dataloader)
    '''