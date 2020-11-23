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



total_epoch = 40
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
    path = '{}/{}_bt{}bv{}_hid{}_lr{}_ep{:03}.pt'.format(work_dir, model_type, batch_train,batch_val, hidden_dims, learning_rate, total_epoch)
    if os.path.exists(path):
        path= '.' + path.split('.')[1] + '_' + str(datetime.now().minute)+ '.pt'
    torch.save(model.state_dict(), path)
    print('Successfully saved to {}'.format(path))

def my_load_model(model, path):
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['graph_model'])
	print('Successfull loaded from {}'.format(path))
	return model


def compute_RMSE(pred, gt, mask): 
    #output mask vale 0 si no visible o visible pero no hay datos en ese frame
    pred = pred*output_mask[0] #Con esto ya quito del error aquellas filas donde no tengo datos.
    gt = features[0][:,6:,3:5]*output_mask[0]  #120 nodos outputmask V,T,C
    xy_error=torch.sum(torch.abs(pred-gt)**2,dim=2) #V,T 
    overall_sum_time = xy_error.sum(dim=0) #T - suma de los errores de los V agentes
    overall_num = output_mask[0].sum(dim=-1).sum(dim=0) #T - num de agentes en cada frame
    return overal_sum_time, overall_num, xy_error

def compute_RMSE_batch(pred, gt, mask): 
    #output mask vale 0 si no visible o no-car o visible pero no hay datos en ese frame  (B*V,T,1), cada fila un nodo de un grafo perteneciente al batch
    pred=pred.view(pred.shape[0],mask.shape[1],-1)
    #gt=gt.view(pred.shape[0],6,-1)
    
    pred = pred*mask #B*V,T,C  (B n grafos en el batch)
    gt = gt*mask  # outputmask BV,T,C
    
    x2y2_error=torch.sum(torch.abs(pred-gt)**2,dim=-1) # x^2+y^2 BV,T
    overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los V agentes
    overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(T)] - num de agentes (Y CON DATOS) en cada frame
    return overall_sum_time, overall_num, x2y2_error

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
        
        for batch_graphs, masks, batch_snorm_n, batch_snorm_e in tqdm(train_dataloader):
            feats = batch_graphs.ndata['x'].float().to(dev)
            feats=feats.view(feats.shape[0],-1)  #Nx18
            batch_e = batch_graphs.edata['w'].float().to(dev)
            #for GATED GCN
            if model == 'gated':
                batch_e=batch_e.view(batch_e.shape[0],1)
            #model = GatedGCN(input_dim=18, hidden_dim=256, output_dim=12).to(dev)
            batch_pred = model(batch_graphs.to(dev), feats, batch_e, batch_snorm_n.to(dev), batch_snorm_e.to(dev))
            #print(batch_pred.shape, masks.shape)

            labels= batch_graphs.ndata['gt'].float().to(dev)
            overall_sum_time, overall_num, _ = compute_RMSE_batch(batch_pred, labels, masks.to(dev))  #(B,6)
            total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))
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
        for batched_graph, output_masks,snorm_n, snorm_e in tqdm(val_dataloader):
            feats = batched_graph.ndata['x'].float().to(dev)
            #reshape to have (B*V,T*C) [c1,c2,...,c6]
            feats = feats.view(feats.shape[0],-1)
            e_w = batched_graph.edata['w'].float().to(dev)
            
            if model_type == 'gated':
                e_w= e_w.view(e_w.shape[0],1)
            
            labels= batched_graph.ndata['gt'].float().to(dev)
            pred = model(batched_graph.to(dev), feats,e_w,snorm_n,snorm_e)
            _, overall_num, x2y2_error = compute_RMSE_batch(pred, labels, output_masks.to(dev))
            #(x2y2_error.shape)  #BV,T
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            #(overall_num.shape)  #BV,T
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  #RMSE para cada nodo en cada T
            
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
        for batched_graph, output_masks,snorm_n, snorm_e in tqdm(test_dataloader):
            feats = batched_graph.ndata['x'].float().to(dev)
            #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
            feats = feats.view(feats.shape[0],-1)
            e_w = batched_graph.edata['w'].float().to(dev)

            if model_type == 'gated':
                e_w= e_w.view(e_w.shape[0],1)

            labels= batched_graph.ndata['gt'].float().to(dev)
            #labels = labels.view(labels.shape[0], -1)
            pred = model(batched_graph.to(dev), feats,e_w,snorm_n,snorm_e)
            _, overall_num, x2y2_error = compute_RMSE_batch(pred, labels, output_masks.to(dev))
            #print(x2y2_error.shape)  #BV,T
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            #print(overall_num.shape)  #BV,T
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  #RMSE para cada nodo en cada T
        
    overall_sum_time=np.sum(overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
    overall_num_time =np.sum(overall_num_list, axis=0)
    overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame

    print('|{}| Test_RMSE: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]])))
    wandb.log({'test/loss_per_sec': overall_loss_time }) 
    wandb.log({'test/log': np.sum(overall_loss_time) }) 
    


def sweep_train():
    wandb.init()
    config = wandb.config
    print('config: ', dict(config))
    
    train_dataloader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=12, collate_fn=collate_batch)
    val_dataloader=DataLoader(val_dataset, batch_size=config.batch_size,  shuffle=False,num_workers=12, collate_fn=collate_batch)

    if config.model_type == 'gat':
        model = My_GAT(input_dim=24, hidden_dim=config.hidden_dims, output_dim=12,dropout=config.dropout, bn=config.bn, bn_gat=config.bn_gat, feat_drop=config.feat_drop, attn_drop=config.attn_drop).to(dev)
    elif config.model_type == 'gcn':
        model = model = GCN(in_feats=24, hid_feats=config.hidden_dims, out_feats=12, dropout=config.dropout).to(dev)
    elif config.model_type == 'gated':
        model = GatedGCN(input_dim=24, hidden_dim=config.hidden_dims, output_dim=12).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("############### TRAIN ####################")
    train(model, config.model_type, train_dataloader,val_dataloader ,opt)
    print("############### TEST ####################")
    test(model,config.model_type, test_dataloader )



if __name__ == '__main__':

    train_dataset = ApolloScape_DGLDataset(train_val='train') #3447
    val_dataset = ApolloScape_DGLDataset(train_val='val')  #919
    test_dataset = ApolloScape_DGLDataset(train_val='test')  #230

    
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch)

    sweep_config = {
    "name": "Sweep gat dropout",
    "method": "grid",
    "metric": {
    'name': 'val/Loss',
    'goal': 'minimize'   
            },
    "parameters": {
            "batch_size": {
                "values": [64]
            },
            "hidden_dims": {
                "values": [256]
            },
            "model_type": {
                "values": ['gat']
            },
            "dropout": {
                "values": [0.25]
            },
            "feat_drop": {
                "values": [0.25, 0.1]
            },
            "attn_drop": {
                "values": [0.25, 0.1]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="dbu_graph")

    wandb.agent(sweep_id, sweep_train)
    '''
    if model_type == 'gat':
        model = My_GAT(input_dim=18, hidden_dim=hidden_dims, output_dim=12).to(dev)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=18, hid_feats=hidden_dims, out_feats=12).to(dev)
    elif model_type == 'gated':
        model = GatedGCN(input_dim=18, hidden_dim=hidden_dims, output_dim=12).to(dev)

    
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    print("############### TRAIN ####################")
    train(model, train_dataloader,val_dataloader ,opt)
    print("############### TEST ####################")
    test(model,test_dataloader )

    '''