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
from tqdm import tqdm
import random

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


history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second
total_epoch = 40
base_lr = 0.001
hidden_dims = 256
model_type = 'gat' #gcn
batch_train=64
batch_val=32
lr_decay_epoch = 5
dev = 'cuda' 
work_dir = './models_checkpoints'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

if not os.path.exists(work_dir):
	os.makedirs(work_dir)

def collate_batch(samples):
    graphs, masks, last_vis_obj = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
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
    return batched_graph.to(dev), masks.to(dev), snorm_n.to(dev), snorm_e.to(dev), last_vis_obj

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
    path = '{}/{}_bt{}bv{}_hid{}_lr{}_ep{:03}.pt'.format(work_dir, model_type, batch_train,batch_val, hidden_dims, base_lr, total_epoch)
    if os.path.exists(path):
        path= path + '_' + str(datetime.now().minute)
    torch.save(model.state_dict(), path)
    print('Successfully saved to {}'.format(path))

def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successfull loaded from {}'.format(pra_path))
	return pra_model


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

def train(model, train_dataloader, val_dataloader, opt):
    train_loss_sum=[]
    val_loss_sum=[]
    val_loss_prev=0
    for epoch in range(total_epoch):
        print('Epoch: ',epoch)
        overall_loss_train=[]
        model.train()
        for batched_graph, output_masks,snorm_n, snorm_e,last_vis_obj in tqdm(train_dataloader):
            feats = batched_graph.ndata['x'].float().to(dev)
            #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
            feats = feats.view(feats.shape[0],-1)
            e_w = batched_graph.edata['w'].float().to(dev)
            
            #for GatedGCN
            #e_w= e_w.view(e_w.shape[0],1)
            
            labels= batched_graph.ndata['gt'].float().to(dev)
            pred = model(batched_graph, feats,e_w,snorm_n,snorm_e)   #70,6,2
            overall_sum_time, overall_num, _ = compute_RMSE_batch(pred, labels, output_masks)  #(B,6)
            total_loss=torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))
            opt.zero_grad() 
            total_loss.backward()
            opt.step()
            overall_loss_train.extend([total_loss.data.item()])
        
        #print('|{}| Train_loss: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_train) + [np.sum(overall_loss_train)]])))
        print('|{}| Train_loss: {}'.format(datetime.now(), np.sum(overall_loss_train)/len(overall_loss_train)))
        train_loss_sum.append(np.sum(overall_loss_train)/len(overall_loss_train))

        val(model, val_dataloader, val_loss_sum)

        if val_loss_prev < val_loss_sum[-1] and epoch !=0:
            patience+=1
            val_loss_prev = val_loss_sum[-1]
        else:
            patience = 0
            val_loss_prev = val_loss_sum[-1]
        if patience > 2:
            print("Early stopping: ")
            print("Difference: {}".format(val_loss_prev-val_loss_sum[-1]))
            break

    print('Val loss sum: {}'.format(val_loss_sum))
    epochs = list(range(total_epoch))
    plt.subplot(1,2,1)
    plt.plot(epochs,train_loss_sum)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs,val_loss_sum)
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.show()

    my_save_model(model)

def val(model, val_dataloader,val_loss_sum):
    model.eval()
    with torch.no_grad():
        overall_num_list=[] 
        overall_x2y2_list=[]
        for batched_graph, output_masks,snorm_n, snorm_e,last_vis_obj in tqdm(val_dataloader):
            feats = batched_graph.ndata['x'].float().to(dev)
            #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
            feats = feats.view(feats.shape[0],-1)
            e_w = batched_graph.edata['w'].float().to(dev)
            
            #for GatedGCN
            #e_w= e_w.view(e_w.shape[0],1)
            
            labels= batched_graph.ndata['gt'][:,:,:].float().to(dev)
            #labels = labels.view(labels.shape[0], -1)
            pred = model(batched_graph, feats,e_w,snorm_n,snorm_e)
            _, overall_num, x2y2_error = compute_RMSE_batch(pred, labels, output_masks[:,:,:])
            #print(x2y2_error.shape)  #BV,T
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            #print(overall_num.shape)  #BV,T
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  #RMSE para cada nodo en cada T
            
    overall_sum_time=np.sum(overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
    overall_num_time =np.sum(overall_num_list, axis=0)
    overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame

    print('|{}| Val_loss: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]])))
    val_loss_sum.append(np.sum(overall_loss_time))


	
def test(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        overall_num_list=[] 
        overall_x2y2_list=[]
        for batched_graph, output_masks,snorm_n, snorm_e,last_vis_obj in tqdm(test_dataloader):
            feats = batched_graph.ndata['x'].float().to(dev)
            #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
            feats = feats.view(feats.shape[0],-1)
            e_w = batched_graph.edata['w'].float().to(dev)

            #for GatedGCN
            #e_w= e_w.view(e_w.shape[0],1)

            labels= batched_graph.ndata['gt'][:,:,:].float().to(dev)
            #labels = labels.view(labels.shape[0], -1)
            pred = model(batched_graph, feats,e_w,snorm_n,snorm_e)
            _, overall_num, x2y2_error = compute_RMSE_batch(pred, labels, output_masks[:,:,:])
            #print(x2y2_error.shape)  #BV,T
            overall_num_list.extend(overall_num.detach().cpu().numpy())
            #print(overall_num.shape)  #BV,T
            overall_x2y2_list.extend((x2y2_error**0.5).detach().cpu().numpy())  #RMSE para cada nodo en cada T

    overall_sum_time=np.sum(overall_x2y2_list,axis=0)  #BV,T->T RMSE medio en cada T
    overall_num_time =np.sum(overall_num_list, axis=0)
    overall_loss_time=(overall_sum_time / overall_num_time) #media del error de cada agente en cada frame

    print('|{}| Test_RMSE: {}'.format(datetime.now(), ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]])))


if __name__ == '__main__':

    train_dataset = ApolloScape_DGLDataset(train_val='train') #3447
    val_dataset = ApolloScape_DGLDataset(train_val='val')  #919
    test_dataset = ApolloScape_DGLDataset(train_val='test')  #230

    train_dataloader=DataLoader(train_dataset, batch_size=batch_train, shuffle=False, collate_fn=collate_batch)
    val_dataloader=DataLoader(val_dataset, batch_size=batch_val, shuffle=False, collate_fn=collate_batch)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    if model_type == 'gat':
        model = My_GAT(input_dim=18, hidden_dim=hidden_dims, output_dim=12).to(dev)
    elif model_type == 'gcn':
        model = model = GCN(in_feats=18, hid_feats=hidden_dims, out_feats=12).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    print("############### TRAIN ####################")
    train(model, train_dataloader,val_dataloader ,opt)
    print("############### TEST ####################")
    test(model,test_dataloader )