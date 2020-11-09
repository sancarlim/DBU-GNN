import pickle
import dgl
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import scipy.sparse as spp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda' 
work_dir = './trained_models'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

if not os.path.exists(work_dir):
	os.makedirs(work_dir)

def my_print(content):
	with open(log_file, 'a') as writer:
		print(content)
		writer.write(pra_content+'\n')

def display_result(results, pra_pref='Train_epoch'):
	all_overall_sum_list, all_overall_num_list = pra_results
	overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0)
	overall_num_time = np.sum(all_overall_num_list, axis=0)
	overall_loss_time = (overall_sum_time / overall_num_time) 
	overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
	my_print(overall_log)
	return overall_loss_time
	

def my_save_model(pra_model, pra_epoch):
	path = '{}/model_epoch_{:04}.pt'.format(work_dir, pra_epoch)
	torch.save(
		{
			'xin_graph_seq2seq_model': pra_model.state_dict(),
		}, 
		path)
	print('Successfull saved to {}'.format(path))

def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successfull loaded from {}'.format(pra_path))
	return pra_model


def preprocess_data():

def compute_RMSE(pred, gt, mask): 
    #output mask vale 0 si no visible o visible pero no hay datos en ese frame
    pred = pred*output_mask[0] #Con esto ya quito del error aquellas filas donde no tengo datos.
    gt = features[0][:,6:,3:5]*output_mask[0]  #120 nodos outputmask V,T,C
    xy_error=torch.sum(torch.abs(pred-gt)**2,dim=2) #V,T 
    overall_sum_time = xy_error.sum(dim=0) #T - suma de los errores de los V agentes
    overall_num = output_mask[0].sum(dim=-1).sum(dim=0) #T - num de agentes en cada frame
    #return overal_sum_time, overall_num, xy_error


with open('../DBU_Graph/data/apollo_train_data.pkl', 'rb') as reader:
    [feat,adj, mean]=pickle.load(reader)

sparse = spp.coo_matrix(adj[0])  #nodos y edges para formar DGL


if __name__ == '__main__':
    model = Model(in_feats, hidden_feats, out_feats)
    model.to(dev)