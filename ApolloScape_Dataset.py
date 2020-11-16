import numpy as np
import dgl
import random
import pickle
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import datasets, transforms
import scipy.sparse as spp
from dgl.data import DGLDataset


class ApolloScape_DGLDataset(torch.utils.data.Dataset):
    _raw_dir = '/home/sandra/PROGRAMAS/DBU_Graph/data/apollo_train_data.pkl'
    '''
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(ApolloScape_DGLDataset, self).__init__(name='apolloscape',
                                        url=url,
                                        raw_dir=self._raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    '''
    def __init__(self, train_val, data_path=None):
        self.raw_dir='/home/sandra/PROGRAMAS/DBU_Graph/data/apollo_train_data.pkl'
        self.train_val=train_val
        self.process()        

    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            # Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
            [all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)
        all_feature=np.transpose(all_feature, (0,3,2,1)) #(N,V,T,C)
        self.all_feature=torch.from_numpy(all_feature[:,:70,:,:]).type(torch.float32)


    def process(self):
        #process data to graph, labels, and splitting masks

        self.load_data()
        total_num = len(self.all_feature)
        
        self.last_vis_obj=[]   #contains number of visible objects in each sequence of the training, i.e. objects in frame 5
        #para hacer grafos de tamaño variable
        for idx in range(len(self.all_adjacency)): 
            for i in range(len(self.all_adjacency[idx])): 
                if self.all_adjacency[idx][i,i] == 0:
                    self.last_vis_obj.append(i)
                    break   
        
        feature_id = [3, 4, 9]#, 10]  #x,y,heading,[visible_mask]
        now_history_frame=6
        object_type = self.all_feature[:,:,:,2].int()  # torch Tensor NxVxT
        mask_car=np.zeros((total_num,self.all_feature.shape[1],now_history_frame)) #NxVx6
        for i in range(total_num):
            mask_car_t=np.array([1  if (j==2 or j==1) else 0 for j in object_type[i,:,5]])
            mask_car[i,:]=np.array(mask_car_t).reshape(mask_car.shape[1],1)+np.zeros(6) #120x6
        
        self.node_features = self.all_feature[:,:,:now_history_frame,feature_id]  #obj type,x,y 6 primeros frames
        self.node_labels=self.all_feature[:,:,now_history_frame:,3:5] #x,y 6 ultimos frames
        self.node_features[:,:,:,-1] *= mask_car   #Pongo 0 en feat 11 [mask] a todos los obj visibles no-car
        self.node_labels[:,:,:,-1] *= mask_car
        self.output_mask= self.all_feature[:,:,6:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,6,1)
        self.output_mask = np.array(self.output_mask.unsqueeze_(-1) )

        #EDGES weights  #5010x120x120[]
        self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,5,:], self.node_features[i][:,5,:]) for i in range(len(self.all_feature))]  #5010x70x70

        # TRAIN VAL SETS
        # Remove empty rows from output mask 
        zero_indeces_list = [i for i in range(len(self.output_mask)) if np.all(np.array(self.output_mask.squeeze(-1))==0, axis=(1,2))[i] == True ]

        id_list = list(set(list(range(total_num))) - set(zero_indeces_list))
        total_valid_num = len(id_list)
        ind=np.random.permutation(id_list)
        self.train_id_list, self.val_id_list, self.test_id_list = ind[:round(total_valid_num*0.75)], ind[round(total_valid_num*0.75):round(total_valid_num*0.95)],ind[round(total_valid_num*0.95):]

        #train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
        #val_id_list = list(set(list(range(total_num))) - set(train_id_list))  
        '''
        if self.train_val.lower() == 'train':
            self.node_features = self.node_features[self.train_id_list]
            self.node_labels = self.node_labels[self.train_id_list]
            self.all_adjacency = self.all_adjacency[self.train_id_list]
            self.output_mask = self.output_mask[self.train_id_list]
            self.all_mean_xy = self.all_mean_xy[self.train_id_list]
        elif self.train_val.lower() == 'val':
            self.node_features = self.node_features[self.val_id_list]
            self.node_labels = self.node_labels[self.val_id_list]
            self.all_adjacency = self.all_adjacency[self.val_id_list]
            self.output_mask = self.output_mask[self.val_id_list]
            self.all_mean_xy = self.all_mean_xy[self.val_id_list]
        '''

    def __len__(self):
        if self.train_val.lower() == 'train':
            return len(self.train_id_list)
        elif self.train_val.lower() == 'val':
            return len(self.val_id_list)
        else:
            return len(self.test_id_list)

    def __getitem__(self, idx):
        if self.train_val.lower() == 'train':
            idx = self.train_id_list[idx]
        elif self.train_val.lower() == 'val':
            idx = self.val_id_list[idx]
        else:
            idx = self.test_id_list[idx]
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:self.last_vis_obj[idx],:self.last_vis_obj[idx]])).int()
        graph = dgl.remove_self_loop(graph)
        
        for n in graph.nodes():
            if graph.in_degrees(n) == 0:
                graph.add_edges(n,n)
        
        #graph = dgl.add_self_loop(graph)
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        norm_distances = [(i-min(distances))/(max(distances)-min(distances)) if (max(distances)-min(distances))!=0 else (i-min(distances))/1.0 for i in distances]
        norm_distances = [1/(i) if i!=0 else 1 for i in distances]
        graph.edata['w']=torch.tensor(norm_distances, dtype=torch.float32)
        graph.ndata['x']=self.node_features[idx,:self.last_vis_obj[idx]] #obj type, x, y
        graph.ndata['gt']=self.node_labels[idx,:self.last_vis_obj[idx]]
        output_mask = self.output_mask[idx,:self.last_vis_obj[idx]]
        
        return graph, output_mask, self.last_vis_obj[idx]