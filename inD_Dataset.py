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
from sklearn.preprocessing import StandardScaler

#history_frames = 3 # 5 second * 1 frame/second
#future_frames = 3 # 3 second * 1 frame/second
#total_frames = history_frames + future_frames
neighbor_distance = 10
max_num_object = 120 #per frame
total_feature_dimension = 12 #pos,heading,vel,recording_id,frame,id, l,w, class, mask

class inD_DGLDataset(torch.utils.data.Dataset):

    def __init__(self, train_val, history_frames, future_frames, data_path=None):
        self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_data.pkl'
        self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_test_data.pkl'
        self.train_val=train_val
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.process()        

    def load_data(self):
        if self.train_val.lower() == 'train':
            with open(self.raw_dir_train, 'rb') as reader:
                [all_feature_train, self.all_adjacency_train, self.all_mean_xy_train]= pickle.load(reader)
            all_feature_train=np.transpose(all_feature_train, (0,3,2,1)) #(N,V,T,C)
            #Choose frames in each sequence
            self.all_feature_train=torch.from_numpy(all_feature_train[:,:50,:self.total_frames,:]).type(torch.float32)#.to('cuda')
        else:
            with open(self.raw_dir_val, 'rb') as reader:
                [all_feature_val, self.all_adjacency_val, self.all_mean_xy_val]= pickle.load(reader)
            all_feature_val=np.transpose(all_feature_val, (0,3,2,1))
            self.all_feature_val=torch.from_numpy(all_feature_val[:,:50,:self.total_frames,:]).type(torch.float32)

    def process(self):
        self.load_data()
        
        if self.train_val.lower() == 'train':
            total_num = len(self.all_feature_train)
            
            self.last_vis_obj=[]   
            for idx in range(len(self.all_adjacency_train)): 
                for i in range(len(self.all_adjacency_train[idx])): 
                    if self.all_adjacency_train[idx][i,i] == 0:
                        self.last_vis_obj.append(i)
                        break   
            
            now_history_frame=self.history_frames-1
            feature_id = [0,1,2,3,4] #pos heading vel 
            object_type = self.all_feature_train[:,:,:,-2].int()  # torch Tensor NxVxT
            mask_car=torch.zeros((total_num,self.all_feature_train.shape[1],self.total_frames))#.to('cuda') #NxVx12
            for i in range(total_num):
                mask_car_t=torch.Tensor([1 if (j==1) else 0 for j in object_type[i,:,now_history_frame]])#.to('cuda')
                mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
 
            #self.all_feature[:,:,:now_history_frame,3:5] = self.all_feature[:,:,:now_history_frame,3:5]/rescale_xy
            self.node_features = self.all_feature_train[:,:,:self.history_frames,feature_id]  #x,y,heading,vx,vy 5 primeros frames 5s
            self.node_labels=self.all_feature_train[:,:,self.history_frames:,feature_id] #x,y 3 ultimos frames    
            self.output_mask= self.all_feature_train[:,:,:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,T_hist)
            self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
            self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:].cpu(), self.node_features[i][:,now_history_frame,:].cpu()) for i in range(len(self.all_feature_train))]  #5010x70x70

            id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
            self.train_id_list = np.random.permutation(id_list)

        else:
            total_num = len(self.all_feature_val)
            
            self.last_vis_obj=[]   
            for idx in range(len(self.all_adjacency_val)): 
                for i in range(len(self.all_adjacency_val[idx])): 
                    if self.all_adjacency_val[idx][i,i] == 0:
                        self.last_vis_obj.append(i)
                        break   
            
            now_history_frame=self.history_frames-1
            feature_id = [0,1,2,3,4] #pos heading vel 
            object_type = self.all_feature_val[:,:,:,-2].int()  # torch Tensor NxVxT
            mask_car=torch.zeros((total_num,self.all_feature_val.shape[1],self.total_frames))#.to('cuda') #NxVx12
            for i in range(total_num):
                mask_car_t=torch.Tensor([1 if (j==1) else 0 for j in object_type[i,:,now_history_frame]])#.to('cuda')
                mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
 
            self.node_features = self.all_feature_val[:,:,:self.history_frames,feature_id]  #x,y,heading,vx,vy 5 primeros frames 5s
            self.node_labels=self.all_feature_val[:,:,self.history_frames:,feature_id] #x,y 3 ultimos frames    
            self.output_mask= self.all_feature_val[:,:,:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,T_hist)
            self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
            self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:].cpu(), self.node_features[i][:,now_history_frame,:].cpu()) for i in range(len(self.all_feature_val))]  #5010x70x70

            id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
            total_valid_num = len(id_list)
            ind=np.random.permutation(id_list)
            self.val_id_list, self.test_id_list = ind[:round(total_valid_num*0.67)], ind[round(total_valid_num*0.67):]
        
        # TRAIN VAL SETS
        # Remove empty rows from output mask . En inD esto no va a pasar (muchos más agentes)
        #zero_indeces_list = [i for i in range(len(self.output_mask[:,:,history_frames:])) if np.all(self.output_mask[:,:,history_frames:].squeeze(-1).cpu().numpy()==0, axis=(1,2))[i] == True ]

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
            self.all_adjacency = self.all_adjacency_train
        elif self.train_val.lower() == 'val':
            idx = self.val_id_list[idx]
            self.all_adjacency = self.all_adjacency_val
        else:
            idx = self.test_id_list[idx]
            self.all_adjacency = self.all_adjacency_val

        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:self.last_vis_obj[idx],:self.last_vis_obj[idx]])).int()
        graph = dgl.remove_self_loop(graph)
        '''
        for n in graph.nodes():
            if graph.in_degrees(n) == 0:
                graph.add_edges(n,n)
        '''
        '''
        #Data Augmentation
        if self.train_val.lower() == 'train' and np.random.random()>0.5:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                [sin_angle, cos_angle]])

            xy = self.node_features[idx,:self.last_vis_obj[idx],:,:2]   #(V,T,C)
            #num_xy = np.sum(xy.sum(axis=-1).sum(axis=-1) != 0) # get the number of valid data

            # angle_mat: (2, 2), xy: (2, 12, 120)
            out_xy = np.einsum('ab,vtb->vta', angle_mat, xy)
            #now_mean_xy = np.matmul(angle_mat, now_mean_xy)
            xy= out_xy

            self.node_features[idx,:self.last_vis_obj[idx],:,:2] = torch.from_numpy(xy).type(torch.float32)
        '''

        graph = dgl.add_self_loop(graph)#.to('cuda')
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        norm_distances = [(i-min(distances))/(max(distances)-min(distances)) if (max(distances)-min(distances))!=0 else (i-min(distances))/1.0 for i in distances]
        norm_distances = [1/(i) if i!=0 else 1 for i in distances]
        graph.edata['w']=torch.tensor(norm_distances, dtype=torch.float32)#.to('cuda')
        graph.ndata['x']=self.node_features[idx,:self.last_vis_obj[idx]] 
        graph.ndata['gt']=self.node_labels[idx,:self.last_vis_obj[idx]]
        output_mask = self.output_mask[idx,:self.last_vis_obj[idx]]
        
        return graph, output_mask