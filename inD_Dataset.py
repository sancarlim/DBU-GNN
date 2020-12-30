import numpy as np
import dgl
import random
import pickle
import math
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import datasets, transforms
import scipy.sparse as spp
from dgl.data import DGLDataset
from sklearn.preprocessing import StandardScaler

max_num_object = 30 #per frame
total_feature_dimension = 12 #pos,heading,vel,recording_id,frame,id, l,w, class, mask

class inD_DGLDataset(torch.utils.data.Dataset):

    def __init__(self, train_val, history_frames, future_frames, test=False, model_type='gat', data_path=None):
        if future_frames == 3:
            #self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_data_cars_20m3s.pkl'
            #self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_test_data_parked_cars_20m3s.pkl'  ####ind_val_test_data.pkl'
            #if model_type == 'rgcn' or model_type == 'hetero':
            self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_pedcar_3s.pkl' 
            self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_pedcar_3s.pkl' 
        elif future_frames==5:
            self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_data.pkl' #_cars_20m.pkl'
            self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_test_data_parked_cars_20m.pkl' 
        elif future_frames==4:
            self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_1.5Hz4f.pkl'
            self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_test_1.5Hz4f.pkl' 
        else:
            self.raw_dir_train='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_train_2.5Hz8f.pkl'
            self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_val_test_2.5Hz8f.pkl' 
        
        if test:
            self.raw_dir_val='/home/sandra/PROGRAMAS/DBU_Graph/data/ind_test_3s.pkl'
        self.train_val=train_val
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.model_type = model_type
        self.test = test
        self.process()        

    def load_data(self):
        if self.train_val.lower() == 'train':
            with open(self.raw_dir_train, 'rb') as reader:
                [all_feature_train, self.all_adjacency_train, self.all_mean_xy_train, self.all_visible_object_idx]= pickle.load(reader)
            all_feature_train=np.transpose(all_feature_train, (0,3,2,1)) #(N,V,T,C)
            #Choose frames in each sequence
            self.all_feature_train=torch.from_numpy(all_feature_train[:,:,:,:]).type(torch.float32)#.to('cuda')
        else:
            with open(self.raw_dir_val, 'rb') as reader:
                [all_feature_val, self.all_adjacency_val, self.all_mean_xy_val, self.all_visible_object_idx]= pickle.load(reader)
            all_feature_val=np.transpose(all_feature_val, (0,3,2,1))
            self.all_feature_val=torch.from_numpy(all_feature_val[:,:,:,:]).type(torch.float32)

    def process(self):
        self.load_data()
        
        if self.train_val.lower() == 'train':
            total_num = len(self.all_feature_train)
            '''
            self.last_vis_obj=[]   
            for idx in range(len(self.all_adjacency_train)): 
                for i in range(len(self.all_adjacency_train[idx])): 
                    if self.all_adjacency_train[idx][i,i] == 0:
                        self.last_vis_obj.append(i)
                        break   
            '''
            now_history_frame=self.history_frames-1
            feature_id = [0,1,2,3,4] #pos heading vel 
            
            if self.model_type == 'grip':
                feature_id = [0,1,2,-1]
            
            self.object_type = self.all_feature_train[:,:,:,-2].int()  # torch Tensor NxVxT
            mask_car=torch.zeros((total_num,self.all_feature_train.shape[1],self.total_frames))#.to('cuda') #NxVx10
            for i in range(total_num):
                mask_car_t=torch.Tensor([1 if j in (1,2) else 0 for j in self.object_type[i,:,now_history_frame]])#.to('cuda')
                mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
            
            #self.all_feature[:,:,:now_history_frame,3:5] = self.all_feature[:,:,:now_history_frame,3:5]/rescale_xy
            self.node_features = self.all_feature_train[:,:,:self.history_frames,feature_id]#*mask_car[:,:,:self.history_frames].unsqueeze(-1)  #x,y,heading,vx,vy 5 primeros frames 5s
            self.node_labels=self.all_feature_train[:,:,self.history_frames:,:2]#*mask_car[:,:,self.history_frames:].unsqueeze(-1)  #x,y 3 ultimos frames    
            self.output_mask= self.all_feature_train[:,:,:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,T_hist)
            self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
            self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:2].cpu(), self.node_features[i][:,now_history_frame,:2].cpu()) for i in range(len(self.all_feature_train))]  #5010x70x70
            self.vel_l2 = [spatial.distance.cdist(self.node_features[i][:,now_history_frame,-2:].cpu(), self.node_features[i][:,now_history_frame,-2:].cpu()) for i in range(len(self.all_feature_train))]
           
            id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
            self.train_id_list = np.random.permutation(id_list)

        else:
            total_num = len(self.all_feature_val)
            '''
            self.last_vis_obj=[]   
            for idx in range(len(self.all_adjacency_val)): 
                for i in range(len(self.all_adjacency_val[idx])): 
                    if self.all_adjacency_val[idx][i,i] == 0:
                        self.last_vis_obj.append(i)
                        break   
            '''
            now_history_frame=self.history_frames-1
            feature_id = [0,1,2,3,4] #pos heading vel 
            info_feats_id = list(range(5,11))  #recording_id,frame,id, l,w, class
            self.object_type = self.all_feature_val[:,:,:,-2].int()  # torch Tensor NxVxT
            
            mask_car=torch.zeros((total_num,self.all_feature_val.shape[1],self.total_frames))#.to('cuda') #NxVx12
            for i in range(total_num):
                mask_car_t=torch.Tensor([1 if j in (1,2) else 0 for j in self.object_type[i,:,now_history_frame]])#.to('cuda')
                mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
            
            self.node_features = self.all_feature_val[:,:,:self.history_frames,feature_id]#*mask_car[:,:,:self.history_frames].unsqueeze(-1)  #x,y,heading,vx,vy 5 primeros frames 5s
            self.node_labels=self.all_feature_val[:,:,self.history_frames:,:2]#*mask_car[:,:,self.history_frames:].unsqueeze(-1)  #x,y 3 ultimos frames    
            self.output_mask= self.all_feature_val[:,:,:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,T_hist)
            self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
            self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:2].cpu(), self.node_features[i][:,now_history_frame,:2].cpu()) for i in range(len(self.all_feature_val))]  #5010x70x70
            #self.vel_l2 = [spatial.distance.cdist(self.node_features[i][:,now_history_frame,-2:].cpu(), self.node_features[i][:,now_history_frame,-2:].cpu()) for i in range(len(self.all_feature_val))]
            self.track_info = self.all_feature_val[:,:,:,info_feats_id]

            if self.test:
                self.object_type *= mask_car.int() #Keep only v2v v2vru vru2vru rel-types
                self.test_id_list = list(set(list(range(total_num))))
            else:
                id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
                total_valid_num = len(id_list)
                val_ids, test_ids = id_list[:round(total_valid_num*0.67)],id_list[round(total_valid_num*0.67):]
                val_ids=np.random.permutation(val_ids)
                self.val_id_list, self.test_id_list = val_ids, test_ids
        
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
            self.all_mean_xy = self.all_mean_xy_train
        elif self.train_val.lower() == 'val':
            idx = self.val_id_list[idx]
            self.all_adjacency = self.all_adjacency_val
            self.all_mean_xy = self.all_mean_xy_val
        else:
            idx = self.test_id_list[idx]
            self.all_adjacency = self.all_adjacency_val
            self.all_mean_xy = self.all_mean_xy_val
            track_info = self.track_info[idx,self.all_visible_object_idx[idx]]

        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:len(self.all_visible_object_idx[idx]),:len(self.all_visible_object_idx[idx])])).int()
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
        feats = self.node_features[idx,self.all_visible_object_idx[idx]] #graph.ndata['x']
        gt = self.node_labels[idx,self.all_visible_object_idx[idx]]  #graph.ndata['gt']
        output_mask = self.output_mask[idx,self.all_visible_object_idx[idx]]
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        #rel_vels = [self.vel_l2[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        #rel_vels = [1/(i) if i!=0 else 1 for i in rel_vels]          
        distances = [1/(i) if i!=0 else 1 for i in distances]
        norm_distances = [(i-min(distances))/(max(distances)-min(distances)) if (max(distances)-min(distances))!=0 else (i-min(distances))/1.0 for i in distances]
        graph.edata['w'] = torch.tensor(distances, dtype=torch.float32)#.to('cuda')

        if self.model_type == 'rgcn' or self.model_type == 'hetero':
            edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
            rel_types = [self.object_type[idx][u,self.history_frames-1]* self.object_type[idx][v,self.history_frames-1] for u,v in edges_uvs]
            rel_types = [r - math.ceil(r/2) for r in rel_types] #0: car-car  1:car-ped  2:ped-ped
            graph.edata['rel_type'] = torch.tensor(rel_types, dtype=torch.uint8)  

            u,v,eid=graph.all_edges(form='all')
            v_canonical = []
            v_canonical.append(v[np.where(np.array(rel_types)==0)])
            v_canonical.append(v[np.where(np.array(rel_types)==1)])
            v_canonical.append(v[np.where(np.array(rel_types)==2)])
            ew_canonical = []
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==0)])
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==1)])
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==2)])
            # calculate norm for each edge type and store in edge
            graph.edata['norm'] = torch.ones(eid.shape[0],1)  
            if self.model_type == 'hetero':
                u_canonical = []
                u_canonical.append(u[np.where(np.array(rel_types)==0)])
                u_canonical.append(u[np.where(np.array(rel_types)==1)])
                u_canonical.append(u[np.where(np.array(rel_types)==2)])
                graph=dgl.heterograph({
                    ('car', 'v2v', 'car'): (u_canonical[0], v_canonical[0]),
                    ('car', 'v2vru', 'ped'): (u_canonical[1], v_canonical[1]),
                    ('ped', 'vru2vru', 'ped'): (u_canonical[2], v_canonical[2]),
                })
                graph.nodes['drug'].data['hv'] = th.ones(3, 1)
                for v, etype in zip(v_canonical, graph.canonical_etypes):
                    _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
                    degrees = count[inverse_index]
                    norm = torch.ones(v.shape[0]).float() / degrees.float()
                    norm = norm.unsqueeze(1)
                    g.edges[etype].data['norm'] = norm
            else:
                for i,v in enumerate(v_canonical):        
                    _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
                    degrees = count[inverse_index]
                    norm = torch.ones(v.shape[0]).float() / degrees.float()
                    norm = norm.unsqueeze(1)
                    #g.edges[etype].data['norm'] = norm
                    graph.edata['norm'][np.where(np.array(rel_types)==i)] = norm

        if self.model_type == 'grip':
            now_feature = self.node_features[idx].permute(2,1,0) # GRIP (C, T, V) = (N, 11, 12, 120)
            now_gt = self.node_labels[idx]  # V T C
            now_mean_xy = self.all_mean_xy[idx]# (2,) = (x, y) 
            now_adjacency = utils.get_adjacency(A=self.all_adjacency[idx])
            #now_A = utils.normalize_adjacency(now_adjacency)
            now_A = torch.from_numpy(now_adjacency).type(torch.float32)
            output_mask = self.output_mask[idx] # V T C
            return now_feature, now_gt, now_A, now_mean_xy, output_mask

        elif self.test:
            mean_xy = self.all_mean_xy[idx]
            return graph, output_mask, track_info, mean_xy, feats, gt, self.object_type[idx,self.all_visible_object_idx[idx],self.history_frames-1]

        else: 
            return graph, output_mask, feats, gt

if __name__ == "__main__":
    history_frames=3
    future_frames=3
    test_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, test=False, model_type='hetero')
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False) 
    next(iter(test_dataloader))
    