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

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(42)

max_num_object = 30 #per frame
total_feature_dimension = 12 #pos,heading,vel,recording_id,frame,id, l,w, class, mask
def collate_batch(samples):
    graphs, masks, feats, gt = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()

    #masks = masks.view(masks.shape[0],-1)
    #masks= masks.view(masks.shape[0]*masks.shape[1],masks.shape[2],masks.shape[3])#.squeeze(0) para TAMAÑO FIJO
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt


class inD_DGLDataset(torch.utils.data.Dataset):

    def __init__(self, train_val, history_frames, future_frames, test=False, model_type='gat', data_path=None, classes=(1,2)):
        
        self.train_val=train_val
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.model_type = model_type
        self.test = test
        self.classes = classes

        if self.total_frames == 6:
            #if model_type == 'rgcn' or model_type == 'hetero':
            self.raw_dir='/media/14TBDISK/sandra/inD_processed/inD_1Hz_3s_nofilter.pkl'
        if self.total_frames == 8:
            self.raw_dir='/media/14TBDISK/sandra/inD_processed/inD_1Hz_3s5s_nofilter.pkl'
        elif self.total_frames == 10:
            self.raw_dir_train='/media/14TBDISK/sandra/inD_processed/train_allobj5s.pkl'#ind_train_data.pkl' #_cars_20m.pkl'
            self.raw_dir_val='/media/14TBDISK/sandra/inD_processed/val_test_allobj5s.pkl'#ind_val_test_data_parked_cars_20m.pkl' 
        elif self.total_frames == 20 or self.total_frames == 16:
            self.raw_dir='/media/14TBDISK/sandra/inD_processed/inD_2.5Hz8_12f_benchmark_train.pkl' #el obs_frame sigue siendo el 7 , me vale para 8/8
            if self.train_val == 'test':
                self.raw_dir ='/media/14TBDISK/sandra/inD_processed/inD_2.5Hz8_12f_benchmark_test.pkl'
        if self.total_frames == 24:
            self.raw_dir_train='/media/14TBDISK/sandra/inD_processed/inD_train_2.5Hz12_12f.pkl'
            self.raw_dir_val='/media/14TBDISK/sandra/inD_processed/inD_val_test_2.5Hz12_12f.pkl' 
            
        
        self.process()        

    def load_data(self):
        #if self.train_val.lower() == 'train':
        with open(self.raw_dir, 'rb') as reader:
            [all_feature_train, self.all_adjacency, self.all_mean_xy, self.all_visible_object_idx]= pickle.load(reader)
        all_feature_train=np.transpose(all_feature_train, (0,3,2,1)) #(N,V,T,C)
        #Choose frames in each sequence
        self.all_feature_train=torch.from_numpy(all_feature_train[:,:,:self.total_frames,:]).type(torch.float32)#.to('cuda')
        '''
        else: #elif self.train_val.lower() == 'val':
            with open(self.raw_dir_val, 'rb') as reader:
                [all_feature_val, self.all_adjacency_val, self.all_mean_xy_val, self.all_visible_object_idx]= pickle.load(reader)
            all_feature_val=np.transpose(all_feature_val, (0,3,2,1))
            self.all_feature_val=torch.from_numpy(all_feature_val[:,:,:,:]).type(torch.float32)
        '''

    def process(self):
        self.load_data()
        
        #if self.train_val.lower() == 'train':
        total_num = len(self.all_feature_train)
        now_history_frame=self.history_frames-1
        feature_id = [0,1,2,3,4] #pos heading vel 
        info_feats_id = list(range(5,11))  #recording_id,frame,id, l,w, class
        
        if self.model_type == 'grip':
            feature_id = [0,1,2,-1]
        
        self.object_type = self.all_feature_train[:,:,:,-2].int()  # torch Tensor NxVxT
        
        '''
        mask_car=torch.zeros((total_num,self.all_feature_train.shape[1],self.total_frames))#.to('cuda') #NxVx10
        for i in range(total_num):
            mask_car_t=torch.Tensor([1 if j in self.classes else 0 for j in self.object_type[i,:,now_history_frame]])#.to('cuda')
            mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
        '''
        #self.all_feature[:,:,:now_history_frame,3:5] = self.all_feature[:,:,:now_history_frame,3:5]/rescale_xy
        self.node_features = self.all_feature_train[:,:,:self.history_frames,feature_id]#*mask_car[:,:,:self.history_frames].unsqueeze(-1)  #x,y,heading,vx,vy 5 primeros frames 5s
        self.node_labels=self.all_feature_train[:,:,self.history_frames:,:2]#*mask_car[:,:,self.history_frames:].unsqueeze(-1)  #x,y 3 ultimos frames    
        self.track_info = self.all_feature_train[:,:,:,info_feats_id]

        self.output_mask= self.all_feature_train[:,:,:,-1]###*mask_car  #mascara only_cars/peds visibles en 6º frame 
        self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
        self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:2].cpu(), self.node_features[i][:,now_history_frame,:2].cpu()) for i in range(len(self.all_feature_train))]  #5010x70x70
        self.vel_l2 = [spatial.distance.cdist(self.node_features[i][:,now_history_frame,-2:].cpu(), self.node_features[i][:,now_history_frame,-2:].cpu()) for i in range(len(self.all_feature_train))]
        

        id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
        total_valid_num = len(id_list)
        #OPCIÓN A1 / A2
        #self.train_id_list ,self.val_id_list, self.test_id_list = id_list[:round(total_valid_num*0.7)],id_list[round(total_valid_num*0.7):round(total_valid_num*0.9)], id_list[round(total_valid_num*0.9):]
        #self.test_id_list = list(range(np.where(self.track_info[:,0,0,0]==30)[0][0],total_valid_num))
        #id_list = list(set(list(range(total_num))) - set(self.test_id_list))
        #self.train_id_list,self.val_id_list = id_list[:round(total_valid_num*0.8)],id_list[round(total_valid_num*0.8):]
        
        #BENCHMARK
        if self.train_val == 'test':
            self.test_id_list = id_list
        else:
            self.val_id_list = list(range(np.where(self.track_info[:,0,0,0]==0)[0][0],np.where(self.track_info[:,0,0,0]==1)[0][0]))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==7)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==8)[0][0])))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==18)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==19)[0][0])))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==30)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==31)[0][0])))
            self.train_id_list = list(set(id_list)- set(self.val_id_list))

        #TEST ROUND
        '''
        self.train_id_list = id_list
        self.val_id_list = list(range(np.where(self.track_info[:,0,0,0]==1)[0][0],np.where(self.track_info[:,0,0,0]==4)[0][0]))
        print(self.val_id_list[0], self.val_id_list[-1])
        self.test_id_list = list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==0)[0][0], np.where(self.track_info[:,0,0,0]==1)[0][0]))
        print(self.test_id_list[0], self.test_id_list[-1])
        '''
        '''
        else:
            total_num = len(self.all_feature_val)
            now_history_frame=self.history_frames-1
            feature_id = [0,1,2,3,4] #pos heading vel 
            info_feats_id = list(range(5,11))  #recording_id,frame,id, l,w, class
            self.object_type = self.all_feature_val[:,:,:,-2].int()  # torch Tensor NxVxT
            
            mask_car=torch.zeros((total_num,self.all_feature_val.shape[1],self.total_frames))#.to('cuda') #NxVx12
            
            for i in range(total_num):
                mask_car_t=torch.Tensor([1 if j in self.classes else 0 for j in self.object_type[i,:,now_history_frame]])#.to('cuda')
                mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
            
            

            self.node_features = self.all_feature_val[:,:,:self.history_frames,feature_id]#*mask_car[:,:,:self.history_frames].unsqueeze(-1)  #x,y,heading,vx,vy 5 primeros frames 5s
            self.node_labels=self.all_feature_val[:,:,self.history_frames:,:2]#*mask_car[:,:,self.history_frames:].unsqueeze(-1)  #x,y 3 ultimos frames    
            
            self.output_mask= self.all_feature_val[:,:,:,-1]*mask_car #mascara obj (car) visibles en 6º frame (5010,120,T_hist)
            self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
            self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:2].cpu(), self.node_features[i][:,now_history_frame,:2].cpu()) for i in range(len(self.all_feature_val))]  #5010x70x70
            #self.vel_l2 = [spatial.distance.cdist(self.node_features[i][:,now_history_frame,-2:].cpu(), self.node_features[i][:,now_history_frame,-2:].cpu()) for i in range(len(self.all_feature_val))]
            self.track_info = self.all_feature_val[:,:,:,info_feats_id]

            
            id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
            total_valid_num = len(id_list)
            val_ids, test_ids = id_list[:round(total_valid_num*0.67)],id_list[round(total_valid_num*0.67):]
            #val_ids=np.random.permutation(val_ids)
            self.val_id_list, self.test_id_list = val_ids, test_ids
            if self.test:
                self.object_type *= mask_car.int() #Keep only v2v v2vru vru2vru rel-types
                self.test_id_list = list(set(list(range(total_num))))
        
        '''
            
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
            #self.all_adjacency = self.all_adjacency_train
            #self.all_mean_xy = self.all_mean_xy_train
        elif self.train_val.lower() == 'val':
            idx = self.val_id_list[idx]
            #self.all_adjacency = self.all_adjacency_val
            #self.all_mean_xy = self.all_mean_xy_val
        else:
            idx = self.test_id_list[idx]
            #self.all_adjacency = self.all_adjacency_val
            #self.all_mean_xy = self.all_mean_xy_val
            track_info = self.track_info[idx,self.all_visible_object_idx[idx]]

        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:len(self.all_visible_object_idx[idx]),:len(self.all_visible_object_idx[idx])])).int()
        graph = dgl.remove_self_loop(graph)

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
    train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4)) #12281
    val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4))  #3509
    test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4))  #1754
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    test_dataloader=iter(DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch) )
    while(1):
        next(test_dataloader)
    