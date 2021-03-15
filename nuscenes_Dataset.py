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

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction

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


class nuscenes_Dataset(torch.utils.data.Dataset):

    def __init__(self, train_val_test, history_frames, future_frames, data_path=None, classes=(1,2,3), rel_types=False):
        '''
            :classes:   categories to take into account
            :rel_types: wether to include relationship types in edge features 
        '''
        self.train_val_test=train_val_test
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.classes = classes
        self.types = rel_types

        if self.train_val_test == 'train':
            self.raw_dir='/media/14TBDISK/sandra/nuscenes_processed/nuscenes_challenge_train.pkl'
        elif self.train_val_test == 'val':  
            self.raw_dir ='/media/14TBDISK/sandra/nuscenes_processed/nuscenes_challenge_val.pkl'  
        else:
            self.raw_dir ='/media/14TBDISK/sandra/nuscenes_processed/nuscenes_challenge_test.pkl'  

        self.load_data()
        self.process()        

    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            [self.all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)

    def process(self):
        '''
        INPUT:
            :all_feature:   x,y_global (zero centralized per sequence), heading, vel, accel, heading_chang_rate, past_xy_local (4f*2), future_xy_local(12f*2), 
                                    mask(12f), type, l,w,h, frame_id, scene_id, node_token (str), sample_token (str) ---> (58) 
            :all_mean_xy:   mean_xy per sequence for zero centralization
            :all_adjacency: Adjacency matrix per sequence for building graph
        RETURNS:
            :node_feats :  x_y_global, past_x_y, heading,vel,accel,heading_change_rate, type (2+8+5 = 15 in_features)
            :node_labels:  future_xy_local (24)
            :output_mask:  mask (12)
            :track_info :  frame, scene_id, node_token, sample_token (4)
        '''
        
        total_num = len(self.all_feature)
        print(f"{self.train_val_test} split has {total_num} sequences.")
        now_history_frame=self.history_frames-1
        self.feature_id = list(range(0,14)) + [-8] 
        self.labels_id = list(range(14,38))
        self.mask_id = list(range(38,50))
        
        #rescale_xy=torch.ones((1,1,1,2))
        #rescale_xy[:,:,0] = torch.max(abs(self.all_feature[:,:,:,0]))  #121 
        #rescale_xy[:,:,1] = torch.max(abs(self.all_feature[:,:,:,1]))   #77   
        #rescale_xy=torch.ones((1,1,2))*10
        #self.all_feature[:,:now_history_frame+1,:2] = self.all_feature[:,:now_history_frame+1,:2]/rescale_xy

        self.xy_dist=[spatial.distance.cdist(self.all_feature[i][:,:2], self.all_feature[i][:,:2]) for i in range(len(self.all_feature))]  #NxV_ixV_i
        

    def __len__(self):
            return len(self.all_feature)

    def __getitem__(self, idx):
        object_type = torch.tensor(self.all_feature[idx][:,-8], dtype=torch.int)
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx])).int()
        graph = dgl.remove_self_loop(graph)
        edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
        rel_types = [(object_type[u]* object_type[v])for u,v in edges_uvs]
        graph = dgl.add_self_loop(graph)
        rel_types.extend(torch.zeros_like(rel_types[0], dtype=torch.float32) for i in range(graph.num_nodes()))
        feats = self.all_feature[idx][:,self.feature_id] 
        gt = self.all_feature[idx][:,self.labels_id]  
        output_mask = self.all_feature[idx][:,self.mask_id]
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        #rel_vels = [self.vel_l2[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        distances = [1/(i) if i!=0 else 1 for i in distances]
        if self.types:
            #rel_vels =  F.softmax(torch.tensor(rel_vels, dtype=torch.float32), dim=0)
            distances = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)
            graph.edata['w'] = torch.tensor([[distances[i],rel_types[i]] for i in range(len(rel_types))], dtype=torch.float32)
        else:
            graph.edata['w'] = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)

        
        return graph, output_mask, feats, gt

if __name__ == "__main__":
    
    train_dataset = nuscenes_Dataset(train_val_test='train', history_frames=history_frames, future_frames=future_frames)  #3509
    #test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4))  #1754
    train_dataloader=iter(DataLoader(train_dataset, batch_size=5, shuffle=False, collate_fn=collate_batch) )
    while(1):
        batched_graph, masks, snorm_n, snorm_e, feats, gt = next(train_dataloader)
        print(feats.shape)
    