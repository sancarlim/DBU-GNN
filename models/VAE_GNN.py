import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torch.utils.data import DataLoader

class MLP_Enc(nn.Module):
    "Encoder: MLP that takes GNN output as input and returns mu and log variance of the latent distribution."
    "The stddev of the distribution is treated as the log of the variance of the normal distribution for numerical stability."
    def __init__(self, in_dim, z_dim, dropout=0.2):
        super(MLP_Enc, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.linear = nn.Linear(in_dim, in_dim//2)
        self.log_var = nn.Linear(in_dim//2, z_dim)
        self.mu = nn.Linear(in_dim//2, z_dim)
        self.dropout_l = nn.Dropout(dropout)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.normal_(self.log_var.weight, 0, sqrt(1. / self.sigma.in_dim))
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu')

    def forward(self, h, gt):
        h = torch.cat([h, gt], dim=-1) #concatenate gnn output and ground-truth
        h = self.dropout_l(h)
        h = F.leaky_relu(self.linear(h))
        log_var = self.log_var(h) 
        mu = self.mu(h)
        return mu, log_var


class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats,  relu=True, feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        self.relu = relu
        if att_ew:
            self.attention_func = nn.Linear(3 * out_feats, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.feat_drop_l = nn.Dropout(feat_drop)
        self.attn_drop_l = nn.Dropout(attn_drop)   
        self.res_con = res_connection
        self.reset_parameters()
      
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.kaiming_normal_(self.linear_self.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_func.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.attention_func.weight, a=0.01, nonlinearity='leaky_relu')
    
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1) #(n_edg,hid)||(n_edg,hid) -> (n_edg,2*hid) 
        
        if self.att_ew:
           concat_z = torch.cat([edges.src['z'], edges.dst['z'], edges.data['w']], dim=-1) 
        
        src_e = self.attention_func(concat_z)  #(n_edg, 1) att logit
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        h_s = nodes.data['h_s']      
        #Attention score
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h,snorm_n):
        with g.local_scope():
            h_in = h.clone()
            g.ndata['h']  = h 
            #feat dropout
            h=self.feat_drop_l(h)
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h =  g.ndata['h'] #+g.ndata['h_s'] 
            #h = h * snorm_n # normalize activation w.r.t. graph node size
            if self.relu:
                h = torch.relu(h)            
            if self.res_con:
                h = h_in + h # residual connection           
            return h #graph.ndata.pop('h') - another option to g.local_scope()


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, relu=True, merge='cat',  feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection))
        self.merge = merge

    def forward(self, g, h, snorm_n):
        head_outs = [attn_head(g, h,snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs))

    
class GAT_VAE(nn.Module):
    def __init__(self, hidden_dim, fc=False, dropout=0.2, feat_drop=0., attn_drop=0., heads=1,att_ew=False, ew_dims=1):
        super().__init__()
        self.heads = heads
        self.fc = fc
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop,att_ew, res_weight=res_weight, res_connection=res_connection ) 
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0.,att_ew, res_weight=res_weight, res_connection=res_connection ) 
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, res_weight=True, res_connection=True , num_heads=heads,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)            
            self.embedding_e = nn.Linear(ew_dims, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads,hidden_dim*heads, res_weight=True , res_connection=True ,num_heads=1, feat_drop=0., attn_drop=0., att_ew=att_ew)    
        if fc:
            self.dropout_l = nn.Dropout(dropout)
            self.linear = nn.Linear(self.gat_2.out_feats, self.gat_2.out_feats//2)
            nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu') 
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""    
        if self.heads > 1:
            nn.init.xavier_normal_(self.embedding_e.weight)
    
    def forward(self, g, h,e_w,snorm_n):
        h = self.gat_1(g, h,snorm_n) 
        if self.heads > 1:
            e = self.embedding_e(e_w)
            g.edata['w']=e
        h = self.gat_2(g, h, snorm_n) 
        if self.fc:
            h = self.dropout_l(h)
            h=F.leaky_relu(self.linear1(h))
        return h
    
class VAE_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, output_dim, fc=False, dropout=0.2, feat_drop=0., 
                    attn_drop=0., heads=1,att_ew=False, ew_dims=1, map_encoding=False):
        super().__init__()
        self.heads = heads
        self.fc = fc
        self.z_dim = z_dim
        self.map_encoding = map_encoding

        # Encode HD Maps
        if self.map_encoding:
            model_ft = resnet18(pretrained=True)
            self.feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
            ct=0
            for child in self.feature_extractor.children():
                ct+=1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False
            
            self.linear_cat = nn.Linear(hidden_dim + 512, hidden_dim) 

        #Input embeddings
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(ew_dims, hidden_dim)
        
        #Input GNN for encoding interactions
        self.GNN_inp = GAT_VAE(hidden_dim, fc=fc, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)

        #Encode ground_truth trajectories
        self.embedding_gt = nn.Linear(output_dim, hidden_dim)
        self.GNN_enc_gt = GAT_VAE(hidden_dim, fc=fc, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=False, ew_dims=ew_dims) #Not using e_w for calculating attention
        
        #ENCODER
        input_enc_dims = hidden_dim*heads if fc else hidden_dim*heads*2 
        self.encoder = MLP_Enc(input_enc_dims, z_dim, dropout=dropout)

        #DECODER
        dec_dims = z_dim + hidden_dim*heads//2 if fc else (z_dim + hidden_dim*heads)  #640  
        self.GNN_decoder = GAT_VAE(dec_dims, fc=fc, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=False, ew_dims=ew_dims) #If att_ew --> embedding_e_dec
        self.MLP_decoder = nn.Sequential(
            nn.Linear(dec_dims*heads, dec_dims),  #1280->640
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims, dec_dims//2),   #640->300
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims//2, output_dim)   #300->12
        )

        if fc:
            self.dropout_l = nn.Dropout(dropout)
            self.linear = nn.Linear(self.gat_2.out_feats, self.gat_2.out_feats//2)
            nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu') 
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.xavier_normal_(self.embedding_e.weight)      
        nn.init.xavier_normal_(self.embedding_gt.weight)      
        nn.init.kaiming_normal_(self.MLP_decoder[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.MLP_decoder[3].weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.MLP_decoder[6].weight)
    
    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def inference(self, g, feats, e_w, snorm_n,snorm_e, maps):
        """
        Samples from a normal distribution and decodes conditioned to the GNN outputs.   
        """
         # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)

        # Input embedding
        h = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e

        #Map Encoding
        if self.map_encoding:
            # Maps feature extraction
            maps_embedding = self.feature_extractor(maps)

            # Embeddings concatenation
            h = torch.cat([maps_embedding.squeeze(), h], dim=-1)
            h = self.linear_cat(h)

        # Input GNN
        h = self.GNN_inp(g, h, e_w, snorm_n)
        #Sample from gaussian distribution (BV, Z_dim)
        z_sample = torch.distributions.Normal(torch.zeros((h.shape[0],self.z_dim), dtype=h.dtype, device=h.device), torch.ones((h.shape[0],self.z_dim), dtype=h.dtype, device=h.device)).sample()
        
        #DECODE 
        h_dec = torch.cat([h, z_sample],dim=-1)
        h = self.GNN_decoder(g,h_dec,e_w,snorm_n)
        recon_y = self.MLP_decoder(h)
        return recon_y
    
    def forward(self, g, feats, e_w, snorm_n, snorm_e, gt, maps):
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        gt = gt.contiguous().view(gt.shape[0],-1)

        # Input embedding
        h = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e

        # Map encoding
        if self.map_encoding:
            # Maps feature extraction
            maps_embedding = self.feature_extractor(maps)

            # Embeddings concatenation
            h = torch.cat([maps_embedding.squeeze(), h], dim=-1)
            h = self.linear_cat(h)

        # Input GNN
        h = self.GNN_inp(g, h, e_w, snorm_n)
        
        #ENCODE
        # Encode ground truth trajectories to have the same shape as h
        h_gt = self.embedding_gt(gt)
        h_gt = self.GNN_enc_gt(g, h_gt, e_w, snorm_n)
        mu, log_var = self.encoder(h, h_gt)   # Latent distribution
        #Sample from the latent distribution
        z_sample = self.reparameterize(mu, log_var)
        
        #DECODE 
        h_dec = torch.cat([h, z_sample],dim=-1)
        h = self.GNN_decoder(g,h_dec,e_w,snorm_n)
        recon_y = self.MLP_decoder(h)
        return recon_y, mu, log_var

if __name__ == '__main__':
    history_frames = 4
    future_frames = 12
    hidden_dims = 768
    heads = 2

    input_dim = 9*history_frames
    output_dim = 2*future_frames 

    hidden_dims = round(hidden_dims / heads) 
    model = VAE_GNN(input_dim, hidden_dims, 16, output_dim, fc=False, dropout=0.2,feat_drop=0., attn_drop=0., heads=2,att_ew=True, ew_dims=2, map_encoding=True)

    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=True, history_frames=history_frames, future_frames=future_frames, map_encodding=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = batch
        e_w = batched_graph.edata['w']
        y, mu, log_var = model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_pos, maps)
        print(y.shape)

    