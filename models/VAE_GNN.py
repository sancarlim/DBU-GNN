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
from models.MapEncoder import MapEncoder
from models.My_GAT import My_GATLayer, MultiHeadGATLayer


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

    def forward(self, g, h):
        h = self.dropout_l(h)
        h = F.leaky_relu(self.linear(h))
        log_var = self.log_var(h) 
        mu = self.mu(h)
        return mu, log_var


    
class GAT_VAE(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, feat_drop=0., attn_drop=0., heads=1,att_ew=False, ew_dims=1):
        super().__init__()
        self.heads = heads
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop,att_ew, res_weight=res_weight, res_connection=res_connection ) 
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0.,att_ew, res_weight=res_weight, res_connection=res_connection ) 
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, res_weight=True, res_connection=True , num_heads=heads,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)            
            self.embedding_e = nn.Linear(ew_dims, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads,hidden_dim*heads, res_weight=True , res_connection=True ,num_heads=1, feat_drop=0., attn_drop=0., att_ew=att_ew)    

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
        return h
    
class VAE_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, output_dim, fc=False, dropout=0.2, feat_drop=0., 
                    attn_drop=0., heads=1,att_ew=False, ew_dims=1, backbone='map_encoder', pretrained=True):
        super().__init__()
        self.heads = heads
        self.fc = fc
        self.z_dim = z_dim

        ###############
        # Map Encoder #
        ###############
        if backbone == 'map_encoder':
            self.map_feature_extractor = MapEncoder(input_channels = 3, input_size=112, 
                                                    hidden_channels = [10,20,10,1], output_size = hidden_dim, 
                                                    kernels = [5,5,5,3], strides = [2,2,1,1])

            enc_dims = hidden_dim*2+output_dim    
            dec_dims = z_dim + hidden_dim*2
        elif backbone == 'resnet18':   
            # Use Resnet18 as backbone
            model_ft = resnet18(pretrained=pretrained)
            self.map_feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
            if pretrained:
                ct=0
                for child in self.map_feature_extractor.children():
                    ct+=1
                    if ct < 5: #7
                        for param in child.parameters():
                            param.requires_grad = False
            
            enc_dims = hidden_dim+512+output_dim   
            dec_dims = z_dim + hidden_dim + 512
        

        ############################
        # Input Features Embedding #
        ############################

        ################# NO MAPS
        #enc_dims = hidden_dim + output_dim
        #dec_dims = hidden_dim + z_dim
        ###############
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(ew_dims, enc_dims) 
        

        #############
        #  ENCODER  #
        #############
        self.GNN_enc = GAT_VAE(enc_dims, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = enc_dims*heads
        self.encoder = MLP_Enc(encoder_dims+output_dim, z_dim, dropout=dropout)


        #############
        #  DECODER  #
        #############
        self.embedding_e_dec = nn.Linear(ew_dims, dec_dims)   
        self.GNN_decoder = GAT_VAE(dec_dims, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=False, ew_dims=ew_dims) #If att_ew --> embedding_e_dec
        self.MLP_decoder = nn.Sequential(
            nn.Linear(dec_dims*heads+z_dim, dec_dims), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims, dec_dims//2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims//2, output_dim) 
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        #nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.xavier_normal_(self.embedding_e.weight)       
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
        h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e = self.embedding_e_dec(e_w)
        g.edata['w']=e

        #Map Encoding
        # Maps feature extraction
        maps_emb = self.map_feature_extractor(maps)

        #Sample from gaussian distribution (BV, Z_dim)
        z_sample = torch.distributions.Normal(torch.zeros((feats.shape[0],self.z_dim), dtype=feats.dtype, device=feats.device), 
                                              torch.ones((feats.shape[0],self.z_dim), dtype=feats.dtype, device=feats.device)).sample()

        #DECODE 
        h_dec = torch.cat([maps_emb.squeeze(dim=-1).squeeze(dim=-1), h_emb, z_sample],dim=-1)
        h = self.GNN_decoder(g,h_dec,e_w,snorm_n)
        h = torch.cat([h, z_sample],dim=-1)
        recon_y = self.MLP_decoder(h)
        return recon_y
    
    def forward(self, g, feats, e_w, snorm_n, snorm_e, gt, maps):
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        gt = gt.contiguous().view(gt.shape[0],-1)

        # Input embedding
        h_emb = self.embedding_h(feats) 
        e = self.embedding_e(e_w)
        g.edata['w']=e

        # Map encoding
        
        # Maps feature extraction
        maps_emb = self.map_feature_extractor(maps)

        # Embeddings concatenation
        h = torch.cat([maps_emb.squeeze(), h_emb, gt], dim=-1)
        #h = self.linear_cat(h)


        # ENCODE
        h = self.GNN_enc(g, h, e_w, snorm_n)
        h = torch.cat([h, gt], dim=-1)            
        mu, log_var = self.encoder(g,h)   # Latent distribution

        # Sample from the latent distribution
        z_sample = self.reparameterize(mu, log_var)
        
        # DECODE 
        h_dec = torch.cat([maps_emb.squeeze(), h_emb, z_sample],dim=-1)
        h = self.GNN_decoder(g,h_dec,e_w,snorm_n)
        h = torch.cat([h, z_sample],dim=-1)
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
    model = VAE_GNN(input_dim, hidden_dims, 16, output_dim, fc=False, dropout=0.2,feat_drop=0., attn_drop=0., heads=2,att_ew=True, ew_dims=2, backbone='map_encoder')

    test_dataset = nuscenes_Dataset(train_val_test='val', rel_types=True, history_frames=history_frames, future_frames=future_frames) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = batch
        e_w = batched_graph.edata['w']
        y, mu, log_var = model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_pos, maps)
        print(y.shape)

    