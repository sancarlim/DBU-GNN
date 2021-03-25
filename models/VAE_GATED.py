import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import matplotlib.pyplot as plt


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


class GatedGCN_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)  
        self.bn_node_e = nn.BatchNorm1d(output_dim) #nn.GroupNorm(32, output_dim) 

        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.A.weight,gain=gain)
        nn.init.xavier_normal_(self.B.weight, gain=gain)
        nn.init.xavier_normal_(self.C.weight, gain=gain) #sigmoid -> relu
        nn.init.xavier_normal_(self.D.weight, gain=gain)
        nn.init.xavier_normal_(self.E.weight, gain=gain)


    def message_func(self, edges):
        Bh_j = edges.src['Bh'] #n_e,256
        # e_ij = Ce_ij + Dhi + Ehj   N*B,256
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh'] #n_e,256
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        #torch.clamp(e.sigmoid_(), min=1e-4, max=1-1e-4) 
        sigma_ij = torch.clamp(torch.sigmoid(e), min=1e-4, max=1-1e-4) 
        # hi = Ahi + sum_j eta_ij * Bhj   
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / torch.sum(sigma_ij, dim=1)  #shape n_nodes*256
        
        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        with g.local_scope():
            h_in = h # residual connection
            e_in = e # residual connection
            
            
            g.ndata['h']  = h
            g.ndata['Ah'] = self.A(h) 
            g.ndata['Bh'] = self.B(h) 
            g.ndata['Dh'] = self.D(h)
            g.ndata['Eh'] = self.E(h) 
            g.edata['e']  = e 
            g.edata['Ce'] = self.C(e)
            
            g.update_all(self.message_func, self.reduce_func)
            
            h = g.ndata['h'] # result of graph convolution
            e = g.edata['e'] # result of graph convolution

            h = h * snorm_n # normalize activation w.r.t. graph node size
            e = e * snorm_e # normalize activation w.r.t. graph edge size
            
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
            
            h = torch.relu(h) # non-linear activation
            e = torch.relu(e) # non-linear activation
            
            h = h_in + h # residual connection
            e = e_in + e # residual connection


            return h, e

    
class GATED_VAE(nn.Module):
    def __init__(self, hidden_dim, fc=False, dropout=0.2):
        super().__init__()
        self.fc = fc
        
        self.GatedGCN1 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.GatedGCN2 = GatedGCN_layer(hidden_dim, hidden_dim)

        if fc:
            self.dropout_l = nn.Dropout(dropout)
            self.linear = nn.Linear(hidden_dim, hidden_dim//2)
            nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu') 
        
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        h, e = self.GatedGCN1(g, h, e, snorm_n, snorm_e)
        h, e = self.GatedGCN2(g, h, e, snorm_n, snorm_e)
        if self.fc:
            h = self.dropout_l(h)
            h=F.leaky_relu(self.linear(h))
        return h, e
    
class VAE_GATED(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, output_dim, fc=False, dropout=0.2,  ew_dims=1, map_encoding=False):
        super().__init__()
        self.fc = fc
        self.z_dim = z_dim
        self.map_encoding = map_encoding

        if self.map_encoding:
            model_ft = torchvision.models.resnet18(pretrained=True)
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
        self.GNN_inp = GATED_VAE(hidden_dim, fc=fc, dropout=dropout)

        #Encode ground_truth trajectories
        self.embedding_gt = nn.Linear(output_dim, hidden_dim)
        self.GNN_enc_gt = GATED_VAE(hidden_dim, fc=fc, dropout=dropout)
        
        #ENCODER
        input_enc_dims = hidden_dim if fc else hidden_dim*2 
        self.encoder = MLP_Enc(input_enc_dims, z_dim, dropout=dropout)

        #DECODER
        dec_dims = z_dim + hidden_dim//2 if fc else (z_dim + hidden_dim)  #768+100  512+100  1024+100
        self.embedding_e_dec = nn.Linear(hidden_dim, dec_dims)
        self.GNN_decoder = GATED_VAE(dec_dims, fc=fc, dropout=dropout) 
        self.MLP_decoder = nn.Sequential(
            nn.Linear(dec_dims, dec_dims//2),  #868->434  ,  612->306  , 1124-->562
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims//2, dec_dims//4),   
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_dims//4, output_dim)  
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
        nn.init.xavier_normal_(self.embedding_e_dec.weight)    
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

        if self.map_encoding:
            # Maps feature extraction
            maps_embedding = self.feature_extractor(maps)

            # Embeddings concatenation
            h = torch.cat([maps_embedding, h], dim=-1)
            h = self.linear_cat(h)

        # Input GNN
        h, e_inp = self.GNN_inp(g, h, e, snorm_n, snorm_e)
        #Sample from gaussian distribution (BV, Z_dim)
        z_sample = torch.distributions.Normal(torch.zeros((h.shape[0],self.z_dim), dtype=h.dtype, device=h.device), torch.ones((h.shape[0],self.z_dim), dtype=h.dtype, device=h.device)).sample()
        
        #DECODE 
        h_dec = torch.cat([h, z_sample],dim=-1)
        #Embedding for having dimmensions of edge feats = dimmensions of node feats
        e_dec = self.embedding_e_dec(e_inp)
        h, _ = self.GNN_decoder(g,h_dec,e_dec,snorm_n, snorm_e)
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

        if self.map_encoding:
            # Maps feature extraction
            maps_embedding = self.feature_extractor(maps)

            # Embeddings concatenation
            h = torch.cat([maps_embedding, h], dim=-1)
            h = self.linear_cat(h)
            
        # Input GNN
        h, e_inp = self.GNN_inp(g, h, e, snorm_n, snorm_e)
        
        #ENCODE
        # Encode ground truth trajectories to have the same shape as h
        h_gt = self.embedding_gt(gt)
        h_gt, _ = self.GNN_enc_gt(g, h_gt, e, snorm_n, snorm_e)
        mu, log_var = self.encoder(h, h_gt)   # Latent distribution
        #Sample from the latent distribution
        z_sample = self.reparameterize(mu, log_var)
        
        #DECODE 
        h_dec = torch.cat([h, z_sample],dim=-1)
        #Embedding for having dimmensions of edge feats = dimmensions of node feats
        e_dec = self.embedding_e_dec(e_inp)
        h, _ = self.GNN_decoder(g,h_dec,e_dec,snorm_n, snorm_e)
        recon_y = self.MLP_decoder(h)
        return recon_y, mu, log_var

if __name__ == '__main__':
    model = VAE_GATED(48, 512, 128, 24, fc=False, dropout=0.2,feat_drop=0., attn_drop=0., heads=2,att_ew=True)
    print(model)

    