
import dgl
import torch
from torch.utils.data import DataLoader
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from nuscenes_Dataset import nuscenes_Dataset
from models.VAE_GNN import VAE_GNN
#from VAE_GATED import VAE_GATED
import wandb
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from argparse import ArgumentParser, Namespace
from utils import str2bool, compute_change_pos
from nuscenes.eval.prediction.data_classes import Prediction
import json

from torchvision import transforms, utils

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction
input_dim_model = history_frames*7 #Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
output_dim = future_frames*2
base_path='/home/sandra/PROGRAMAS/DBU_Graph/NuScenes'

scene_blacklist = [499, 515, 517]

patch_margin = 2
min_diff_patch = 30


def collate_batch(samples):
    graphs, masks, feats, gt, tokens, mean_xy = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens[0], mean_xy



class LitGNN(pl.LightningModule):
    def __init__(self, model,  train_dataset, val_dataset, test_dataset, history_frames: int=3, future_frames: int=3, lr: float = 1e-3, 
                    batch_size: int = 64, wd: float = 1e-1, beta: float = 0., delta: float = 1., rel_types: bool = False, 
                    scale_factor=1):
        super().__init__()
        self.model= model
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   
        return pred
    
    def configure_optimizers(self):
        pass
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass
         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, tokens_eval, mean_xy = test_batch
        rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
        last_loc = feats[:,-1:,:2].detach().clone() 
        last_loc = last_loc*rescale_xy       
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        # Prediction: Prediction of model [num_modes, n_timesteps, state_dim] = [25, 12, 2]
        prediction_all_agents = []  # [num_agents, num_modes, n_timesteps, state_dim]
        for i in range(25):
            #Model predicts relative_positions
            preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e)  # [N_agents, 12, 2]
            preds=preds.view(preds.shape[0],self.future_frames,-1)  
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc

            # Provide predictions in global-coordinates
            pred_x = preds[:,:,0].cpu().numpy() + mean_xy[0][0]  # [N_agents, T]
            pred_y = preds[:,:,1].cpu().numpy() + mean_xy[0][1]
            
            prediction_all_agents.append(np.stack([pred_x, pred_y],axis=-1))

        prediction_all_agents = np.array(prediction_all_agents)
        
        
        #VISUALIZE SEQUENCE
        #Get Scene from sample token ie current frame
        sample_token = tokens_eval[0][1]
        scene=nuscenes.get('scene', nuscenes.get('sample',sample_token)['scene_token'])
        scene_name = scene['name']
        scene_id = int(scene_name.replace('scene-', ''))
        if scene_id in scene_blacklist:
            print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)
              
        log=nuscenes.get('log', scene['log_token'])
        location = log['location']
        nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=location)

        #Render map with ego poses
        sample_tokens = nuscenes.field2token('sample', 'scene_token', scene['token'])
        ego_poses=[]
        for sample_token in sample_tokens:
            sample_record = nuscenes.get('sample', sample_token)
            
            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])

            # Calculate the pose on the map and append.
            ego_poses.append(pose_record['translation'])
        # Check that ego poses aren't empty.
        assert len(ego_poses) > 0, 'Error: Found 0 ego poses. Please check the inputs.'
        ego_poses = np.vstack(ego_poses)[:, :2]
        
        # Render the map patch with the current ego poses.
        min_patch = np.floor(ego_poses.min(axis=0) - patch_margin)
        max_patch = np.ceil(ego_poses.max(axis=0) + patch_margin)
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
        
        fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10), alpha=0.3,
                                    render_egoposes_range=False,
                                    render_legend=True, bitmap=None)

        #Print agents trajectories

        for token in tokens_eval:
            idx = np.where(np.array(tokens_eval)== token[0])[0][0]
            instance, sample_token = token
            prediction = prediction_all_agents[:,idx]
            history = feats[idx]
            future = labels_pos[idx]

            # Plot predictions
            for t in range(prediction.shape[1]):
                sns.kdeplot(x=prediction[:,t,0], y=prediction[:,t,1],
                    ax=ax, shade=True, thresh=0.05, 
                    color='b', zorder=600, alpha=0.8)

            #Plot history
            ax.plot(history[:, 0], history[:, 1], 'k--')

            #Plot ground truth
            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
            
            # Current Node Position
            node_circle_size=0.3
            circle_edge_width=0.5
            circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)
        
        ax.axis('off')
        fig.savefig(os.path.join(base_path, 'visualizations' , scene_id + '.pdf'), dpi=300, bbox_inches='tight')


   
def main(args: Namespace):
    print(args)

    seed=seed_everything(0)

    test_dataset = nuscenes_Dataset(raw_dir=os.path.join(base_path, 'nuscenes_mini_val.pkl'), train_val_test='test', 
                    rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, challenge_eval=True)  #25 seq 2 scenes 103, 916

    if args.model_type == 'vae_gated':
        model = VAE_GATED(input_dim_model, args.hidden_dims, z_dim=args.z_dims, output_dim=output_dim, fc=False, dropout=args.dropout,  ew_dims=args.ew_dims)
    else:
        model = VAE_GNN(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, feat_drop=args.feat_drop, attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, ew_dims=args.ew_dims)

    LitGNN_sys = LitGNN(model=model, history_frames=history_frames, future_frames= future_frames, train_dataset=None, val_dataset=None,
                 test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)
      
    trainer = pl.Trainer(gpus=args.gpus, deterministic=True, precision=16, profiler=True) 
 
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=LitGNN_sys.model, history_frames=history_frames, future_frames= future_frames,
                    train_dataset=None, val_dataset=None, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)

    
    trainer.test(LitGNN_sys)
   

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=10, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--z_dims", type=int, default=64, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=768)
    parser.add_argument("--model_type", type=str, default='vae_gat', help="Choose aggregation function between GAT or GATED",
                                        choices=['vae_gat', 'vae_gated'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.25)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=False, help="Add edge features in attention function (GAT)")
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path.')   
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    hparams = parser.parse_args()

    main(hparams)




