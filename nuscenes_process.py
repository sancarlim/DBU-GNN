import sys
import os
import numpy as np
from scipy import spatial 
import pickle
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import create_splits_scenes
import pandas as pd
from collections import defaultdict
from pyquaternion import Quaternion

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


#508 0 sequences???
scene_blacklist = [499, 515, 517]

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction

# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   #850 scenes
# Helper for querying past and future data for an agent.
helper = PredictHelper(nuscenes)

#DEFINE ATTENTION RADIUS FOR CONNECTING NODES
VEH_VEH_RADIUS= 35
VEH_PED_RADIUS= 20
VEH_BIC_RADIUS= 20
PED_PED_RADIUS= 10 
PED_BIC_RADIUS= 15
BIC_BIC_RADIUS= 25
neighbor_distance = VEH_VEH_RADIUS

'''
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


# To get data for an agent at a particular timestamp
instance_token, sample_token = train[0].split("_")
'''

def pol2cart(th, r):
    """
    Transform polar to cartesian coordinates.
    :param th: Nx1 ndarray
    :param r: Nx1 ndarray
    :return: Nx2 ndarray
    """

    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))

    cart = np.array([x, y]).transpose()
    return cart


def cart2pol(cart):
    """
    Transform cartesian to polar coordinates.
    :param cart: Nx2 ndarray
    :return: 2 Nx1 ndarrays
    """
    if cart.shape == (2,):
        cart = np.array([cart])

    x = cart[:, 0]
    y = cart[:, 1]

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return th, r

def calculate_rotated_bboxes(center_points_x, center_points_y, length, width, rotation=0):
    """
    Calculate bounding box vertices from centroid, width and length.
    :param centroid: center point of bbox
    :param length: length of bbox
    :param width: width of bbox
    :param rotation: rotation of main bbox axis (along length)
    :return:
    """

    centroid = np.array([center_points_x, center_points_y]).transpose()

    centroid = np.array(centroid)
    if centroid.shape == (2,):
        centroid = np.array([centroid])

    # Preallocate
    data_length = centroid.shape[0]
    rotated_bbox_vertices = np.empty((data_length, 4, 2))

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2

    for i in range(4):
        th, r = cart2pol(rotated_bbox_vertices[:, i, :])
        rotated_bbox_vertices[:, i, :] = pol2cart(th + rotation, r).squeeze()
        rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid

    return rotated_bbox_vertices

def process_tracks(track, current_frame):
    '''
        Tracks: a list of (n_frames ~40f = 20s) tracks_per_frame ordered by frame.
                Each row (track) contains a dict, where each key corresponds to an array of data from all agents in that frame.
        
        Returns data processed for a sequence of 8s (2s of history, 6s of labels)
    '''
    sample_token = track['sample_token'][0]
    visible_node_id_list = track["node_id"]  #All agents in the current frame      
    num_visible_object = len(visible_node_id_list)

    #Zero-centralization per frame (sequence)
    mean_xy = [track['x_global'].mean(),track['y_global'].mean()]
    track['position'][:,:2] = track['position'][:,:2] - mean_xy

    # You can convert global coords to local frame with: helper.convert_global_coords_to_local(coords,starting_annotation['translation'], starting_annotation['rotation'])
    # x_global y_global are centralized in 0 taking into account all objects positions in the current frame
    xy = track['position'][:, :2].astype(float)
    # Compute distance between any pair of objects
    dist_xy = spatial.distance.cdist(xy, xy)  
    # If their distance is less than ATTENTION RADIUS (neighbor_distance), we regard them as neighbors.
    neighbor_matrix = np.zeros((num_visible_object, num_visible_object))
    neighbor_matrix = (dist_xy<neighbor_distance).astype(int)

    #Retrieve all past and future trajectories
    '''
    future_xy_local = helper.get_future_for_sample(sample_token, seconds=future, in_agent_frame=True)
    past_xy_local = helper.get_past_for_sample(sample_token, seconds=history, in_agent_frame=True)
    future_xy_local_list=[value for key,value in future_xy_local.items()]
    past_xy_local_list=[value for key,value in past_xy_local.items()]
    '''
    # Retrieve features and labels for each agent (N, 23)

    # Get past and future trajectories
    future_xy_local = np.zeros((num_visible_object, future_frames*2))
    past_xy_local = np.zeros((num_visible_object, 2*history_frames))
    mask = np.zeros((num_visible_object, future_frames))
    for i, node_id in enumerate(track['node_id']):
        future_xy_i=helper.get_future_for_agent(node_id,sample_token, seconds=future, in_agent_frame=True).reshape(-1)
        past_xy_i=helper.get_past_for_agent(node_id,sample_token, seconds=history, in_agent_frame=True).reshape(-1)
        past_xy_local[i,:len(past_xy_i)] = past_xy_i
        future_xy_local[i, :len(future_xy_i)] = future_xy_i # Some agents don't have 6s of future or 2s of history, pad with 0's
        mask[i,:len(future_xy_i)//2] += np.ones((len(future_xy_i)//2))
        
    object_features = np.column_stack((
            track['position'], track['motion'], past_xy_local, future_xy_local, mask, track['info_agent'],
            track['info_sequence'] ))  # 3 + 3 + 8 + 24 + 12 + 4 + 2 + 1 + 1 = 58   

    inst_sample_tokens = np.column_stack((track['node_id'], track['sample_token']))
    '''
    object_feature_list = []
    for frame_ind in range(start_frame, end_frame):	
        for node_id in visible_node_id_list:
            node_idx = np.where(tracks[frame_ind]['node_id']==node_id)[0]
            if node_idx.size:
                now_frame_feature_dict = {node_id : (
                    list(tracks[frame_ind]['position'][node_idx][0])+ 
                    list(tracks[frame_ind]['velocity'][node_idx][0]) + 
                    list(tracks[frame_ind]['info_sequence'])+
                    [node_id]+
                    list(tracks[frame_ind]['info_agent'][node_idx[0]]) 
                    )}
        # if the current object is not at this frame, we return all 0s 
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_node_id_list])
        object_feature_list.append(now_frame_feature)

    object_feature_list = np.array(object_feature_list)  # T,V,C
    object_frame_feature = np.zeros((num_visible_object, end_ind-start_ind, total_feature_dimension))
    object_frame_feature = np.transpose(object_feature_list, (1,0,2))
    '''
    return object_features, neighbor_matrix, mean_xy, inst_sample_tokens


def process_scene(scene):
    '''
    Returns a list of (n_frames ~40f = 20s) tracks_per_frame ordered by frame.
    Each row contains a dict, where each key corresponds to an array of data from all agents in that frame.
    '''
    scene_id = int(scene['name'].replace('scene-', ''))   #419 la que data empieza en frame 4 data.frame_id.unique() token '8c84164e752a4ab69d039a07c898f7af'
    data = pd.DataFrame(columns=['scene_id',
                                 'sample_token',
                                 'frame_id',
                                 'type',
                                 'node_id',
                                 'x_global',
                                 'y_global', 
                                 'heading',
                                 'velocity',
                                 'acceleration',
                                 'heading_change_rate',
                                 'length',
                                 'width',
                                 'height'])
    sample_token = scene['first_sample_token']
    sample = nuscenes.get('sample', sample_token)
    frame_id = 0
    mean_xy = []
    while sample['next']:
        annotations = helper.get_annotations_for_sample(sample_token)
        for i,annotation in enumerate(annotations):
            #print(f'{i} out of {len(annotations)} annotations')
            instance_token = annotation['instance_token']
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nuscenes.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                continue

            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                node_type = 2
            elif 'bicycle' in category or 'motorcycle' in category:
                node_type = 3
            elif 'vehicle' in category: # and 'parked' not in attribute: #filter parked vehicles                
                node_type = 1
            else:
                continue

            #if first sample returns nan
            heading_change_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            velocity =  helper.get_velocity_for_agent(instance_token, sample_token)
            acceleration = helper.get_acceleration_for_agent(instance_token, sample_token)
            

            data_point = pd.Series({'scene_id': scene_id,
                                    'sample_token': sample_token,
                                    'frame_id': frame_id,
                                    'type': node_type,
                                    'node_id': instance_token,
                                    'x_global': annotation['translation'][0],
                                    'y_global': annotation['translation'][1],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                    'velocity': velocity,
                                    'acceleration': acceleration,
                                    'heading_change_rate': heading_change_rate,
                                    'length': annotation['size'][0],
                                    'width': annotation['size'][1],
                                    'height': annotation['size'][2]}).fillna(0)    #inplace=True         

            data = data.append(data_point, ignore_index=True)

        sample = nuscenes.get('sample', sample['next'])
        sample_token = sample['token']
        frame_id += 1
        '''
        #Zero-centralization per frame (sequence)
        mean_xy.append([data['x_global'].mean(),data['y_global'].mean()])
        data[-1]['x_global'] = data['x_global'] - mean_xy[-1][0]
        data['y_global'] = data['y_global'] - mean_xy[-1][1]
        '''

    #data.sort_values('frame_id', inplace=True)
    tracks_per_frame=data.groupby(['frame_id'], sort=True)
    '''
    Tracks is a list of n_frames rows ordered by frame.
    Each row contains a dict, where each key corresponds to an array of data from all agents in that frame.
    '''
    tracks = []
    for frame, track_rows in tracks_per_frame:
        #track_rows contains info of all agents in frame
        track = track_rows.to_dict(orient="list")
        
        for key, value in track.items():
            if key not in ["frame_id", "scene_id", "node_id", "sample_token"]:
                track[key] = np.array(value)
            
        track['info_sequence'] = np.stack([track["frame_id"],track["scene_id"]], axis=-1)
        track['info_agent'] = np.stack([track["type"],track["length"],track["width"],track["height"]], axis=-1)
        track["position"] = np.stack([track["x_global"], track["y_global"], track["heading"]], axis=-1)
        track['motion'] = np.stack([track["velocity"], track["acceleration"], track["heading_change_rate"]], axis=-1)
        track["bbox"] = calculate_rotated_bboxes(track["x_global"], track["y_global"],
                                                track["length"], track["width"],
                                                np.deg2rad(track["heading"]))
    
        tracks.append(track)


    frame_id_list = list(range(len(tracks)))   #list(range(data.frame_id.unique()[0], range(data.frame_id.unique()[-1])))
    #assert len(frame_id_list) == len(tracks), f"{len(frame_id_list)} != {len(tracks)}"
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    tokens_list = []
    visible_object_indexes_list=[]
    step=2 #iterate over 1s
    for start_ind in frame_id_list[:-total_frames+1:step]:
        current_frame = start_ind + history_frames
        object_frame_feature, neighbor_matrix, mean_xy, inst_sample_tokens = process_tracks(tracks[current_frame], current_frame)  
        #print(f"Processed sequence with current frame {current_frame}")
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)	
        all_mean_list.append(mean_xy)
        tokens_list.append(inst_sample_tokens)

    all_adjacency = np.array(all_adjacency_list, dtype=object)
    all_mean = np.array(all_mean_list)                            
    all_feature = np.array(all_feature_list, dtype=object)
    tokens = np.array(tokens_list, dtype=object)
    return all_feature, all_adjacency, all_mean, tokens

    '''
    # Generate Maps
    map_name = nuscenes.get('log', scene['log_token'])['location']
    nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)

    type_map = dict()
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
    patch_angle = 0  # Default orientation where North is up
    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                   'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
    map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
        np.uint8)
    map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    '''


# Data splits for the CHALLENGE - returns instance and sample token  

# Train: 5883 seq (475 scenes) Train_val: 2219 seq (185 scenes)  Val: 1682 seq (138 scenes) 
ns_scene_names = dict()
ns_scene_names['train'] = get_prediction_challenge_split("train", dataroot=DATAROOT) 
ns_scene_names['val'] =  get_prediction_challenge_split("train_val", dataroot=DATAROOT)
ns_scene_names['test'] = get_prediction_challenge_split("val", dataroot=DATAROOT)



#scenes_df=[]
for data_class in ['train', 'val', 'test']:
    scenes=[]
    for ann in ns_scene_names[data_class]:
        _, sample_token=ann.split("_")
        sample = nuscenes.get('sample', sample_token)
        scenes.append(nuscenes.get('scene', sample['scene_token'])['token'])
    scenes_token_set = list(set(scenes))

    all_data=[]
    all_adjacency=[]
    all_mean_xy=[]
    all_tokens=[]
    for scene_token in scenes_token_set:
        all_feature_sc, all_adjacency_sc, all_mean_sc, tokens_sc = process_scene(nuscenes.get('scene', scene_token))
        print(f"Scene {nuscenes.get('scene', scene_token)['name']} processed! {all_adjacency_sc.shape[0]} sequences of 8 seconds.")
        all_data.extend(all_feature_sc)
        all_adjacency.extend(all_adjacency_sc)
        all_mean_xy.extend(all_mean_sc)
        all_tokens.extend(tokens_sc)
        #scenes_df.append(scene_df)
        #scene_df.to_csv(os.path.join('./nuscenes_processed/', nuscenes.get('scene', scene_token)['name'] + '.csv'))
    all_data = np.array(all_data, dtype=object)  
    all_adjacency = np.array(all_adjacency, dtype=object) 
    all_mean_xy = np.array(all_mean_xy) 
    all_tokens = np.array(all_tokens, dtype=object)
    save_path = '/media/14TBDISK/sandra/nuscenes_processed/nuscenes_challenge_' + data_class + '.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy], writer)
    print(f'Processed {all_data.shape[0]} sequences and {len(ns_scene_names[data_class])} scenes.')


'''
#Usual split: Train 8536 (700 scenes)  Val: 1828 (150 scenes)
splits = create_splits_scenes()
ns_scene_names = dict()
ns_scene_names['train'] =  splits['train']  
ns_scene_names['val'] =  splits['val']
ns_scene_names['test'] = splits['test']

for data_class in ['train', 'val', 'test']:
    all_data=[]
    all_adjacency=[]
    all_mean_xy=[]
    all_tokens=[]
    for ns_scene_name in ns_scene_names[data_class]:
        scene_token = nuscenes.field2token('scene', 'name', ns_scene_name)
        ns_scene = nuscenes.get('scene', scene_token[0])
        scene_id = int(ns_scene['name'].replace('scene-', ''))
        if scene_id in scene_blacklist:  # Some scenes have bad localization
            continue
        all_feature_sc, all_adjacency_sc, all_mean_sc, tokens_sc = process_scene(ns_scene)
        print(f"Scene {ns_scene_name} processed! {all_adjacency_sc.shape[0]} sequences of 8 seconds.")
        all_data.extend(all_feature_sc)
        all_adjacency.extend(all_adjacency_sc)
        all_mean_xy.extend(all_mean_sc)
        all_tokens.extend(tokens_sc)
'''
    

#To return the past/future data for the entire sample (local/global - in_agent_frame=T/F)
sample_ann = helper.get_annotations_for_sample(sample_token)
future_xy_global = helper.get_future_for_sample(sample_token, seconds=3, in_agent_frame=False)
past_xy_global = helper.get_past_for_sample(sample_token, seconds=3, in_agent_frame=False)

# The goal of the nuScenes prediction task is to predict the future trajectories of objects in the nuScenes dataset. 
# A trajectory is a sequence of x-y locations. For this challenge, the predictions are 6-seconds long and sampled at 2 hertz (n_timesteps is 12) and 2s of history.
# The leaderboard will be ranked according to performance on the nuScenes val set. To prevent overfitting on the val set, the top 5 submissions
# on the leaderboard will be asked to send us their code and we will run their model on the test set.
# We release annotations for the train and val set, but not for the test set. We have created a hold out set for validation from 
# the training set called the train_val set.

# 1. Iterate over all scenes (train and val)
# 2. Iterate over each sample
# 3. Iterate over N instances (objects) in each sample
#       3.1. Filter category and save: category, instance_token (node_id), frame
#       3.2. Get past and future trajectories (local frame) and CURRENT: x,y (global frame), heading, velocity, acceleration and heading_change_rate
#
# df_scenes: list of 475 dataframes (475 scenes) (train) 
#     Each dataframe: 40 samples (keyframes)
#          Each sample: N instances ~43