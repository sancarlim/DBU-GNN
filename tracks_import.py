import pandas
import glob
import numpy as np
from loguru import logger
from scipy import interpolate as interp
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import BSpline, make_interp_spline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import griddata


def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i-1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def read_all_recordings_from_csv(base_path="../data/"):
    """
    This methods reads the tracks and meta information for all recordings given the path of the inD dataset.
    :param base_path: Directory containing all csv files of the inD dataset
    :return: a tuple of tracks, static track info and recording meta info
    """
    tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
    static_tracks_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
    recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

    all_tracks = []
    all_static_info = []
    all_meta_info = []
    for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                                   static_tracks_files,
                                                                   recording_meta_files):
        logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
        tracks, static_info, meta_info = read_from_csv(track_file, static_tracks_file, recording_meta_file)
        all_tracks.extend(tracks)
        all_static_info.extend(static_info)
        all_meta_info.extend(meta_info)

    return all_tracks, all_static_info, all_meta_info


def read_from_csv(track_file, static_tracks_file, recordings_meta_file, pred_file):
    """
    This method reads tracks including meta data for a single recording from csv files.
    :param track_file: The input path for the tracks csv file.
    :param static_tracks_file: The input path for the static tracks csv file.
    :param recordings_meta_file: The input path for the recording meta csv file.
    :return: tracks, static track info and recording info
    """
    static_info = read_static_info(static_tracks_file)
    meta_info = read_meta_info(recordings_meta_file)
    tracks = read_tracks(track_file, meta_info, static_tracks_file)
    preds = read_preds(pred_file, meta_info)
    return tracks, static_info, meta_info, preds

def read_preds(pred_file, meta_info):
    df = pandas.read_csv(pred_file)
    df = df[df.recording_id != 0]
    raw_preds = df.groupby(["obj_id"], sort=False)
    ortho_px_to_meter = meta_info["orthoPxToMeter"]
    preds=[]
    for pred_id, pred_rows in raw_preds:
        pred = pred_rows.to_dict(orient="list")

        for key, value in pred.items():
            if key in ["obj_id", "recording_id"]:
                pred[key] = value[0]
            else:
                pred[key] = np.array(value)

        pred["pred_center"] = np.stack([pred["pred_x"], pred["pred_y"]], axis=-1)
        # Create special version of some values needed for visualization
        pred["xCenterVis"] = pred["pred_x"] / ortho_px_to_meter
        pred["yCenterVis"] = -pred["pred_y"] / ortho_px_to_meter
        pred["centerVis"] = np.stack([pred["pred_x"], -pred["pred_y"]], axis=-1) / ortho_px_to_meter

        preds.append(pred)
    
    #Polynomial fit
    
    for ind, pred_id in enumerate(preds):
        pred_center_points = pred_id['centerVis']  # a (nbre_points x nbre_dim) array
        if pred_center_points.max() != 0 and len(pred_center_points)>3:
            #coefs = np.polyfit(pred_center_points[:,0],pred_center_points[:,1],3)
            #pred_center_points =np.array([list(pred_center_points[:,0]), list(np.polyval(coefs,pred_center_points[:,0]))]).transpose() 
            
            t = np.arange(len(pred_center_points))
            ti = np.linspace(0, t.max(), 10 * t.size)
            xi = interp1d(t, pred_center_points[:,0], kind='cubic')(ti)   #spline, quadratic, cubic interpolation of 1st, 2nd, 3rd order 
            yi = interp1d(t, pred_center_points[:,1], kind='cubic')(ti)
            pred_center_points =np.array([list(xi), list(yi)]).transpose()
            #pred_center_points=interpolate_polyline(pred_center_points, len(pred_center_points))
        preds[ind]['centerVis'] = pred_center_points
    
    return preds


def read_tracks(track_file, meta_info, track_meta):
    # Read the csv file to a pandas dataframe
    df = pandas.read_csv(track_file)
    df_meta = pandas.read_csv(track_meta)

    #filter some of the parked vehicles in 3rd intersection
    if meta_info['recordingId'] > 17 and meta_info['recordingId'] < 30: 
        max_num_frames = df_meta['numFrames'].max()
        id_parked_objects = list(df_meta[df_meta['numFrames']==max_num_frames].trackId)
        del id_parked_objects[-2:]  #keep 2 parked cars
        df = df[~df['trackId'].isin(id_parked_objects)]

    # To extract every track, group the rows by the track id
    raw_tracks = df.groupby(["trackId"], sort=False)
    ortho_px_to_meter = meta_info["orthoPxToMeter"]
    tracks = []
    for track_id, track_rows in raw_tracks:
        track = track_rows.to_dict(orient="list")

        # Convert scalars to single value and lists to numpy arrays
        for key, value in track.items():
            if key in ["trackId", "recordingId"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)

        track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
        track["bbox"] = calculate_rotated_bboxes(track["xCenter"], track["yCenter"],
                                                 track["length"], track["width"],
                                                 np.deg2rad(track["heading"]))

        # Create special version of some values needed for visualization
        track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
        track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
        track["centerVis"] = np.stack([track["xCenter"], -track["yCenter"]], axis=-1) / ortho_px_to_meter
        track["widthVis"] = track["width"] / ortho_px_to_meter
        track["lengthVis"] = track["length"] / ortho_px_to_meter
        track["headingVis"] = track["heading"] * -1
        track["headingVis"][track["headingVis"] < 0] += 360
        track["bboxVis"] = calculate_rotated_bboxes(track["xCenterVis"], track["yCenterVis"],
                                                    track["lengthVis"], track["widthVis"],
                                                    np.deg2rad(track["headingVis"]))

        tracks.append(track)
    return tracks


def read_static_info(static_tracks_file):
    """
    This method reads the static info file from highD data.
    :param static_tracks_file: the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    return pandas.read_csv(static_tracks_file).to_dict(orient="records")


def read_meta_info(recordings_meta_file):
    """
    This method reads the recording info file from ind data.
    :param recordings_meta_file: the path for the recording meta csv file.
    :return: the meta dictionary
    """
    return pandas.read_csv(recordings_meta_file).to_dict(orient="records")[0]


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