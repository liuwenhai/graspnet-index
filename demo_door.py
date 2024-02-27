""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNetIndex,GraspNet, pred_decode,pred_decodeIndex
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.001, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNetIndex(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_pointcloud(data_dir):
    data = np.load(data_dir)    # ['cam2_wolrd', 'per_coord_world', 'instance_mask', 'handle_mask']
    cam2_wolrd = data['cam2_wolrd']
    per_coord_world = data['per_coord_world']
    instance_mask = data['instance_mask']
    handle_mask = data['handle_mask']

    # color_sampled = np.tile(np.array([[0., 0., 1.]]), (per_coord_world.shape[0], 1)).astype(np.float32)
    # red = np.tile(np.array([[1., 0., 0.]]), (per_coord_world.shape[0], 1)).astype(np.float32)
    # color_sampled[np.where(instance_mask==handle_mask[2])] = red[np.where(instance_mask==handle_mask[2])]
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(per_coord_world.astype(np.float32))
    # # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_sampled)
    # o3d.visualization.draw_geometries([cloud])


    r = cam2_wolrd[:3,:3].T
    t = - r @ cam2_wolrd[:3,3]
    cloud_masked = per_coord_world @ r.T + t.T

    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = cloud_masked[idxs]

    seg_mask = instance_mask[idxs]
    # index_mask = (seg_mask == handle_mask[0] | seg_mask == handle_mask[1] | seg_mask == handle_mask[2]).astype(np.int32)
    index_mask = np.zeros(seg_mask.shape).astype(np.int32)
    for handle_index in handle_mask:
        index_mask = np.logical_or(seg_mask == handle_index, index_mask)
    # index_mask = (seg_mask == handle_mask[0]).astype(np.int32)
    # convert data
    color_sampled = np.tile(np.array([[0.,0.,1.]]), (cloud_sampled.shape[0],1)).astype(np.float32)
    red = np.tile(np.array([[1.,0.,0.]]), (cloud_sampled.shape[0],1)).astype(np.float32)
    # color_sampled[np.where(seg_mask == handle_mask[0])] = red[np.where(seg_mask == handle_mask[0])]
    color_sampled[np.where(index_mask)] = red[np.where(index_mask)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_sampled)
    # o3d.visualization.draw_geometries([cloud])

    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    seg_mask = torch.from_numpy(seg_mask[np.newaxis]).to(device)
    index_mask = torch.from_numpy(index_mask[np.newaxis]).to(device)

    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    end_points['seg_mask'] = seg_mask
    end_points['index_mask'] = index_mask

    return end_points, cloud



def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decodeIndex(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    # gg.nms()
    gg.sort_by_score()
    gg = gg[:20]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    # end_points, cloud = get_and_process_data(data_dir)
    doors = os.listdir(data_dir)
    for door in doors:
        end_points, cloud = get_pointcloud(os.path.join(data_dir, door))
        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = '/home/wenhai/data/articulated_data/partmanip/handle'
    demo(data_dir)
