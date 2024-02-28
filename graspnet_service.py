#!/usr/bin/env python
""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import argparse
import torch
from graspnetAPI import GraspGroup

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(ROOT_DIR,'..','..','..')
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector

import rospy
from geometry_msgs.msg import Point, Quaternion
from gamma_msg_srv.srv import GraspNetList,GraspNetListResponse
from gamma_msg_srv.msg import Grasp
from sensor_msgs import point_cloud2
from pyquaternion import Quaternion as Quat
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=os.path.join(ROOT_DIR,'logs/log_rs/checkpoint.tar'), help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
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



def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg



def srv_handle(req):
    pc_msg = req.pointcloud
    try:
        pc = point_cloud2.read_points_list(pc_msg,field_names=("x", "y", "z"))
        pc = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
    # sample points
    if len(pc) >= cfgs.num_point:
        idxs = np.random.choice(len(pc), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(pc))
        idxs2 = np.random.choice(len(pc), cfgs.num_point - len(pc), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = pc[idxs]
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled

    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, pc)
    gg.nms()
    # gg.sort_by_score()
    gg_array = gg.grasp_group_array

    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    gg_list = []
    for _gg in gg_array:
        grasp_score, grasp_width, grasp_height, grasp_depth = _gg[:4]
        rotation_matrix = _gg[4:13]
        grasp_center = _gg[13:16]
        obj_ids = _gg[16]
        gg_msg = Grasp()
        gg_msg.grasp_score, gg_msg.grasp_width, gg_msg.grasp_height, gg_msg.grasp_depth = \
            grasp_score, grasp_width, grasp_height, grasp_depth
        gg_msg.obj_ids = obj_ids
        grasp_center_msg = Point()
        grasp_center_msg.x, grasp_center_msg.y, grasp_center_msg.z = grasp_center
        gg_msg.grasp_center = grasp_center_msg
        rotation = Quaternion()
        quat = Quat._from_matrix(matrix=rotation_matrix.reshape((3,3)), rtol=1e-03, atol=1e-03)
        rotation.w, rotation.x, rotation.y, rotation.z = quat.q
        gg_msg.rotation = rotation
        gg_list.append(gg_msg)
    # gg = GraspGroup(gg_array)
    print("Return %d Grasp."%len(gg_array))
    return GraspNetListResponse(gg=gg_list)

if __name__=='__main__':
    net = get_net()
    rospy.init_node("GraspNet_server")
    srv = rospy.Service("GraspNet",GraspNetList,srv_handle)
    print("Ready to get Grasp List.")
    rospy.spin()