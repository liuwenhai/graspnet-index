#!/usr/bin/env python
""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import scipy.io as scio
from PIL import Image
import torch

from graspnetAPI import GraspGroup
from sensor_msgs.msg import PointCloud2, PointField
from gamma_msg_srv.srv import GraspNetIndexList
import rospy
from pyquaternion import Quaternion as Quat

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(ROOT_DIR,'..','..','..')
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

def xyzl_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs.PointCloud2 from an array
        of points (x, y, z, l)
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            # PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg

def get_pointcloud(data_dir):
    data = np.load(data_dir)       # ['cam2_wolrd', 'per_coord_world', 'instance_mask', 'handle_mask']
    cam2_wolrd = data['cam2_wolrd']
    per_coord_world = data['per_coord_world']
    instance_mask = data['instance_mask']
    handle_mask = data['handle_mask']


    r = cam2_wolrd[:3,:3].T
    t = - r @ cam2_wolrd[:3,3]
    cloud_masked = per_coord_world @ r.T + t.T

    index_mask = np.zeros(instance_mask.shape).astype(np.int32)
    for handle_index in handle_mask:
        index_mask = np.logical_or(instance_mask == handle_index, index_mask)
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    return cloud_masked, index_mask, cloud


def vis_grasps(gg, cloud):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers, axis_pcd])



if __name__=='__main__':
    data_dir = 'handle'

    doors = os.listdir(data_dir)
    for door in doors:
        point_clouds, index_mask, cloud = get_pointcloud(os.path.join(data_dir, door))
        pc_msg = xyzl_array_to_pointcloud2(point_clouds)
        index_mask_msg = index_mask.astype(np.int8).tolist()
        rospy.wait_for_service("GraspNetIndex")

        try:
            srv_handle = rospy.ServiceProxy("GraspNetIndex",GraspNetIndexList)
            rep = srv_handle(pointcloud=pc_msg,index_mask=index_mask_msg)
            gg_list = rep.gg
            # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
            gg_array = []
            for gg in gg_list:
                grasp_score = gg.grasp_score
                grasp_width = gg.grasp_width
                grasp_height = gg.grasp_height
                grasp_depth = gg.grasp_depth
                pre_ = [grasp_score, grasp_width, grasp_height, grasp_depth]
                rotation = gg.rotation
                grasp_center = gg.grasp_center
                obj_ids = gg.obj_ids
                quat = Quat(rotation.w, rotation.x, rotation.y, rotation.z)
                rotation_matrix = quat.rotation_matrix.flatten().tolist()
                grasp_center = [grasp_center.x, grasp_center.y, grasp_center.z]
                gg_array.append(pre_ + rotation_matrix + grasp_center + [obj_ids])
            gg_array = np.array(gg_array)
            gg = GraspGroup(gg_array)
            vis_grasps(gg, cloud)

        except rospy.ServiceException as e:
            print("Service call failed : %s"%e)
    