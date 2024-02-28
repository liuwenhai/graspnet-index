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

from graspnetAPI import GraspGroup
from sensor_msgs.msg import PointCloud2, PointField
from gamma_msg_srv.srv import GraspNetList
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

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    return cloud


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])



if __name__=='__main__':
    data_dir = 'doc/example_data'
    data_dir = os.path.join(ROOT_DIR, data_dir)
    cloud = get_and_process_data(data_dir)
    pc = np.array(cloud.points)
    pc_msg = xyzl_array_to_pointcloud2(pc)
    rospy.wait_for_service("GraspNet")
    try:
        srv_handle = rospy.ServiceProxy("GraspNet",GraspNetList)
        rep = srv_handle(pc_msg)
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