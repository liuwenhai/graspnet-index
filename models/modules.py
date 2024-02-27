""" Modules for GraspNet baseline model.
    Author: chenxi-wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2+self.num_view)

    def forward(self, seed_xyz, seed_features, end_points):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, num_seed, _ = seed_xyz.size()
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        objectness_score = features[:, :2, :] # (B, 2, num_seed)
        view_score = features[:, 2:2+self.num_view, :].transpose(1,2).contiguous() # (B, num_seed, num_view)
        end_points['objectness_score'] = objectness_score
        end_points['view_score'] = view_score

        # print(view_score.min(), view_score.max(), view_score.mean())
        top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed)
        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        template_views = generate_grasp_views(self.num_view).to(features.device) # (num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3)
        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot

        return end_points

class ApproachNetIndex(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2+self.num_view)

    def forward(self, seed_xyz, seed_features, end_points):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, num_seed, _ = seed_xyz.size()
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        objectness_score = features[:, :2, :] # (B, 2, num_seed)
        view_score = features[:, 2:2+self.num_view, :].transpose(1,2).contiguous() # (B, num_seed, num_view)
        end_points['objectness_score'] = objectness_score
        end_points['view_score'] = view_score # (B, num_seed, num_view)
        # import pdb;pdb.set_trace()
        # end_points['fp2_inds'], (B, num_seed)
        # end_points['index_mask'] (B, 20000)

        # obtain num_seed grasps from index_mask
        index_mask = torch.gather(end_points['index_mask'], 1, end_points['fp2_inds'].to(torch.int64)) # (B, num_seed)
        if (index_mask.sum(1)<=3).sum():
            print("index_mask.sum().item() <=3, cannot get 1024 grasps")
            import pdb; pdb.set_trace()

        index_num = index_mask.sum(1)
        num_per_seed = (float(num_seed) / index_num * 1.2).to(torch.int32)

        end_points['index'] = []
        end_points['grasp_top_view_inds'] = []
        end_points['grasp_top_view_score'] = []
        end_points['fp2_inds_new'] = []
        end_points['fp2_xyz_new'] = []

        for i in range(B):
            index_num = index_mask[i].sum().item()
            num_per_seed = int(num_seed / float(index_num) * 1.2)

            index = torch.where(index_mask[i])[0]  # where seed is index seed

            sorted_view_score, sorted_view_index = torch.sort(view_score[i], dim=1,descending=True)  # (num_seed, num_view)

            sorted_view_score_ = sorted_view_score[index][:, :num_per_seed]  # (index_num, num_per_seed)
            sorted_view_index_ = sorted_view_index[index][:, :num_per_seed]  # # (index_num, num_per_seed)

            end_points['index'].append(index)

            top_view_scores, index_seed = torch.sort(sorted_view_score_.flatten(), -1, descending=True)
            top_view_scores, index_seed = top_view_scores[:num_seed], index_seed[:num_seed] # (num_seed,)

            end_points['grasp_top_view_score'].append(top_view_scores)

            top_view_inds = sorted_view_index_.flatten()[index_seed].contiguous()  # # (num_seed,)
            end_points['grasp_top_view_inds'].append(top_view_inds)


            fp2_inds_new = index.repeat_interleave(num_per_seed)[index_seed] # (num_seed,)
            end_points['fp2_inds_new'].append(fp2_inds_new)

        end_points['index'] = torch.stack(end_points['index'], 0)
        end_points['grasp_top_view_inds'] = torch.stack(end_points['grasp_top_view_inds'], 0)
        end_points['grasp_top_view_score'] = torch.stack(end_points['grasp_top_view_score'], 0)
        end_points['fp2_inds_new'] = torch.stack(end_points['fp2_inds_new'], 0)

        fp2_inds_new = end_points['fp2_inds_new'].view(1, -1, 1).expand(-1, -1, 3)
        end_points['fp2_xyz_new'] = torch.gather(end_points['fp2_xyz'], 1, fp2_inds_new)

        template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1,-1).contiguous()  # (B, num_seed, num_view, 3)

        top_view_inds_ = end_points['grasp_top_view_inds'].view(B, num_seed, 1, 1).expand(-1, -1, -1,3).contiguous()  # (B, num_seed,1,3)
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)

        end_points['grasp_top_view_xyz'] = vp_xyz  # # (B, num_seed, 3)
        end_points['grasp_top_view_rot'] = vp_rot  # (B, num_seed, 3, 3)

        # ## suppose B = 1
        # index_num = index_mask.sum().item()
        # index = torch.where(index_mask)  # where seed is index seed
        # num_per_seed = int(num_seed/float(index_num)*1.2)
        # sorted_view_score, sorted_view_index = torch.sort(view_score,dim=2,descending=True) # (B, num_seed, num_view)
        #
        #
        # sorted_view_score_ = sorted_view_score[index][:, :num_per_seed] # (index_num, num_per_seed)
        # sorted_view_index_ = sorted_view_index[index][:, :num_per_seed]  # # (index_num, num_per_seed)
        # end_points['index'] = index[-1].view(B,-1)
        #
        # top_view_scores, index_seed = torch.sort(sorted_view_score_.flatten(),-1,descending=True)
        # top_view_scores, index_seed = top_view_scores[:num_seed], index_seed[:num_seed]
        # top_view_scores = top_view_scores.view(B, -1).contiguous()
        #
        # top_view_inds = sorted_view_index_.flatten()[index_seed].view(B, -1).contiguous() # # (B, num_seed)
        # top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()  # (B, num_seed,1,3)
        #
        # fp2_inds_new = index[1].repeat_interleave(num_per_seed)[index_seed]
        # end_points['fp2_inds_new'] = fp2_inds_new.view(B,-1)
        # fp2_inds_new = fp2_inds_new.view(1,-1,1).expand(-1,-1,3)
        # end_points['fp2_xyz_new'] = torch.gather(end_points['fp2_xyz'],1,fp2_inds_new)
        #
        # # import pdb;pdb.set_trace()
        # # fp2_xyz_new = end_points['fp2_xyz_new'].cpu().numpy()[0]
        # # fp2_xyz = end_points['fp2_xyz'].cpu().numpy()[0]
        # # import numpy as np
        # # import open3d as o3d
        # #
        # # color_sampled = np.tile(np.array([[0., 0., 1.]]), (fp2_xyz.shape[0], 1)).astype(np.float32)
        # # red = np.tile(np.array([[1., 0., 0.]]), (fp2_xyz.shape[0], 1)).astype(np.float32)
        # # # color_sampled[np.where(instance_mask==handle_mask[2])] = red[np.where(instance_mask==handle_mask[2])]
        # # color_sampled[index[-1].cpu().numpy()] = red[index[-1].cpu().numpy()]
        # # cloud = o3d.geometry.PointCloud()
        # # cloud.points = o3d.utility.Vector3dVector(fp2_xyz.astype(np.float32))
        # # # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        # # cloud.colors = o3d.utility.Vector3dVector(color_sampled)
        # #
        # # cloud1 = o3d.geometry.PointCloud()
        # # cloud1.points = o3d.utility.Vector3dVector(fp2_xyz_new.astype(np.float32))
        # # black = np.tile(np.array([[0., 0., 0.]]), (fp2_xyz_new.shape[0], 1)).astype(np.float32)
        # # cloud1.colors = o3d.utility.Vector3dVector(black)
        # #
        # # o3d.visualization.draw_geometries([cloud,cloud1])
        #
        # template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
        # template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()  # (B, num_seed, num_view, 3)
        #
        # vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
        # vp_xyz_ = vp_xyz.view(-1, 3)
        # batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        # end_points['grasp_top_view_inds'] = top_view_inds  # (B, num_seed)
        # end_points['grasp_top_view_score'] = top_view_scores  # (B, num_seed)
        # end_points['grasp_top_view_xyz'] = vp_xyz  # # (B, num_seed, 3)
        # end_points['grasp_top_view_rot'] = vp_rot  # (B, num_seed, 3, 3)


        return end_points

class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]
        
        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            )) # (batch_size, feature_dim, num_seed, nsample)
        grouped_features = torch.stack(grouped_features, dim=3) # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed*num_depth, self.nsample) # (batch_size, feature_dim, num_seed*num_depth, nsample)

        vp_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features

        
class OperationNet(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3*num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2*self.num_angle:3*self.num_angle]
        return end_points

    
class ToleranceNet(nn.Module):
    """ Grasp tolerance prediction.
    
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points