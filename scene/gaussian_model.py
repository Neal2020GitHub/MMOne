# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._thermal_dc = torch.empty(0)  # thermal
        self._thermal_rest = torch.empty(0)  # thermal
        self._thermal_opacity = None  # thermal opacity
        self._language_feature = None  # language
        self._language_opacity = None  # language opacity
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        
        # Decompose gradients of RGB / Thermal / Language
        self.xyz_gradient_accum_rgb = torch.empty(0)
        self.xyz_gradient_accum_thermal = torch.empty(0)
        self.xyz_gradient_accum_language = torch.empty(0)
        
        self.off_state = 0.003
        self.active_threshold = 1 / 256
        
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, args):
        if args.rgb_thermal and args.include_language:
            if args.thermal_density and args.language_density:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._thermal_opacity,
                    self._language_feature,
                    self._language_opacity,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
            else:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._language_feature,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
        
        elif args.rgb_thermal:
            assert self._thermal_dc is not None and self._thermal_rest is not None, "Thermal features are not set!"
            if self._thermal_opacity is not None:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._thermal_opacity,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
            else:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
        
        elif args.include_language:
            assert self._language_feature is not None, "Language feature is not set!"
            if self._language_opacity is not None:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._language_feature,
                    self._language_opacity,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
            else:
                return (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._thermal_dc,
                    self._thermal_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._language_feature,
                    self.max_radii2D,
                    self.xyz_gradient_accum,
                    self.denom,
                    self.optimizer.state_dict(),
                    self.spatial_lr_scale,
                )
        
        else:  # RGB
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args, mode='train'):
        if len(model_args) == 17:
            if training_args.rgb_thermal and training_args.include_language:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._thermal_opacity,
                self._language_feature,
                self._language_opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
        if len(model_args) == 16:
            if training_args.include_language and training_args.language_density:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._language_feature,
                self._language_opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
        elif len(model_args) == 15:
            if training_args.rgb_thermal and training_args.include_language:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._language_feature,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
            elif training_args.rgb_thermal and training_args.thermal_density:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._thermal_opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
            elif training_args.include_language:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._language_feature,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
        elif len(model_args) == 14:
            if training_args.rgb_thermal:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
            else:
                (self.active_sh_degree, 
                self._xyz, 
                self._features_dc, 
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
        elif len(model_args) == 13:
            if training_args.include_language:
                (self.active_sh_degree, 
                self._xyz,
                self._features_dc, 
                self._features_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self._language_feature,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                self.spatial_lr_scale) = model_args
        elif len(model_args) == 12:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        
        if mode == 'train':
            self.training_setup(training_args)
            # If the checkpoint is opacity, we need to continue training the opacity separately, it will raise error
            if len(self.optimizer.param_groups) != len(opt_dict['param_groups']):
                # We need to add the thermal opacity or language opacity to the optimizer
                param = self.optimizer.param_groups[-1].copy()
                param.pop('params', None)
                param['params'] = [len(opt_dict['param_groups'])]
                opt_dict['param_groups'].append(param)
                opt_dict['state'][len(opt_dict['state'])] = {}
            self.optimizer.load_state_dict(opt_dict)
            if training_args.thermal_density:
                self.optimizer.state[self._thermal_opacity] = self.optimizer.state.get(self._opacity, None).copy()
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            
        # self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_thermal_features(self):  
        thermal_dc = self._thermal_dc
        thermal_rest = self._thermal_rest
        return torch.cat((thermal_dc, thermal_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_thermal_opacity(self):
        if self._thermal_opacity is not None:
            return self.opacity_activation(self._thermal_opacity)
        else:
            raise ValueError("Thermal opacity is not set!")
    
    @property
    def get_language_opacity(self):
        if self._language_opacity is not None:
            return self.opacity_activation(self._language_opacity)
        else:
            raise ValueError("Language opacity is not set!")
    
    @property
    def get_language_feature(self):
        if self._language_feature is not None:
            return self._language_feature
        else:
            raise ValueError("Language feature is not set!")
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.training_args = training_args
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        # Decompose gradients of RGB / Thermal / Language
        self.xyz_gradient_accum_rgb = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_accum_thermal = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_accum_language = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        
        self.rgb_mask = self.get_xyz.shape[0]  # RGB mask for densification
        self.rgb_optim_mask = self.get_xyz.shape[0]  # RGB mask for optimizer
        self.rgb_prune_mask = torch.ones((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")  # RGB mask for pruning

        if training_args.rgb_thermal and training_args.include_language:  # RGB + Thermal + Language
            if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
                # Intialize language feature
                language_feature = torch.zeros((self._xyz.shape[0], 3), device="cuda")  # dim = 3
                self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            
            if training_args.thermal_density and training_args.language_density:
                if self._thermal_opacity is None or self._thermal_opacity.shape[0] != self._xyz.shape[0]:
                    self._thermal_opacity = nn.Parameter(self._opacity.data.clone())
                if self._language_opacity is None or self._language_opacity.shape[0] != self._xyz.shape[0]:
                    self._language_opacity = nn.Parameter(self._opacity.data.clone())

                l = [
                        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                        {'params': [self._thermal_dc], 'lr': training_args.feature_lr, "name": "t_dc"},
                        {'params': [self._thermal_rest], 'lr': training_args.feature_lr / 20.0, "name": "t_rest"},
                        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                        {'params': [self._thermal_opacity], 'lr': training_args.opacity_lr, "name": "thermal_opacity"},
                        {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"},
                        {'params': [self._language_opacity], 'lr': training_args.opacity_lr, "name": "language_opacity"}
                    ]
            else:
                assert not training_args.thermal_density and not training_args.language_density, "Thermal density and language density should be enabled at the same time!"
                
                l = [
                        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                        {'params': [self._thermal_dc], 'lr': training_args.feature_lr, "name": "t_dc"},
                        {'params': [self._thermal_rest], 'lr': training_args.feature_lr / 20.0, "name": "t_rest"},
                        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                        {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"},
                    ]
                
        elif training_args.rgb_thermal:  # RGB + Thermal
            if training_args.thermal_density:
                if self._thermal_opacity is None or self._thermal_opacity.shape[0] != self._xyz.shape[0]:
                    self._thermal_opacity = nn.Parameter(self._opacity.data.clone())  # use opacity as thermal opacity initial value

                l = [
                        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                        {'params': [self._thermal_dc], 'lr': training_args.thermal_feature_lr, "name": "t_dc"},
                        {'params': [self._thermal_rest], 'lr': training_args.thermal_feature_lr / 20.0, "name": "t_rest"},
                        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                        {'params': [self._thermal_opacity], 'lr': training_args.opacity_lr, "name": "thermal_opacity"},
                    ]

            else:
                l = [
                        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                        {'params': [self._thermal_dc], 'lr': training_args.thermal_feature_lr, "name": "t_dc"},
                        {'params': [self._thermal_rest], 'lr': training_args.thermal_feature_lr / 20.0, "name": "t_rest"},
                        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                    ]
        
        elif training_args.include_language:  # RGB + Language
            if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
                # Intialize language feature
                language_feature = torch.zeros((self._xyz.shape[0], 3), device="cuda")  # dim = 3
                self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            
            if training_args.language_density:
                if self._language_opacity is None or self._language_opacity.shape[0] != self._xyz.shape[0]:
                    self._language_opacity = nn.Parameter(self._opacity.data.clone())  # Use opacity as language opacity initial value

                l = [
                    {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                    {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                    {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    {'params': [self._thermal_dc], 'lr': 0, "name": "t_dc"},
                    {'params': [self._thermal_rest], 'lr': 0, "name": "t_rest"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                    {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"},
                    {'params': [self._language_opacity], 'lr': training_args.opacity_lr, "name": "language_opacity"}
                ]
            
            else:
                l = [
                    {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                    {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                    {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    {'params': [self._thermal_dc], 'lr': 0, "name": "t_dc"},
                    {'params': [self._thermal_rest], 'lr': 0, "name": "t_rest"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                    {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"}
                ]
        
        else:  # RGB
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._thermal_dc], 'lr': 0, "name": "t_dc"},  # set lr = 0
                {'params': [self._thermal_rest], 'lr': 0, "name": "t_rest"},  # set lr = 0
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
            
            assert self._thermal_feature is None, "Thermal feature should be None when training original 3DGS"
            assert self._thermal_opacity is None, "Thermal opacity should be None when training original 3DGS"
            assert self._language_feature is None, "Language feature should be None when training original 3DGS"
            assert self._language_opacity is None, "Language opacity should be None when training original 3DGS"

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        if self.training_args.thermal_density:  # reset thermal opacity
            thermal_opacities_new = inverse_sigmoid(torch.min(self.get_thermal_opacity, torch.ones_like(self.get_thermal_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(thermal_opacities_new, "thermal_opacity")
            self._thermal_opacity = optimizable_tensors["thermal_opacity"]
        if self.training_args.language_density:  # reset language opacity
            language_opacities_new = inverse_sigmoid(torch.min(self.get_language_opacity, torch.ones_like(self.get_language_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(language_opacities_new, "language_opacity")
            self._language_opacity = optimizable_tensors["language_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        # print("Pruning {} Gaussians".format(mask.sum()))

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["t_dc"]
        self._thermal_rest = optimizable_tensors["t_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if "language_feature" in optimizable_tensors:
            self._language_feature = optimizable_tensors["language_feature"]
            
        if "thermal_opacity" in optimizable_tensors:
            self._thermal_opacity = optimizable_tensors["thermal_opacity"]
        
        if "language_opacity" in optimizable_tensors:
            self._language_opacity = optimizable_tensors["language_opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # Decompose gradients of RGB / Thermal / Language
        self.xyz_gradient_accum_rgb = self.xyz_gradient_accum_rgb[valid_points_mask]
        self.xyz_gradient_accum_thermal = self.xyz_gradient_accum_thermal[valid_points_mask]
        self.xyz_gradient_accum_language = self.xyz_gradient_accum_language[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacities, new_scaling, new_rotation, new_language_feature=None, new_thermal_opacities=None, new_language_opacities=None):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "t_dc": new_thermal_dc,
            "t_rest": new_thermal_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "language_feature" : new_language_feature,
            "thermal_opacity" : new_thermal_opacities,
            "language_opacity" : new_language_opacities,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["t_dc"]  #
        self._thermal_rest = optimizable_tensors["t_rest"]  #
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if new_language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]
        
        if new_thermal_opacities is not None:
            self._thermal_opacity = optimizable_tensors["thermal_opacity"]
        
        if new_language_opacities is not None:
            self._language_opacity = optimizable_tensors["language_opacity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # Decompose gradients of RGB / Thermal / Language
        self.xyz_gradient_accum_rgb = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_accum_thermal = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_accum_language = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print("Splitting {} Gaussians".format(selected_pts_mask.sum()))

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means =torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_thermal_dc = self._thermal_dc[selected_pts_mask].repeat(N, 1, 1)
        new_thermal_rest = self._thermal_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask].repeat(N, 1)
        
        new_thermal_opacity = None
        if self._thermal_opacity is not None:
            new_thermal_opacity = self._thermal_opacity[selected_pts_mask].repeat(N, 1)
        
        new_language_opacity = None
        if self._language_opacity is not None:
            new_language_opacity = self._language_opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacity, new_scaling, new_rotation, new_language_feature, new_thermal_opacity, new_language_opacity)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
        return prune_filter

    def densify_and_clone(self, grads, grad_threshold, scene_extent):        
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_thermal_dc = self._thermal_dc[selected_pts_mask]
        new_thermal_rest = self._thermal_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask]
        
        new_thermal_opacities = None
        if self._thermal_opacity is not None:
            new_thermal_opacities = self._thermal_opacity[selected_pts_mask]
        
        new_language_opacities = None
        if self._language_opacity is not None:
            new_language_opacities = self._language_opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacities, new_scaling, new_rotation, new_language_feature, new_thermal_opacities, new_language_opacities)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # Multimodal Prune
        prune_mask = self.mm_prune(prune_mask, min_opacity)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    
    
    def mm_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, mm_decompose_threshold):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        # Decompose gradients of RGB / Thermal / Language
        grads_rgb = self.xyz_gradient_accum_rgb / self.denom
        grads_rgb[grads_rgb.isnan()] = 0.0
        grads_thermal = self.xyz_gradient_accum_thermal / self.denom
        grads_thermal[grads_thermal.isnan()] = 0.0
        grads_language = self.xyz_gradient_accum_language / self.denom
        grads_language[grads_language.isnan()] = 0.0
        
        # Multimodal Decomposition
        decompose_mask = self.multimodal_decompose(grads, grads_rgb, grads_thermal, grads_language, mm_decompose_threshold)
        # After decompose, these Gaussians cannot clone, split, and prune immediately
        grads[decompose_mask] = 0.0
        save_mask = torch.ones(self.get_xyz.shape[0], device="cuda", dtype=bool)
        save_mask[:len(decompose_mask)] = decompose_mask
        
        self.densify_and_clone(grads, max_grad, extent)
        prune_filter = self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # Multimodal Prune
        prune_mask = self.mm_prune(prune_mask, min_opacity)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # After decompose, these Gaussians cannot be pruned immediately
        save_mask = torch.cat((save_mask[:len(decompose_mask)][~prune_filter[:len(decompose_mask)]], save_mask[len(decompose_mask):]))
        prune_mask[:len(save_mask)][save_mask] = False
        
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def add_multimodal_densification_stats(self, means2D_RGB, means2D_thermal, means2D_language, update_filter):
        means2D_grad = 0.0
        if means2D_RGB.grad is not None:
            means2D_grad += means2D_RGB.grad[update_filter, :2]
        if means2D_thermal.grad is not None:
            means2D_grad += means2D_thermal.grad[update_filter, :2]
        if means2D_language.grad is not None:
            means2D_grad += means2D_language.grad[update_filter, :2]
        self.xyz_gradient_accum[update_filter] += torch.norm(means2D_grad, dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        # Decompose gradients of RGB / Thermal / Language
        if means2D_RGB.grad is not None:
            self.xyz_gradient_accum_rgb[update_filter] += means2D_RGB.grad[update_filter, :2]
        if means2D_thermal.grad is not None:
            self.xyz_gradient_accum_thermal[update_filter] += means2D_thermal.grad[update_filter, :2]
        if means2D_language.grad is not None:
            self.xyz_gradient_accum_language[update_filter] += means2D_language.grad[update_filter, :2]
    
    def mm_prune(self, prune_mask, min_opacity, opacity_threshold=0.5):
        if self.training_args.thermal_density and self.training_args.language_density:
            # Hard Prune
            # prune_mask = torch.logical_or(prune_mask, (self.get_thermal_opacity < min_opacity).squeeze())
            # prune_mask = torch.logical_or(prune_mask, (self.get_language_opacity < min_opacity).squeeze())
            
            # Soft Prune
            prune_mask_save = torch.logical_and(prune_mask, torch.logical_or(self.get_thermal_opacity > opacity_threshold, 
                                                                    self.get_language_opacity > opacity_threshold).squeeze())
            prune_mask = torch.logical_and(prune_mask, ~prune_mask_save)
            self._opacity = self._opacity.index_put(
                (prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
            
            thermal_prune_mask = (self.get_thermal_opacity < min_opacity).squeeze()
            thermal_prune_mask_save = torch.logical_and(thermal_prune_mask, torch.logical_or(self.get_opacity > opacity_threshold,
                                                                            self.get_language_opacity > opacity_threshold).squeeze())
            thermal_prune_mask = torch.logical_and(thermal_prune_mask, ~thermal_prune_mask_save)
            prune_mask = torch.logical_or(prune_mask, thermal_prune_mask)
            self._thermal_opacity = self._thermal_opacity.index_put(
                (thermal_prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
            
            language_prune_mask = (self.get_language_opacity < min_opacity).squeeze()
            language_prune_mask_save = torch.logical_and(language_prune_mask, torch.logical_or(self.get_opacity > opacity_threshold,
                                                                            self.get_thermal_opacity > opacity_threshold).squeeze())
            language_prune_mask = torch.logical_and(language_prune_mask, ~language_prune_mask_save)
            prune_mask = torch.logical_or(prune_mask, language_prune_mask)
            self._language_opacity = self._language_opacity.index_put(
                (language_prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
        
        elif self.training_args.thermal_density:
            # Hard Prune
            # prune_mask = torch.logical_or(prune_mask, (self.get_thermal_opacity < min_opacity).squeeze())
            
            # Soft Prune
            prune_mask_save = torch.logical_and(prune_mask, (self.get_thermal_opacity > opacity_threshold).squeeze())
            prune_mask = torch.logical_and(prune_mask, (self.get_thermal_opacity < opacity_threshold).squeeze())
            self._opacity = self._opacity.index_put(
                (prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
            
            thermal_prune_mask = (self.get_thermal_opacity < min_opacity).squeeze()
            thermal_prune_mask_save = torch.logical_and(thermal_prune_mask, (self.get_opacity > opacity_threshold).squeeze())
            thermal_prune_mask = torch.logical_and(thermal_prune_mask, (self.get_opacity < opacity_threshold).squeeze())
            prune_mask = torch.logical_or(prune_mask, thermal_prune_mask)
            self._thermal_opacity = self._thermal_opacity.index_put(
                (thermal_prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
        
        elif self.training_args.language_density:
            # Hard Prune
            # prune_mask = torch.logical_or(prune_mask, (self.get_language_opacity < min_opacity).squeeze())
            
            # Soft Prune
            prune_mask_save = torch.logical_and(prune_mask, (self.get_language_opacity > opacity_threshold).squeeze())
            prune_mask = torch.logical_and(prune_mask, (self.get_language_opacity < opacity_threshold).squeeze())
            self._opacity = self._opacity.index_put(
                (prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
            
            language_prune_mask = (self.get_language_opacity < min_opacity).squeeze()
            language_prune_mask_save = torch.logical_and(language_prune_mask, (self.get_opacity > opacity_threshold).squeeze())
            language_prune_mask = torch.logical_and(language_prune_mask, (self.get_opacity < opacity_threshold).squeeze())
            prune_mask = torch.logical_or(prune_mask, language_prune_mask)
            self._language_opacity = self._language_opacity.index_put(
                (language_prune_mask_save,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
        
        return prune_mask
    
    def multimodal_decompose(self, grads, grads_rgb, grads_thermal, grads_language, grad_threshold, N=2):
        # Multimodal Densification: Decompose RGB / Thermal / Language based on gradient difference
        grads_rt = torch.norm((grads_rgb - grads_thermal), dim=-1, keepdim=True)
        grads_rl = torch.norm((grads_rgb - grads_language), dim=-1, keepdim=True)
        grads_tl = torch.norm((grads_thermal - grads_language), dim=-1, keepdim=True)
        
        mask_rt = torch.where(torch.norm(grads_rt, dim=-1) >= grad_threshold, True, False)
        mask_rl = torch.where(torch.norm(grads_rl, dim=-1) >= grad_threshold, True, False)
        mask_tl = torch.where(torch.norm(grads_tl, dim=-1) >= grad_threshold, True, False)
        
        # Gaussians need to decompose
        if grads_language.max() == 0:
            selected_pts_mask = mask_rt
            N -= 1
        elif grads_thermal.max() == 0:
            selected_pts_mask = mask_rl
            N -= 1
        else:
            selected_pts_mask = torch.logical_or(torch.logical_or(mask_rt, mask_rl), mask_tl)
            
        # Ensure decompose multi-modal Gaussians rather than single-modal Gaussians
        selected_pts_mask = torch.logical_and(selected_pts_mask, (self.get_opacity > self.active_threshold).squeeze())
        if grads_thermal.max() != 0:
            selected_pts_mask = torch.logical_and(selected_pts_mask, (self.get_thermal_opacity > self.active_threshold).squeeze())
        if grads_language.max() != 0:
            selected_pts_mask = torch.logical_and(selected_pts_mask, (self.get_language_opacity > self.active_threshold).squeeze())
        
        grad_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, grad_mask)
        
        # Do not delete Gaussians, only clone
        new_xyz = self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = new_scaling = self._scaling[selected_pts_mask].repeat(N, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_thermal_dc = self._thermal_dc[selected_pts_mask].repeat(N, 1, 1)
        new_thermal_rest = self._thermal_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask].repeat(N, 1)
        
        new_thermal_opacity = None
        if self._thermal_opacity is not None:
            new_thermal_opacity = self._thermal_opacity[selected_pts_mask].repeat(N, 1)
        
        new_language_opacity = None
        if self._language_opacity is not None:
            new_language_opacity = self._language_opacity[selected_pts_mask].repeat(N, 1)

        # Set original Gaussians to RGB Gaussians, new Gaussians to Thermal / Language Gaussians
        if grads_thermal.max() != 0:
            self._thermal_opacity = self._thermal_opacity.index_put(
                (selected_pts_mask,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
        if grads_language.max() != 0:
            self._language_opacity = self._language_opacity.index_put(
                (selected_pts_mask,),
                self.inverse_opacity_activation(torch.tensor(self.off_state, device="cuda"))
            )
        
        num = torch.sum(selected_pts_mask).item()
        # print("Decomposing {} Gaussians.".format(num), "{:.2f}".format(num/len(grads_rt)))
        new_opacity = inverse_sigmoid(torch.ones_like(new_opacity) * self.off_state)
        if grads_language.max() != 0 and grads_thermal.max() != 0:
            # First num Gaussians are Thermal Gaussians
            new_language_opacity[:num] = inverse_sigmoid(torch.ones_like(new_language_opacity[:num]) * self.off_state)
            # Last num Gaussians are Language Gaussians
            new_thermal_opacity[num:] = inverse_sigmoid(torch.ones_like(new_thermal_opacity[num:]) * self.off_state)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacity, new_scaling, new_rotation, new_language_feature, new_thermal_opacity, new_language_opacity)

        return selected_pts_mask
