#
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
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, args, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Decompose gradients of RGB / Thermal / Language
    means2D_RGB = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    means2D_thermal = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    means2D_language = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D_RGB.retain_grad()
        means2D_thermal.retain_grad()
        means2D_language.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    from diff_gaussian_rasterization import GaussianRasterizationSettings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    from diff_gaussian_rasterization import GaussianRasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # RGB / Thermal rasterizer
    
    if args.include_language:
        from language_rasterization import GaussianRasterizer
        language_rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # Language rasterizer
        
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3D covariance is provided, use it. If not, then it will be computed from scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors from SHs in Python, do it. 
    # If not, then SH -> RGB conversion will be done by rasterizer.
    thermal_shs = None
    thermal_precomp = None
    shs = None
    colors_precomp = None
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            thermal_shs = pc.get_thermal_features  # ThermalGaussian
    else:
        colors_precomp = override_color
    
    if args.thermal_density:
        thermal_opacity = pc.get_thermal_opacity
    else:
        thermal_opacity = opacity
        
    if args.include_language:
        language_feature_precomp = pc.get_language_feature
        language_feature_precomp = language_feature_precomp / (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)  # normalize
    else:
        language_feature_precomp = None
    
    if args.language_density:
        language_opacity = pc.get_language_opacity
    else:
        language_opacity = opacity
    
    # RGB
    image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D_RGB,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Thermal
    thermals = None
    if args.rgb_thermal:
        thermals, radii = rasterizer(
            means3D = means3D,
            means2D = means2D_thermal,
            shs = thermal_shs,
            colors_precomp = thermal_precomp,
            opacities = thermal_opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
    
    # Language
    language = None
    if args.include_language:
        _, language, radii, _ = language_rasterizer(
            means3D = means3D,
            means2D = means2D_language,
            shs = shs,
            colors_precomp = colors_precomp,
            language_feature_precomp = language_feature_precomp,
            opacities = language_opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    image = image.clamp(0, 1)
    if thermals is not None:
        thermals = thermals.clamp(0, 1)

    out = {
        "thermals": thermals,
        "image": image,
        "viewspace_points": screenspace_points,
        "means2D_RGB": means2D_RGB,
        "means2D_thermal": means2D_thermal,
        "means2D_language": means2D_language,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "language": language,
    }
    
    return out
