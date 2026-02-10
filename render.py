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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    if args.joint:
        rgb_path = os.path.join(model_path, "rgb")
        if args.rgb_thermal:
            thermal_path = os.path.join(model_path, "thermal")
        if args.include_language:
            language_path = os.path.join(model_path, "language")
    else:
        model_path = os.path.join(model_path, "rgb")
    
    if args.joint:
        rgb_render_path = os.path.join(rgb_path, name, "ours_{}".format(iteration), "renders")
        rgb_gts_path = os.path.join(rgb_path, name, "ours_{}".format(iteration), "gt")
        makedirs(rgb_render_path, exist_ok=True)
        makedirs(rgb_gts_path, exist_ok=True)
        if args.rgb_thermal:
            thermal_render_path = os.path.join(thermal_path, name, "ours_{}".format(iteration), "renders")
            thermal_gts_path = os.path.join(thermal_path, name, "ours_{}".format(iteration), "gt")
            makedirs(thermal_render_path, exist_ok=True)
            makedirs(thermal_gts_path, exist_ok=True)
        if args.include_language:
            language_render_path = os.path.join(language_path, name, "ours_{}".format(iteration), "renders")
            language_gts_path = os.path.join(language_path, name, "ours_{}".format(iteration), "gt")
            language_render_npy_path = os.path.join(language_path, name, "ours_{}".format(iteration), "renders_npy")
            language_gts_npy_path = os.path.join(language_path, name, "ours_{}".format(iteration), "gt_npy")
            makedirs(language_render_path, exist_ok=True)
            makedirs(language_gts_path, exist_ok=True)
            makedirs(language_render_npy_path, exist_ok=True)
            makedirs(language_gts_npy_path, exist_ok=True)
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)
        if args.joint:
            rgb = output["image"]
            if args.rgb_thermal:
                thermal = output["thermals"]
            if args.include_language:
                language = output["language"]
        elif args.include_language:
            rendering = output["language"]
        else:
            rendering = output["image"]
        
        if args.joint:
            gt_rgb = view.original_image[0:3, :, :]
            if args.rgb_thermal:
                gt_thermal = view.thermal[0:3, :, :]
            if args.include_language:
                gt, mask = view.get_language_feature(language_feature_dir=os.path.join(source_path, "language_features_dim3"), feature_level=args.feature_level)
        else:
            gt = view.original_image[0:3, :, :]
        
        if args.joint:
            torchvision.utils.save_image(rgb, os.path.join(rgb_render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt_rgb, os.path.join(rgb_gts_path, '{0:05d}'.format(idx) + ".png"))
            if args.rgb_thermal:
                torchvision.utils.save_image(thermal, os.path.join(thermal_render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt_thermal, os.path.join(thermal_gts_path, '{0:05d}'.format(idx) + ".png"))
            if args.include_language:
                torchvision.utils.save_image(language, os.path.join(language_render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(language_gts_path, '{0:05d}'.format(idx) + ".png"))
                np.save(os.path.join(language_render_npy_path, '{0:05d}'.format(idx) + ".npy"), language.permute(1, 2, 0).cpu().numpy())
                np.save(os.path.join(language_gts_npy_path, '{0:05d}'.format(idx) + ".npy"), gt.permute(1, 2, 0).cpu().numpy())
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if args.rgb_thermal and args.include_language:
            checkpoint = os.path.join(args.model_path, 'thermal_language_chkpnt30000.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
        elif args.rgb_thermal:
            checkpoint = os.path.join(args.model_path, 'thermal_chkpnt30000.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
        elif args.include_language:
            checkpoint = os.path.join(args.model_path, 'language_chkpnt30000.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
        else:
            checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, dataset.source_path, "train", first_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            render_set(dataset.model_path, dataset.source_path, "test", first_iter, scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--joint", action="store_true", default=True)  # Joint training
    parser.add_argument("--rgb_thermal", action="store_true", default=False)  # thermal
    parser.add_argument("--thermal_density", action="store_true", default=False)  # Density for thermal
    parser.add_argument("--include_language", action="store_true", default=False)  # language
    parser.add_argument("--language_density", action="store_true", default=False)  # Density for language
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    if args.include_language:
        args.feature_level = int(args.model_path[-1])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
