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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from utils.loss_utils import smoothness_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = \
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=False)
    if not checkpoint:  # If checkpoint, setup when restore
        gaussians.training_setup(args)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    
    ema_rgb_for_log = 0.0
    ema_thermal_for_log = 0.0
    ema_language_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, args)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        rgb_loss, thermal_loss, language_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        if args.joint and args.rgb_thermal and args.include_language:  # RGB + Thermal + Language
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)  # l1 loss on RGB
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            thermals = render_pkg["thermals"]
            gt_thermals = viewpoint_cam.thermal.cuda()
            Ll1_thermal = l1_loss(thermals, gt_thermals)
            smoothloss_thermal = smoothness_loss(thermals)
            thermal_loss = (1.0 - opt.lambda_dssim) * Ll1_thermal + opt.lambda_dssim * (1.0 - ssim(thermals, gt_thermals)) + 0.6 * smoothloss_thermal
            
            language = render_pkg["language"]
            gt_language, language_mask = viewpoint_cam.get_language_feature(language_feature_dir=args.lf_path, feature_level=args.feature_level)
            language_loss = l1_loss(language * language_mask, gt_language * language_mask)
            
            loss = 0.5 * rgb_loss + 0.5 * thermal_loss + 0.2 * language_loss
        
        elif args.joint and args.rgb_thermal:  # RGB + Thermal
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            thermals = render_pkg["thermals"]
            gt_thermals = viewpoint_cam.thermal.cuda()
            Ll1_thermal = l1_loss(thermals, gt_thermals)
            smoothloss_thermal = smoothness_loss(thermals)
            thermal_loss = (1.0 - opt.lambda_dssim) * Ll1_thermal + opt.lambda_dssim * (1.0 - ssim(thermals, gt_thermals)) + 0.6 * smoothloss_thermal
            
            loss = (rgb_loss + thermal_loss) * 0.5
        
        elif args.joint and args.include_language:  # RGB + Language
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            language = render_pkg["language"]
            gt_language, language_mask = viewpoint_cam.get_language_feature(language_feature_dir=args.lf_path, feature_level=args.feature_level)
            language_loss = l1_loss(language * language_mask, gt_language * language_mask)
            
            loss = (rgb_loss + language_loss) * 0.5
        
        else:  # RGB only
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss = rgb_loss
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_rgb_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_rgb_for_log
            ema_thermal_for_log = 0.4 * thermal_loss.item() + 0.6 * ema_thermal_for_log
            ema_language_for_log = 0.4 * language_loss.item() + 0.6 * ema_language_for_log
            
            if iteration % 10 == 0:
                if args.rgb_thermal and args.include_language:
                    loss_dict = {
                            "RGB": f"{ema_rgb_for_log:.{5}f}",
                            "Thermal": f"{ema_thermal_for_log:.{5}f}",
                            "Language": f"{ema_language_for_log:.{5}f}"
                        }
                elif args.rgb_thermal:
                    loss_dict = {
                            "RGB": f"{ema_rgb_for_log:.{5}f}",
                            "Thermal": f"{ema_thermal_for_log:.{5}f}"
                        }
                elif args.include_language:
                    loss_dict = {
                            "RGB": f"{ema_rgb_for_log:.{5}f}",
                            "Language": f"{ema_language_for_log:.{5}f}"
                        }
                else:
                    loss_dict = {
                            "RGB": f"{ema_rgb_for_log:.{5}f}"
                        }
                
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/rgb_loss', ema_rgb_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/thermal_loss', ema_thermal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/language_loss', ema_language_for_log, iteration)
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, args))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # Decompose gradients of RGB / Thermal / Language
            means2D_RGB, means2D_thermal, means2D_language = render_pkg["means2D_RGB"], render_pkg["means2D_thermal"], render_pkg["means2D_language"]
            gaussians.add_multimodal_densification_stats(means2D_RGB, means2D_thermal, means2D_language, visibility_filter)
            
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.mm_densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.mm_decompose_threshold)
            
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()
        
        with torch.no_grad():
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if args.rgb_thermal and args.include_language:
                    torch.save((gaussians.capture(args), iteration), scene.model_path + "/thermal_language_chkpnt" + str(iteration) + ".pth")
                elif args.rgb_thermal:
                    torch.save((gaussians.capture(args), iteration), scene.model_path + "/thermal_chkpnt" + str(iteration) + ".pth")
                elif args.include_language:
                    torch.save((gaussians.capture(args), iteration), scene.model_path + "/language_chkpnt" + str(iteration) + ".pth")
                else:
                    torch.save((gaussians.capture(args), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_rgb, l1_test_thermal, l1_test_language = 0.0, 0.0, 0.0
                psnr_test_rgb, psnr_test_thermal = 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["image"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if args.joint:
                        l1_test_rgb += l1_loss(image, gt_image).mean().double()
                        psnr_test_rgb += psnr(image, gt_image).mean().double()
                        
                        if args.rgb_thermal:
                            thermal = render_pkg["thermals"]
                            gt_thermal = viewpoint.thermal.to("cuda")
                            l1_test_thermal += l1_loss(thermal, gt_thermal).mean().double()
                            psnr_test_thermal += psnr(thermal, gt_thermal).mean().double()
                        
                        if args.include_language:
                            language = render_pkg["language"]
                            gt_language, language_mask = viewpoint.get_language_feature(language_feature_dir=args.lf_path, feature_level=args.feature_level)
                            l1_test_language += l1_loss(language * language_mask, gt_language * language_mask).mean().double()                  
                    
                    else:
                        l1_test_rgb += l1_loss(image, gt_image).mean().double()
                        psnr_test_rgb += psnr(image, gt_image).mean().double()
                
                psnr_test_rgb /= len(config['cameras'])
                l1_test_rgb /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: RGB: L1 {} PSNR {}".format(iteration, config['name'], l1_test_rgb, psnr_test_rgb))
                if args.rgb_thermal:
                    psnr_test_thermal /= len(config['cameras'])
                    l1_test_thermal /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: Thermal: L1 {} PSNR {}".format(iteration, config['name'], l1_test_thermal, psnr_test_thermal))
                if args.include_language:
                    l1_test_language /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: Language: L1 {}".format(iteration, config['name'], l1_test_language))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '_RGB' + '/loss_viewpoint - l1_loss', l1_test_rgb, iteration)
                    tb_writer.add_scalar(config['name'] + '_RGB' + '/loss_viewpoint - psnr', psnr_test_rgb, iteration)
                    if args.rgb_thermal:
                        tb_writer.add_scalar(config['name'] + '_Thermal' + '/loss_viewpoint - l1_loss', l1_test_thermal, iteration)
                        tb_writer.add_scalar(config['name'] + '_Thermal' + '/loss_viewpoint - psnr', psnr_test_thermal, iteration)
                    if args.include_language:
                        tb_writer.add_scalar(config['name'] + '_Language' + '/loss_viewpoint - l1_loss', l1_test_language, iteration)
                    
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)  # Optimization parameters
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])  # save checkpoints
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    parser.add_argument("--joint", action="store_true", default=True)  # Joint training
    parser.add_argument("--rgb_thermal", action="store_true", default=False)  # Train thermal
    parser.add_argument("--thermal_density", action="store_true", default=False)  # Density for thermal
    parser.add_argument("--include_language", action="store_true", default=False)  # Train language
    parser.add_argument("--language_density", action="store_true", default=False)  # Density for language
    parser.add_argument("--feature_level", type=int, default=-1)  # Feature level for language
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # print(args)

    if args.include_language:
        assert args.feature_level >= 0, "Feature level must be set to 0, 1, 2 or 3"
        args.model_path = args.model_path + f"_{str(args.feature_level)}"  # Add feature level to model path
        args.lf_path = os.path.join(args.source_path, "language_features_dim3")  # Language feature path
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
