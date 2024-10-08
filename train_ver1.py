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
import numpy as np
import random
import os
import torch 
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss, GradL1Loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef
import torch.nn.functional as F
import lpips
from utils.scene_utils import render_training_image
from time import time
def to8b(x): return (255*np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter),
                        desc="Training progress")
    first_iter += 1

    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()

    for iteration in range(first_iter, final_iter+1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []

        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        #extent 
        gs_normal = []
        confidences = []

        for viewpoint_cam in viewpoint_cams:
            # render_pkg = render(viewpoint_cam, gaussians,
            #                     pipe, background, stage=stage)
            #EXTENT
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,
                                cam_type=scene.dataset_type, multi_scale=False, iteration=0)
            #----------------------------------------------------------------------------------

            # image, depth, viewspace_point_tensor, visibility_filter, radii = \
            #     render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # EXTENT 
            image, viewspace_point_tensor, radii, depth = \
                render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["radii"], render_pkg['depth']
            visibility_filter = radii > 0
            #-----------------------------------------------------------
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()

            images.append(image.unsqueeze(0))
            gt_depth = gt_depth.permute(2,0,1)
            # print(gt_depth.shape)
            # print(depth.shape)
            dep_mask = torch.logical_and(gt_depth>0,depth>0)
            gt_depth = gt_depth * dep_mask
            depth = depth * dep_mask

            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

            #extend 
            gs_normal.append(render_pkg['normal'].unsqueeze(0))
            confidences.append(render_pkg['confidence'].unsqueeze(0))
            #------------------------------------------

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        mask_tensor = torch.cat(masks,0)
        
        rendered_images = torch.cat(images, 0) * mask_tensor
        rendered_depths = torch.cat(depths, 0) * mask_tensor
        gt_images = torch.cat(gt_images, 0) * mask_tensor
        gt_depths = torch.cat(gt_depths, 0) * mask_tensor
        # gt_depths = gt_depths.permute(0,3,1,2)
        masks = torch.cat(masks, 0)

        #extend 
        gs_normal = torch.cat(gs_normal, 0) * mask_tensor
        confidences = torch.cat(confidences, 0) * mask_tensor
        #----------------------------------------------
        Ll1 = l1_loss(rendered_images, gt_images, mask_tensor)

        #ALTERNATIVE
        # if (gt_depths != 0).sum() < 10:
        #     depth_loss = torch.tensor(0.).cuda()
        # rendered_depths_reshape = rendered_depths.reshape(-1, 1)
        # gt_depths_reshape = gt_depths.reshape(-1, 1)
        # depth_loss = 0.001 * \
        #     (1 - pearson_corrcoef(gt_depths_reshape, rendered_depths_reshape))

        #EXTENT loss depth
        depth_weight = 0.01
        # print('rendered_depths: ',rendered_depths.shape)
        # print('gt_depths: ',gt_depths.shape)
        depth_loss = l1_loss(rendered_depths/(rendered_depths.max()+1e-6), gt_depths/(gt_depths.max()+1e-6),
                                mask=mask_tensor)*depth_weight
        loss = Ll1 + depth_loss
        #-----------------------------------------------
        #EXTEND  grad loss
        grad_loss = GradL1Loss()
        grad_weight = 0.001
        sm_loss = grad_loss(rendered_depths, gt_depths,mask_tensor) * grad_weight
        loss += sm_loss
        #-----------------------------------------------



        #EXTENT normal loss 
        from utils.graphics_utils import get_pseudo_normal
        from utils.loss_utils import mae_loss
        normal_weight = 0.001
        pseudo_normal = get_pseudo_normal(gt_depths, mask_tensor.unsqueeze(0))
        pseudo_normal = F.interpolate(pseudo_normal, gs_normal.shape[2:4])
        normal_loss = mae_loss(
            gs_normal, pseudo_normal, mask_tensor)*normal_weight
        loss += normal_loss
        #-----------------------------------------------


        #EXTEND confidence
        from utils.loss_utils import confidence_loss
        un_img_weight = 0.001
        un_dep_weight = 0.001
        confidence_loss_img = confidence_loss(gt_images, rendered_images,
                                                confidences, mask_tensor.unsqueeze(0))*un_img_weight
        confidence_loss_dep = confidence_loss(gt_depths/gt_depths.max(),
                                                rendered_depths/rendered_depths.max(), confidences, mask_tensor.unsqueeze(0))*un_dep_weight
        loss += confidence_loss_img
        loss += confidence_loss_dep
        #----------------------------------------------

        # mask_tmp = mask.reshape(-1)
        # rendered_depths_reshape, gt_depths_reshape = rendered_depths_reshape[
        #     mask_tmp != 0, :], gt_depths_reshape[mask_tmp != 0, :]
        # else:
        #     raise ValueError(f"{scene.mode} is not implemented.")

        # depth_tvloss = TV_loss(rendered_depths)
        # img_tvloss = TV_loss(rendered_images)
        # tv_loss = 0.03 * (img_tvloss + depth_tvloss)

        # loss = Ll1 + depth_loss + tv_loss

        psnr_ = psnr(rendered_images, gt_images, mask=mask_tensor).mean().double()

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(2e-2, 2e-2, 2e-2)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images, gt_images)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips != 0:
            lpipsloss = lpips_loss(rendered_images, gt_images, lpips_model)
            loss += opt.lambda_lpips * lpipsloss

        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + \
                viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                #EXTENT
                # string_dict = {"Loss": f"{Ll1.item():.{4}f}",
                #                "psnr": f"{psnr_:.{2}f}"}
                # if stage == 'fine':
                #     string_dict['tv'] = f"{tv_loss:.{4}f}"
                # string_dict["Dep"] = f"{depth_loss:.{4}f}"

                # string_dict["Sm"] = f"{sm_loss:.{4}f}"
                # string_dict["Norm"] = f"{normal_loss:.{4}f}"
                # string_dict["Un_img"] = f"{confidence_loss_img:.{4}f}"
                # string_dict["Un_dep"] = f"{confidence_loss_dep:.{4}f}"

                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(
                iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                        or (iteration < 3000 and iteration % 50 == 1) \
                    or (iteration < 10000 and iteration % 100 == 1) \
                        or (iteration < 60000 and iteration % 100 == 1):
                    render_training_image(scene, gaussians, video_cams, render, pipe,
                                          background, stage, iteration-1, timer.get_elapsed_time())
            timer.start()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * \
                        (opt.opacity_threshold_fine_init -
                         opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * \
                        (opt.densify_grad_threshold_fine_init -
                         opt.densify_grad_threshold_after)/(opt.densify_until_iter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(
                        densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold,
                                    scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
    if not args.no_fine:
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "fine", tb_writer, opt.iterations, timer)


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(
            f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(
            f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    '''
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask = viewpoint.mask.to("cuda")
                    
                    image, gt_image, mask = image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
        '''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+",
                        type=int, default=[i*500 for i in range(0, 120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[
                        2000, 3000, 4000, 5000, 6000, 9000, 10000, 12000, 14000, 16000, 30000, 45000, 60000])
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        # from mmengine.config import Config
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
