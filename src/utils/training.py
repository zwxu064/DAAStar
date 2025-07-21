# -------------------------------------------------------------------------------------------------------------
# File: training.py
# Project: DAA*: Deep Angular A Star for Image-based Path Planning
# Contributors:
#     Zhiwei Xu <zwxu064@gmail.com>
#
# Copyright (c) 2025 Zhiwei Xu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import random, re, sys, os, shutil
import numpy as np
import pytorch_lightning as pl
import torch, torch.optim
import torch.nn as nn

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.planner.astar import VanillaAstar
from src.utils.metrics import cal_path_angle, generate_path_stack, chamfer_distance, calculate_area_sim
from src.utils.viz import save_figure_mixed

# =============================================================================================================

def statistics_all_methods(
    methods,
    neural_astars,
    test_loader,
    config,
    device='cuda',
    store_hist_coordinates=False,
    enable_save_path_stack=False,
    resume_path=None,
    enable_save_image=False,
    split="test",
    save_image_list=None,
    disable_compute_path_angle=False
):
    # Run over data loop
    all_info_dict = {}

    for method, astar_obj in zip(methods, neural_astars):
        astar_obj.eval()
        planner = PlannerModule(astar_obj, config).to(device)
        sample_count = 0

        output_list = []

        if resume_path is not None and enable_save_image:
            if resume_path.find('/focal.pth') > -1:
                save_dir = config.transpath.save_dir
            else:
                save_dir = resume_path.replace('.ckpt', '')
                save_dir = save_dir.replace('.pth', '')
                save_dir = save_dir.replace(f"/{save_dir.split('/')[-2]}", '')
                save_dir = save_dir.replace('model_weights', 'results')

            if os.path.exists(save_dir): shutil.rmtree(save_dir)  # avoid historic generations
        else:
            save_dir = \
                f"{config.motion_planning_lib.save_dir}" \
                f"/{config.dataset_name}/{config.dataset_file_name}" \
                f"/{split}"

        for idx, data in enumerate(test_loader):
            if idx % 10 == 0: print(f'Batch: {idx+1}/{len(test_loader)}.')

            # Put data into gpu or cpu
            for key in data.keys():
                data[key] = data[key].to(device)

            #
            viz_maps = data['maps']
            img_c, img_h, img_w = viz_maps.shape[-3:]
            h, w = data['starts'].shape[-2:]

            if img_c > 3:  # for some maze datasets batch*rand*h*w, others batch*c*h*w
                viz_maps = viz_maps.view(-1, 1, img_h, img_w)
            else:
                viz_maps = viz_maps.view(-1, img_c, img_h, img_w)

            num_samples = viz_maps.shape[0]
            viz_idx_list = [sample_count + inner_idx for inner_idx in range(num_samples)]

            if enable_save_image \
                and save_image_list is not None \
                and len(save_image_list) > 0:
                viz_idx_exist = np.sum([v in viz_idx_list for v in save_image_list]) > 0

                if not viz_idx_exist:
                    sample_count += num_samples
                    continue

            #
            outputs = planner.validation_step(
                data,
                idx,
                store_hist_coordinates=store_hist_coordinates,
                enable_save_path_stack=enable_save_path_stack,
                disable_compute_path_angle=disable_compute_path_angle
            )

            if outputs is None:
                continue

            output_list.append(outputs)

            # Draw and save images for writing
            if enable_save_image:
                # Gray: data['maps'] batch*15*h*w, outputs['nAstar']['paths']: (batch*15)*1*h*w
                # RGB: data['map'] batch*3*h*w, outputs['nAstar']['paths']: batch*1*h*w
                # Warcraft and Pokemon resize
                for inner_idx in range(num_samples):
                    viz_idx = sample_count + inner_idx

                    if save_image_list is not None and len(save_image_list) > 0 and viz_idx not in save_image_list:
                        continue

                    # aug_tmpd has around 64000 val and test samples, only save the first 2000
                    if config.dataset_name in ['aug_tmpd'] and viz_idx >= 2000: continue

                    # 20240702: random sampling within the dataloader
                    if config.dataset_name in ['mpd', 'tmpd', 'street']:
                        num_random_samplings = 3
                    elif config.dataset_name in ['warcraft', 'pkmn']:
                        num_random_samplings = 4
                    else:
                        num_random_samplings = 1

                    # if viz_idx in save_image_indices:
                    num_starts = config.num_starts_valid if split == "val" else config.num_starts_test

                    if viz_idx % (num_starts * num_random_samplings) == 0:
                        if config.enable_motion_planning_lib and not config.motion_planning_lib.save_solution:
                            save_dir_idx = f'{save_dir}/{config.motion_planning_lib.method}/{viz_idx}'
                        else:
                            save_dir_idx = f'{save_dir}/{viz_idx}'

                        os.makedirs(save_dir_idx, exist_ok=True)

                        viz_starts = data['starts'].view(-1, 1, h, w)
                        viz_goals = data['goals'].view(-1, 1, h, w)

                        for viz_method in ['GT', 'nAstar', 'vAstar']:
                            viz_obj = outputs[viz_method]
                            if viz_obj is None: continue
                            viz_paths = viz_obj['paths'].view(-1, 1, h, w)
                            path_angles = viz_obj['path_angles']
                            cost_maps = viz_obj['cost_maps']
                            if cost_maps is not None: cost_maps = cost_maps.view(-1, 1, h, w)

                            if viz_obj['histories'] is not None \
                                and config.dataset_name in ['mpd', 'tmpd', 'street', 'aug_tmpd']:
                                viz_hists = viz_obj['histories'].view(-1, 1, h, w)
                            else:
                                viz_hists = None

                            # Keep all inputs as batch*c*h*w
                            save_path = f"{save_dir_idx}/{viz_method}_mixed.png"

                            # Expected is in the original map, the obstacles are 0 value
                            # to be excluded from path; but aug_tmpd is inverse,
                            # so change it for visualization
                            if config.dataset_name in ['aug_tmpd']:
                                viz_map_per = 1 - viz_maps[inner_idx:inner_idx+1]

                                # For enable_diag_astar, the cost maps is actually prob maps,
                                # so change it for visualization
                                if cost_maps is not None \
                                    and config.enable_transpath \
                                    and config.transpath.enable_diag_astar \
                                    and viz_method in ['nAstar', 'vAstar']:
                                    cost_maps = 1 - cost_maps
                            else:
                                viz_map_per = viz_maps[inner_idx:inner_idx+1]

                            save_figure_mixed(
                                viz_map_per,
                                viz_starts[inner_idx:inner_idx+1],
                                viz_goals[inner_idx:inner_idx+1],
                                save_path,
                                path=viz_paths[inner_idx:inner_idx+1],
                                path_angle=path_angles[inner_idx],
                                hist=viz_hists[inner_idx:inner_idx+1] if viz_hists is not None else None,
                                cost_map=cost_maps[inner_idx:inner_idx+1] if cost_maps is not None else None,
                                is_reference=viz_method=='GT'
                            )

                sample_count += num_samples

        # Get samples and solutions (from python_motion_planning), use solution as ground truth.
        if len(planner.val_set["solutions"]) > 0:
            planner.save_data_solution(planner.val_set, save_dir=save_dir)

        if len(output_list) == 0:
            return config, None, None

        info_dict = planner.gather_valid_step_to_epoch(output_list)
        all_info_dict.update({method: info_dict})

        # Compute the metrics over the entire valid set
        metrics_dict = {}

        if info_dict['metrics'] is not None:
            for s_key in info_dict['metrics'].keys():
                values = info_dict['metrics'][s_key]
                if len(values) == 0 or isinstance(values, list): continue
                values = values.double()
                mean_v = values.cpu().numpy().mean()
                metrics_dict.update({f'mean_{s_key}': mean_v})

        all_info_dict[method].update({'mean_metrics': metrics_dict})

    del output_list, neural_astars

    return config, all_info_dict

# =============================================================================================================

def compare_two_vectors(
    v_src,
    v_tar
):
    assert v_src.shape[0] == v_tar.shape[0]
    equal = (v_tar == v_src).float().sum()
    gre = (v_tar > v_src).float().sum()
    less = (v_tar < v_src).float().sum()

    return {'less': less.cpu(), 'equal': equal.cpu(), 'gre': gre.cpu()}

# =============================================================================================================

def load_from_ptl_checkpoint(checkpoint_path):
    print(f"Resume from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    state_dict_extracted = dict()

    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted

# =============================================================================================================

class PlannerModule(pl.LightningModule):
    def __init__(
        self,
        planner,
        config,
        wandb_logger=None
    ):
        super().__init__()
        # While aug_tmpd also support vanilla a star, it is too slow, so igore it here
        vastar_available_list = ['street', 'tmpd', 'mpd']

        self.planner = planner
        self.vanilla_astar = VanillaAstar(config)
        self.config = config
        self.wandb_logger = wandb_logger
        self.global_direction_weight = []
        self.EPS = 1e-10
        self.support_vastar = config.dataset_name in vastar_available_list
        self.loss_fnc = nn.L1Loss()

        # To store ground truth paths generated by python_motion_planning.
        # Basically for mpd, tmpd, street because empirical dijkstra policy
        # from NeuralA* repo has zig-zag.
        # This will store with epoch, so would be a bit slow.
        self.train_set = {'inputs': [], 'solutions': []}
        self.val_set = {'inputs': [], 'solutions': []}

        # Save time by avoiding runing vanilla a star
        if config.dataset_name in ['aug_tmpd']:
            self.path_loss_weight = config.transpath.path_loss_weight
            self.enable_mask = config.transpath.mask
        else:
            self.path_loss_weight = 1.0
            self.enable_mask = False

        # Not run vanilla A* on aug_tmpd, too slow.
        if self.support_vastar is False and config.dataset_name != "aug_tmpd":
            print(f'!!! {config.dataset_name} does not support vanilla A*.')

        if config.dataset_name in ['aug_tmpd'] and not config.enable_motion_planning_lib:
            self.recon_criterion = nn.L1Loss() if config.transpath.mode == 'h' else nn.MSELoss()
        else:
            self.recon_criterion = None

        self.automatic_optimization = False

    def forward(
        self,
        map_designs,
        start_maps,
        goal_maps,
        store_hist_coordinates=False,
        prob_maps=None,
        disable_compute_path_angle=False
    ):
        return self.planner(
            map_designs,
            start_maps,
            goal_maps,
            store_hist_coordinates=store_hist_coordinates,
            prob_maps=prob_maps,
            disable_compute_path_angle=disable_compute_path_angle
        )

    def configure_optimizers(self):
        lr = self.config.params.lr

        return torch.optim.RMSprop(self.planner.parameters(), lr)

    def save_data_solution(
        self,
        data_set,
        save_dir=None,
        epoch=None
    ):
        maps = np.concatenate([v["maps"] for v in data_set["inputs"]], axis=0)
        starts = np.concatenate([v["starts"] for v in data_set["inputs"]], axis=0)
        goals = np.concatenate([v["goals"] for v in data_set["inputs"]], axis=0)
        previous_solutions = np.concatenate([v["optimal_paths"] for v in data_set["inputs"]], axis=0)
        solutions = np.concatenate(data_set["solutions"], axis=0)

        starts = np.bool_(starts)
        goals = np.bool_(goals)
        previous_solutions = np.bool_(previous_solutions)
        solutions = np.bool_(solutions)

        if save_dir is not None and save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

            # Save data
            save_path = os.path.join(save_dir, "data.npz" if epoch is None else f"data_epoch{epoch}.npz")
            np.savez(
                save_path,
                map_designs=maps,
                starts=starts,
                goals=goals,
                opt_trajs=previous_solutions
            )

            # Save solution
            method = self.config.motion_planning_lib.method
            save_path = os.path.join(
                save_dir,
                f"{method}.npy" if epoch is None else f"{method}_epoch{epoch}.npy"
            )
            np.save(save_path, solutions)

    def append_data_solution_batch(
        self,
        data,
        solutions,
        target
    ):
        # Store samples and solutions if necessary
        for key in data.keys():
            data[key] = data[key].detach().cpu().numpy()

        solutions = solutions.view(data["starts"].shape).detach().cpu().numpy()

        target['inputs'].append(data)
        target['solutions'].append(solutions)

    def training_step(
        self,
        train_batch,
        batch_idx
    ):
        map_designs = train_batch['maps']
        start_maps = train_batch['starts']
        goal_maps = train_batch['goals']
        opt_trajs = train_batch['optimal_paths']
        num_starts = start_maps.shape[1]

        # Merge the augmented start nodes, num_starts_[valid|test], to batch
        if num_starts > 1:
            height, width = start_maps.shape[-2:]
            map_designs = map_designs.view(-1, 1, height, width)
            start_maps = start_maps.view(-1, 1, height, width)
            goal_maps = goal_maps.view(-1, 1, height, width)
            opt_trajs = opt_trajs.view(-1, 1, height, width)

        # Ignore invalid opt_trajs samples
        valid_sample_indices = opt_trajs.sum((1, 2, 3)) > 2
        map_designs = map_designs[valid_sample_indices]
        start_maps = start_maps[valid_sample_indices]
        goal_maps = goal_maps[valid_sample_indices]
        opt_trajs = opt_trajs[valid_sample_indices]
        num_samples = map_designs.shape[0]

        if self.config.dataset_name in ['warcraft', 'pkmn', 'aug_tmpd'] \
            and self.config.transpath.enable_gt_ppm:
            prob_maps = train_batch['ppm']
        else:
            prob_maps = None

        outputs, cost_maps = self.forward(
            map_designs,
            start_maps,
            goal_maps,
            prob_maps=prob_maps,
        )

        if self.config.enable_motion_planning_lib:
            return map_designs.new_zeros(1)

        gt_cost_maps = None
        cost_map_loss = 0
        loss = 0

        if self.config.dataset_name in ['aug_tmpd']:
            gt_cost_maps = train_batch['ppm']

            if self.enable_mask:
                # It is actually PPM, and this is to exclude high prob areas
                # such that they can be learned through path loss
                cost_map_mask = cost_maps < 0.5  # only consider the less prob areas
                cost_map_loss = self.recon_criterion(
                    cost_maps[cost_map_mask],
                    gt_cost_maps[cost_map_mask]
                )
            else:
                cost_map_loss = self.recon_criterion(cost_maps, gt_cost_maps)
        elif self.config.dataset_name in ['warcraft', 'pkmn']:
            gt_cost_maps = train_batch['ppm']

        path_histories = outputs.histories

        if self.config.dataset_name in ['aug_tmpd']:
            if self.config.transpath.loss_mode.find('path') > -1:
                path_loss = self.loss_fnc(path_histories, opt_trajs)
                path_loss = self.path_loss_weight * path_loss
                loss += path_loss
                self.log("train/path_loss", path_loss)
                self.log("train/path_loss_epoch", path_loss, on_step=False, on_epoch=True)
            if self.config.transpath.loss_mode.find('heat') > -1:
                loss += cost_map_loss
                self.log("train/costmap_loss", cost_map_loss)
                self.log("train/costmap_loss_epoch", cost_map_loss, on_step=False, on_epoch=True)
        else:
            path_loss = self.loss_fnc(path_histories, opt_trajs)
            loss += path_loss
            self.log("train/path_loss", path_loss)
            self.log("train/path_loss_epoch", path_loss, on_step=False, on_epoch=True)

        # Loss
        self.optimizers().zero_grad()
        loss.backward()
        self.optimizers().step()

        if self.config.dataset_name in ['aug_tmpd']:
            self.lr_schedulers()

        self.log("train/loss", loss)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True)

        with torch.no_grad():
            traj_diff = 0.5 * (opt_trajs - outputs.paths).abs().view(num_samples, -1).sum(dim=1) \
                / opt_trajs.view(num_samples, -1).sum(dim=1)
            traj_diff = torch.clamp(traj_diff, min=0, max=1)
            traj_sim = 1 - traj_diff
            traj_sim = traj_sim.mean()

        self.log('train/traj_sim', traj_sim, on_step=False, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(
        self,
        val_batch,
        batch_idx,
        store_hist_coordinates=False,
        enable_save_path_stack=False,
        disable_compute_path_angle=False
    ):
        metrics_dict = {
            'batch': batch_idx,
            'GT': None,
            'nAstar': None,
            'vAstar': None,
            'metrics': None,
            'loss': None
        }

        map_designs = val_batch['maps']
        start_maps = val_batch['starts']
        goal_maps = val_batch['goals']
        opt_trajs = val_batch['optimal_paths']
        num_starts = start_maps.shape[1]
        height, width = start_maps.shape[-2:]

        # Merge the augmented start nodes, num_starts_[valid|test], to batch
        if num_starts > 1:
            map_designs = map_designs.view(-1, 1, height, width)
            start_maps = start_maps.view(-1, 1, height, width)
            goal_maps = goal_maps.view(-1, 1, height, width)
            opt_trajs = opt_trajs.view(-1, 1, height, width)

        # Ignore invalid opt_trajs samples
        valid_sample_indices = opt_trajs.sum((1, 2, 3)) > 2
        map_designs = map_designs[valid_sample_indices]
        start_maps = start_maps[valid_sample_indices]
        goal_maps = goal_maps[valid_sample_indices]
        opt_trajs = opt_trajs[valid_sample_indices]

        num_samples = start_maps.shape[0]
        num_elements = height * width

        # Run vanilla A star
        va_hist_ratio = torch.tensor([-1], dtype=torch.float32, device=map_designs.device)

        if self.support_vastar:
            va_outputs, _ = self.vanilla_astar(
                map_designs,
                start_maps,
                goal_maps,
                store_hist_coordinates=store_hist_coordinates,
                disable_compute_path_angle=disable_compute_path_angle
            )

            if hasattr(va_outputs, 'hist_coordinates'):
                va_hist_coordinates = va_outputs.hist_coordinates
                if va_hist_coordinates is not None: va_hist_coordinates = va_hist_coordinates.cpu().float()
            else:  # TransPath's planner's DifferentiableDiagAstar does not have this
                va_hist_coordinates = None

            va_num_hists = va_outputs.histories.view(num_samples, -1).sum(1)
            va_hist_ratio = va_num_hists / num_elements

            metrics_dict['vAstar'] = {
                'num_steps': va_outputs.paths.view(num_samples, -1).sum(1).cpu().float(),
                'num_hists': va_num_hists.cpu().float(),
                'hist_ratio': va_hist_ratio.cpu().float(),
                'path_angles': va_outputs.path_angles.cpu().float(),
                'histories': va_outputs.histories.cpu().float(),
                'hist_coordinates': va_hist_coordinates,
                'paths': va_outputs.paths.cpu().float(),
                'cost_maps': va_outputs.cost_maps.cpu().float()
            }

        # Run Neural A star
        if self.config.transpath.enable_gt_ppm:
            prob_maps = val_batch['ppm']
        else:
            prob_maps = None

        outputs, cost_maps = self.forward(
            map_designs,
            start_maps,
            goal_maps,
            store_hist_coordinates=store_hist_coordinates,
            prob_maps=prob_maps,
            disable_compute_path_angle=disable_compute_path_angle
        )

        if hasattr(outputs, 'hist_coordinates'):
            na_hist_coordinates = outputs.hist_coordinates
        else:
            na_hist_coordinates = None

        na_num_hists = outputs.histories.view(num_samples, -1).sum(1)
        na_hist_ratio = na_num_hists / num_elements

        metrics_dict['nAstar'] = {
            'num_steps': outputs.paths.view(num_samples, -1).sum(1).cpu().float(),
            'num_hists': na_num_hists.cpu().float(),
            'hist_ratio': na_hist_ratio,
            'path_angles': outputs.path_angles.cpu().float(),
            'histories': outputs.histories.cpu().float(),
            'hist_coordinates': na_hist_coordinates.cpu().float() if na_hist_coordinates is not None else None,
            'paths': outputs.paths.cpu().float(),
            'cost_maps': outputs.cost_maps.cpu().float()
        }

        # Get losses
        gt_cost_maps = None

        if not self.config.enable_motion_planning_lib:
            loss = 0
            cost_map_loss = 0

            if self.config.dataset_name in ['aug_tmpd']:
                # This cost_maps are actually PPM
                gt_prob_maps = val_batch['ppm']
                pred_prob_maps = cost_maps
                gt_cost_maps = 1 - gt_prob_maps

                if self.enable_mask:
                    # It is actually PPM, and this is to exclude high prob areas
                    # such that they can be learned through path loss
                    cost_map_mask = cost_maps < 0.5  # only consider the less prob areas
                    cost_map_loss = self.recon_criterion(
                        pred_prob_maps[cost_map_mask],
                        gt_prob_maps[cost_map_mask]
                    )
                else:
                    cost_map_loss = self.recon_criterion(pred_prob_maps, gt_prob_maps)
            elif self.config.dataset_name in ['warcraft', 'pkmn']:
                gt_cost_maps = val_batch['ppm']

            # Calculate loss for learning optimization
            predictions = outputs.histories

            if self.config.dataset_name in ['aug_tmpd']:
                if self.config.transpath.loss_mode.find('path') > -1:
                    path_loss = self.loss_fnc(predictions, opt_trajs)
                    path_loss = self.path_loss_weight * path_loss
                    loss += path_loss
                    self.log("valid/path_loss", path_loss, on_step=False, on_epoch=True)

                if self.config.transpath.loss_mode.find('heat') > -1:
                    loss += cost_map_loss
                    self.log("valid/costmap_loss", cost_map_loss, on_step=False, on_epoch=True)
            else:
                path_loss = self.loss_fnc(predictions, opt_trajs)
                loss += path_loss
                self.log("valid/path_loss", path_loss, on_step=False, on_epoch=True)

            metrics_dict['loss'] = loss
            self.log("valid/loss", loss, on_step=False, on_epoch=True)

        # Get GT
        gt_path_stack = generate_path_stack(opt_trajs, goal_maps)
        gt_path_angles = cal_path_angle(gt_path_stack)

        metrics_dict['GT'] = {
            'num_steps': opt_trajs.view(num_samples, -1).sum(1).cpu().float(),
            'path_angles': gt_path_angles.cpu().float(),
            'histories': None,
            'hist_coordinates': None,
            'paths': opt_trajs.cpu().float(),
            'cost_maps': gt_cost_maps.cpu().float() if gt_cost_maps is not None else None
        }

        # Calculate num_steps and path_angles as evaluation metrics
        if enable_save_path_stack:
            metrics_dict['nAstar'].update({'path_stack': outputs.path_stack.cpu().float()})
            metrics_dict['GT'].update({'path_stack': gt_path_stack.cpu().float()})

            if self.support_vastar:
                metrics_dict['vAstar'].update({
                    'path_stack': va_outputs.path_stack.cpu().float()
                })

        # Warcraft does not go here because the input to vanilla_astar() is 3*96*96
        # but the start_maps is 1*12*12 for 144 grids.
        # Need tensor format for batche-wise concatenation of metrics_dict['metrics]
        pathlen_gt = opt_trajs.view(num_samples, -1).sum(1)
        pathlen_nastar = outputs.paths.view(num_samples, -1).sum(1)
        p_opt = (pathlen_gt >= pathlen_nastar).float()

        exp_nastar = outputs.histories.view(num_samples, -1).sum(1)

        angle_gt = gt_path_angles
        angle_nastar = outputs.path_angles
        angle_exp_nastar = torch.clamp((angle_gt - angle_nastar) / (angle_gt + self.EPS), min=0, max=1)

        traj_diff = 0.5 * (opt_trajs - outputs.paths).abs().view(num_samples, -1).sum(dim=1) \
            / opt_trajs.view(num_samples, -1).sum(dim=1)
        traj_diff = torch.clamp(traj_diff, min=0, max=1)
        traj_sim = 1 - traj_diff
        c_distance = chamfer_distance(outputs.paths, opt_trajs)

        # SIM metric, very slow, numpy based, not batch wise
        if self.config.enable_resume:  # only apply to test phase
            if hasattr(outputs, 'path_stack'):
                na_path_stack = outputs.path_stack
            else:
                na_path_stack = None

            grid_size = (height - 1) * (width - 1)
            area_sim = calculate_area_sim(gt_path_stack, na_path_stack, grid_size)
        else:
            area_sim = torch.tensor([-1], dtype=torch.float32, device=map_designs.device)

        metrics_dict['metrics'] = {
            'p_opt_nastar': p_opt.cpu().float(),
            'angle_exp_nastar': angle_exp_nastar.cpu().float(),
            'num_samples': num_samples,
            'traj_sim': traj_sim.cpu().float(),
            'area_sim': area_sim.cpu().float(),
            'c_distance': c_distance.cpu().float(),
            'hist_ratio_nastar': na_hist_ratio.cpu().float(),
            'hist_ratio_vastar': va_hist_ratio.cpu().float()
        }

        self.log("valid/p_opt_nastar", p_opt.mean().item(), on_step=False, on_epoch=True)
        self.log("valid/p_angle", angle_exp_nastar.mean().item(), on_step=False, on_epoch=True)
        self.log("valid/traj_sim", traj_sim.mean().item(), on_step=False, on_epoch=True)
        self.log("valid/area_sim", area_sim.mean().item(), on_step=False, on_epoch=True)
        self.log("valid/c_distance", c_distance.mean().item(), on_step=False, on_epoch=True)
        self.log("valid/hist_ratio_nastar", na_hist_ratio.mean().item(), on_step=False, on_epoch=True)

        if self.support_vastar:
            pathlen_vastar = va_outputs.paths.view(num_samples, -1).sum(1)
            p_opt_vastar = (pathlen_gt >= pathlen_vastar).float()

            exp_vastar = va_outputs.histories.view(num_samples, -1).sum(1)
            p_exp = torch.clamp((exp_vastar - exp_nastar) / (exp_vastar + self.EPS), min=0, max=1)

            # Just follow the neural A star paper, but this is not accurate, use the *.ipynb one
            h_mean_list = [p_opt, traj_sim, p_exp]
            metrics_dict['metrics'].update({'p_exp': p_exp.cpu().float()})

            self.log("valid/p_opt_vastar", p_opt_vastar.mean().item(), on_step=False, on_epoch=True)
            self.log("valid/p_exp", p_exp.mean().item(), on_step=False, on_epoch=True)
            self.log("valid/hist_ratio_vastar", va_hist_ratio.mean().item(), on_step=False, on_epoch=True)
        else:  # SDD, warcraft, pkmn
            # Just follow the neural A star paper, but this is not accurate, use the *.ipynb one
            if self.config.dataset_name == 'aug_tmpd':
                h_mean_list = [traj_sim, 1 - na_hist_ratio, p_opt]
            else:
                h_mean_list = [traj_sim, 1 - na_hist_ratio]

        h_mean = len(h_mean_list) / (1.0 / (torch.stack(h_mean_list, dim=0) + self.EPS)).sum(0)
        metrics_dict['metrics'].update({'h_mean': h_mean.cpu().float()})

        self.log("valid/h_mean", h_mean.mean().item(), on_step=False, on_epoch=True)

        return metrics_dict

    # This will be called externally
    def gather_valid_step_to_epoch(
        self,
        step_outputs
    ):
        keys = [
            'num_steps',
            'path_angles',
            'histories',
            'hist_coordinates',
            'path_stack',
            'num_hists',
            'hist_ratio'
        ]
        metrics_keys = [
            'p_opt_nastar',
            'p_exp',
            'h_mean',
            'c_distance',
            'traj_sim',
            'area_sim',
            'hist_ratio_nastar',
            'hist_ratio_vastar'
        ]
        info_dict = {
            'batch': [],
            'GT': {},
            'nAstar': {},
            'vAstar': {},
            'metrics': {}
        }

        for key_per in keys:
            info_dict['GT'].update({key_per: []})
            info_dict['nAstar'].update({key_per: []})
            info_dict['vAstar'].update({key_per: []})

        for key_per in metrics_keys:
            info_dict['metrics'].update({key_per: []})

        for data_per in step_outputs:
            info_dict['batch'].append(data_per['batch'])

            for key_per in keys:
                for key_src_per in ['GT', 'nAstar', 'vAstar']:
                    if data_per[key_src_per] is not None:
                        if key_per in data_per[key_src_per].keys():
                            info_dict[key_src_per][key_per].append(data_per[key_src_per][key_per])
                    else:
                        info_dict[key_src_per] = None

            if data_per['metrics'] is not None:
                for key_per in metrics_keys:
                    if key_per in data_per['metrics'].keys():
                        info_dict['metrics'][key_per].append(data_per['metrics'][key_per])
            else:
                info_dict['metrics'] = None

        for key in info_dict.keys():
            if (key in ['GT', 'nAstar', 'vAstar', 'metrics']) and (info_dict[key] is not None):
                for s_key in info_dict[key].keys():
                    v_list = info_dict[key][s_key]

                    # (num search steps, batch, 3), num search steps is different for each batch,
                    # so do not concatenate them.
                    if s_key != 'hist_coordinates':
                        if (len(v_list) > 0) and (v_list[0] is not None):
                            info_dict[key][s_key] = torch.concat(info_dict[key][s_key], dim=0)

        return info_dict

    # This will not work for pytorch-lightning >=2.0.0
    @torch.no_grad()
    def validation_epoch_end(self, validation_step_outputs):
        # Visual check g_ratio and rotation_const
        viz_obj = self.planner.astar
        g_ratio_value = viz_obj.get_g_ratio()
        rotation_const_value = viz_obj.get_rotation_const()
        rotation_weight_value = viz_obj.get_rotation_weight()

        if torch.is_tensor(g_ratio_value): g_ratio_value = g_ratio_value.item()
        if torch.is_tensor(rotation_const_value): rotation_const_value = rotation_const_value.item()
        if torch.is_tensor(rotation_weight_value): rotation_weight_value = rotation_weight_value.item()

        print(
            f'\n===> Check, train g ratio: {self.config.enable_train_g_ratio}' \
            f', value: {g_ratio_value:.4f}.'
        )
        print(
            f'===> Check, train rotation const: {self.config.enable_train_rotation_const}' \
            f', value: {rotation_const_value:.4f}.'
        )
        print(
            f'===> Check, train rotation weight: {self.config.enable_train_rotation_const}' \
            f', value: {rotation_weight_value:.4f}.\n'
        )

        self.log(f"valid/g_ratio", g_ratio_value, on_step=False, on_epoch=True)
        self.log(f"valid/rotation_const", rotation_const_value, on_step=False, on_epoch=True)
        self.log(f"valid/rotation_weight", rotation_weight_value, on_step=False, on_epoch=True)

# =============================================================================================================

def set_global_seeds(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
