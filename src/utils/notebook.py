# -------------------------------------------------------------------------------------------------------------
# File: notebook.py
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

import os, torch, csv, sys
import numpy as np
import pandas as pd

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from copy import deepcopy
from src.utils.training import (
    statistics_all_methods,
    load_from_ptl_checkpoint,
    set_global_seeds
)
from src.planner import NeuralAstar, MotionPlanning
from src.utils.data import create_dataloader, create_game_dataloader
from glob import glob
from torch.utils.data import DataLoader
from third_party.transpath.data.hmaps import GridData

# =============================================================================================================

def save_csv(
    csv_file_path,
    keys,
    data
):
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*data): writer.writerow(row)
        f.close()

# =============================================================================================================

def read_csv_and_sort(
    csv_file_path,
    sort_key,
    topk=30
):
    df = pd.read_csv(csv_file_path)

    if sort_key in df.keys():
        metric_v = np.array(df[sort_key])
        if sort_key in ['c_distance']:
            save_image_indices = np.argsort(metric_v)[:topk]  # ascending
        else:
            save_image_indices = np.argsort(metric_v)[::-1][:topk]  # descending
    else:
        save_image_indices = []
        print(f'!!! Error, sort key: {sort_key} not exist.')

    return save_image_indices

# =============================================================================================================

def construct_a_star(
    config,
    return_resume_path=False
):
    device = 'cuda'
    config_modified = deepcopy(config)

    if config.enable_motion_planning_lib:
        motion_planner = MotionPlanning(config_modified)

        if return_resume_path:
            return motion_planner, None
        else:
            return motion_planner
    else:
        neural_astar = NeuralAstar(config_modified).to(device)

        if config_modified.dataset_name == 'aug_tmpd' \
            and config_modified.enable_transpath \
            and config_modified.transpath.pretrained_model_path is not None \
            and os.path.exists(config_modified.transpath.pretrained_model_path):
            ckpt_file = config_modified.transpath.pretrained_model_path

            neural_astar.load_transpath_pretrained_model(ckpt_file)
            print(f'!!! Load pretrained model from {ckpt_file}.')
        else:
            if config.enable_resume:
                ckpt_files = search_checkpoints(config_modified)
                if len(ckpt_files) >= 1: ckpt_file = ckpt_files[-1]
                else: ckpt_file = None
            else:
                ckpt_file = None

            assert ckpt_file is not None, f'!!!Error, enable resume but checkpoint not exist: {config.resume_path_dir}'

            neural_astar.load_state_dict(load_from_ptl_checkpoint(ckpt_file), strict=False)
            current_g_ratio = neural_astar.astar.get_g_ratio()
            current_rotation_const = neural_astar.astar.get_rotation_const()
            if isinstance(current_g_ratio, torch.Tensor): current_g_ratio = current_g_ratio.item()
            if isinstance(current_rotation_const, torch.Tensor): current_rotation_const = current_rotation_const.item()
            print(f'Model G ratio: {current_g_ratio:.4f}.')
            print(f'Model rotation const: {current_rotation_const:.4f}.')

        if return_resume_path:
            return neural_astar, ckpt_file
        else:
            return neural_astar

# =============================================================================================================

def parse_folder_path(config):
    if config.dataset_file_name is None:
        dataset_file_names = get_dataset_file_names(config)
        config.dataset_file_name = dataset_file_names[0]

    folder_path = f'{config.dataset_name}/{config.dataset_file_name}'

    if config.dataset_name.find("sdd") > -1:
        if 'test_scene' in config.sdd:
            folder_path += f'/{config.sdd.test_scene}'
        else:
            folder_path += f'/{config.sdd.test_scene_dict[config.sdd.test_scene_type][0]}'

    folder_path += f'/{config.encoder.arch}/Enconst{config.encoder.const:.3f}'

    return folder_path

# =============================================================================================================

def get_dataloaders(
    config,
    split=None,
    enable_train_shuffle=None,
    disable_random_start=False
):
    data_loader = None
    num_workers = 2

    if enable_train_shuffle is None:
        enable_train_shuffle = True

    if config.dataset_name == 'aug_tmpd':
        num_workers = 4
        dataset_dir = config.dataset_dir + '/aug_tmpd'

        if split is None or split == 'train':
            trainset = GridData(
                f'{dataset_dir}/train',
                mode=config.transpath.mode,
                split='train'
            )

            train_loader = DataLoader(
                trainset,
                batch_size=config.params.batch_size,
                shuffle=enable_train_shuffle,
                num_workers=num_workers,
                pin_memory=False
            )

            data_loader = train_loader

        if split is None or split == 'val':
            valset = GridData(
                f'{dataset_dir}/val',
                mode=config.transpath.mode,
                split='val'
            )

            val_loader = DataLoader(
                valset,
                batch_size=config.params.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            data_loader = val_loader

        if split is None or split == 'test':
            testset = GridData(
                f'{dataset_dir}/test',
                mode=config.transpath.mode,
                split='test'
            )

            test_loader = DataLoader(
                testset,
                batch_size=config.params.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            data_loader = test_loader
    elif config.dataset_name in ['warcraft', 'pkmn']:
        dataset_dir = f'{config.dataset_dir}/{config.dataset_name}'
        train_loader = create_game_dataloader(
            dataset_dir,
            'train',
            config.params.batch_size,
            enable_train_shuffle
        )

        val_loader = create_game_dataloader(
            dataset_dir,
            'val',
            config.params.batch_size
        )

        test_loader = create_game_dataloader(
            dataset_dir,
            'test',
            config.params.batch_size
        )

        if split is not None and data_loader is None:
            if split == 'train':
                data_loader = train_loader
            elif split == 'val':
                data_loader = val_loader
            elif split == 'test':
                data_loader = test_loader
    else:
        dataset_dir = os.path.join(config.dataset_dir, config.dataset_name.split("_")[0])

        if config.dataset_name.find('sdd') > -1:
            dataset_file_path = os.path.join(
                dataset_dir,
                config.dataset_file_name
            )

            if 'test_scene' in config.sdd:
                test_scene = config.sdd.test_scene
            else:
                test_scene = config.sdd.test_scene_dict[config.sdd.test_scene_type][0]

            hardness_threshold = config.sdd.hardness_threshold
        else:
            dataset_file_path = os.path.join(
                dataset_dir,
                config.dataset_file_name
            ) + '.npz'
            test_scene = None
            hardness_threshold = None

        # Use config.motion_planning_lib.method for training, but empirical policy with zig-zag.
        train_loader = create_dataloader(
            dataset_file_path,
            "train",
            config.params.batch_size,
            eval_seed=config.eval_seed,
            shuffle=enable_train_shuffle,
            num_starts=config.num_starts_train,
            test_scene=test_scene,
            hardness_threshold=hardness_threshold,
            disable_random_start=disable_random_start
        )

        val_loader = create_dataloader(
            dataset_file_path,
            "valid",
            config.params.batch_size,
            eval_seed=config.eval_seed,
            shuffle=False,
            num_starts=config.num_starts_valid,
            test_scene=test_scene,
            disable_random_start=disable_random_start
        )

        test_loader = create_dataloader(
            dataset_file_path,
            "test",
            config.params.batch_size,
            eval_seed=config.eval_seed,
            shuffle=False,
            num_starts=config.num_starts_test,
            test_scene=test_scene,
            disable_random_start=disable_random_start
        )

        if split is not None and data_loader is None:
            if split == 'train':
                data_loader = train_loader
            elif split == 'val':
                data_loader = val_loader
            elif split == 'test':
                data_loader = test_loader

    if split is None:
        return train_loader, val_loader, test_loader
    else:
        return data_loader

# =============================================================================================================

def search_checkpoints(config):
    if config.enable_transpath and os.path.exists(config.transpath.pretrained_model_path):
        checkpoint_list = [config.transpath.pretrained_model_path]
    else:
        if os.path.isdir(config.resume_path_dir):
            checkpoint_path = config.resume_path_dir
            checkpoint_path += f'/{parse_folder_path(config)}'
            checkpoint_list = sorted(glob(
                f"{checkpoint_path}_pid*/**/checkpoints/*.ckpt",
                recursive=True
            ))
        elif os.path.isfile(config.resume_path_dir):
            checkpoint_list = [config.resume_path_dir]
        else:
            print(f'!!!Warning, {config.resume_path_dir} should be a directory.')
            checkpoint_list = []

    return checkpoint_list

# =============================================================================================================

def get_dataset_file_names(config):
    if config.dataset_file_name is None:
        if config.dataset_name == 'mpd':
            dataset_file_names = [
                'alternating_gaps_032_moore_c8',
                'bugtrap_forest_032_moore_c8',
                'forest_032_moore_c8',
                'gaps_and_forest_032_moore_c8',
                'mazes_032_moore_c8',
                'multiple_bugtraps_032_moore_c8',
                'shifting_gaps_032_moore_c8',
                'single_bugtrap_032_moore_c8'
            ]
        elif config.dataset_name == 'tmpd':
            dataset_file_names = ['all_064_moore_c16']
        elif config.dataset_name.find('sdd') > -1:
            dataset_file_names = ['s064_0.5_128_300']
        elif config.dataset_name == 'street':
            dataset_file_names = ['mixed_064_moore_c16']
        elif config.dataset_name in ['aug_tmpd']:
            dataset_file_names = ['transpath']
        elif config.dataset_name in ['warcraft', 'pkmn']:
            dataset_file_names = ['game']
        else:
            assert False, f'!!!Error, invalid dataset name:{config.dataset_name}.'
    else:
        dataset_file_names = [config.dataset_file_name]

    return dataset_file_names

# =============================================================================================================

def run_statistics_all_methods(
    config,
    dataset_names,
    split,
    g_ratio_list,
    enable_save_path_stack=False,
    enable_save_image=False,
    save_image_list=None,
    disable_compute_path_angle=False,
    disable_random_start=False
):
    encoder_archs = [config.encoder.arch]
    encoder_arch = config.encoder.arch
    device = 'cuda'
    store_hist_coordinates = True

    if config.enable_motion_planning_lib:
        save_dir = config.motion_planning_lib.save_dir
        file_name_key = f"average_{config.motion_planning_lib.method}"
    else:
        save_dir = config.logdir
        file_name_key = "average"

    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in dataset_names:
        # Get dataset files
        dataset_file_names = get_dataset_file_names(config)

        # For mpd with many datasets
        mean_dict = {
            'mean_p_opt_nastar': [],
            'mean_p_exp': [],
            'mean_h_mean': [],
            'mean_loss': [],
            'mean_traj_sim': [],
            'mean_area_sim': [],
            'mean_c_distance': [],
            'mean_g_ratio': [],
            'mean_rotation_const': [],
            'mean_rotation_weight': [],
            'mean_hist_ratio_nastar': [],
            'mean_hist_ratio_vastar': []
        }

        # Set test scene for SDD
        if config.dataset_name.find('sdd') > -1:
            if 'test_scene' not in config.sdd or config.sdd.test_scene is None:
                sdd_test_scenes = config.sdd.test_scene_dict[config.sdd.test_scene_type]
            else:
                sdd_test_scenes = [config.sdd.test_scene]
        else:
            sdd_test_scenes = ['']  # put one value to avoid skipping evaluation

        result_string_all = ''

        for g_ratio_per in g_ratio_list:
            for dataset_file_name_per in dataset_file_names:
                # Loop over test_scene_type for SDD
                for test_scene in sdd_test_scenes:
                    config_modified = deepcopy(config)
                    config_modified.encoder.arch = encoder_arch
                    config_modified.dataset_name = dataset_name
                    config_modified.dataset_file_name = dataset_file_name_per
                    config_modified.g_ratio = g_ratio_per
                    config_modified.sdd.test_scene = test_scene

                    # Run over the resume model per dataset_file_name
                    neural_astars = []
                    neural_astar_per, resume_path = construct_a_star(config_modified, return_resume_path=True)
                    neural_astars.append(neural_astar_per)

                    g_ratio_v = neural_astars[-1].astar.get_g_ratio()
                    rotation_const_v = neural_astars[-1].astar.get_rotation_const()
                    rotation_weight_v = neural_astars[-1].astar.get_rotation_weight()
                    if isinstance(g_ratio_v, torch.Tensor): g_ratio_v = g_ratio_v.item()
                    if isinstance(rotation_const_v, torch.Tensor): rotation_const_v = rotation_const_v.item()
                    if isinstance(rotation_weight_v, torch.Tensor): rotation_weight_v = rotation_weight_v.item()

                    mean_dict['mean_g_ratio'].append(g_ratio_v)
                    mean_dict['mean_rotation_const'].append(rotation_const_v)
                    mean_dict['mean_rotation_weight'].append(rotation_weight_v)

                    result_string = f"{config_modified.dataset_name} ({config_modified.dataset_file_name})" \
                        f", resume from {resume_path}.\n" \
                        f"G ratio: {g_ratio_v:.4f}" \
                        f", rotation const: {rotation_const_v:.4f}" \
                        f", rotation weight: {rotation_weight_v:.4f}.\n"

                    # May not be useful as no shuffle here, but just in case
                    set_global_seeds(config_modified.seed)

                    # Motion planning library would be very slow for aug_tmpd, so reduce the size
                    data_loader = get_dataloaders(
                        config_modified,
                        split=split,
                        disable_random_start=disable_random_start
                    )

                    #
                    _, all_info_dict = statistics_all_methods(
                        encoder_archs,
                        neural_astars,
                        data_loader,
                        config_modified,
                        device=device,
                        store_hist_coordinates=store_hist_coordinates,
                        enable_save_path_stack=enable_save_path_stack,
                        resume_path=resume_path,
                        enable_save_image=enable_save_image,
                        split=split,
                        save_image_list=save_image_list,
                        disable_compute_path_angle=disable_compute_path_angle
                    )

                    # Save path_stack coordinates to compute HM over all methods
                    if enable_save_path_stack:
                        if resume_path is None:
                            save_path_stack_dir = f"{config.resume_path_dir}/{config.dataset_name}/{dataset_file_name_per}"
                            if (test_scene is not None) and (test_scene != ""): save_path_stack_dir += f"/{test_scene}"
                            os.makedirs(save_path_stack_dir, exist_ok=True)
                            save_path_stack_path = f"{save_path_stack_dir}/path_stack_HarmonicMean.npy"
                        else:
                            resume_file_name = resume_path.split("/")[-1].replace(".ckpt", "").replace(".pth", "")
                            save_path_stack_path = resume_path.replace(
                                f"checkpoints/{resume_file_name}.ckpt",
                                f"{resume_file_name}_HarmonicMean.npy"
                            )
                            save_path_stack_path = save_path_stack_path.replace(
                                f"{resume_file_name}.pth",
                                f"{resume_file_name}_HarmonicMean.npy"
                            )

                        nastar_path_stack = all_info_dict[config.encoder.arch]['nAstar']['path_stack'].permute(0, 2, 1).cpu().numpy()
                        gt_path_stack = all_info_dict[config.encoder.arch]['GT']['path_stack'].permute(0, 2, 1).cpu().numpy()
                        metrics = all_info_dict[config.encoder.arch]["metrics"]

                        if all_info_dict[config.encoder.arch]['vAstar'] is None:
                            vastar_path_stack = None
                        else:
                            vastar_path_stack = all_info_dict[config.encoder.arch]['vAstar']['path_stack'].permute(0, 2, 1).cpu().numpy()

                        for metric_key in metrics.keys():
                            if isinstance(metrics[metric_key], torch.Tensor):
                                metrics[metric_key] = metrics[metric_key].cpu().numpy()

                        np.save(
                            save_path_stack_path,
                            {
                                'nastar_path_stack': nastar_path_stack,
                                'gt_path_stack': gt_path_stack,
                                'vastar_path_stack': vastar_path_stack,
                                "metrics": all_info_dict[config.encoder.arch]["metrics"]
                            }
                        )

                    # Print mean metrics values to check with the training log
                    for method_name in all_info_dict.keys():
                        current_info_dict = all_info_dict[method_name]['mean_metrics']

                        if current_info_dict is not None:
                            for s_key in current_info_dict.keys():
                                s_value = current_info_dict[s_key]

                                # Mean over H_mean across different datasets
                                mean_dict[s_key].append(s_value)

                                if (config.dataset_name == 'mpd') and (config.dataset_file_name is None):
                                    result_string += f"Method: {method_name}, {s_key} ({dataset_file_name_per}): {s_value:.4f}.\n"
                                elif config.dataset_name.find('sdd') > -1:
                                    result_string += f"Method: {method_name}, {s_key} ({test_scene}): {s_value:.4f}.\n"
                                else:
                                    result_string += f"Method: {method_name}, {s_key}: {s_value:.4f}.\n"

                    # Write to a txt file for each test_scene in SDD or dataset_file_name for MPD
                    if resume_path is None:
                        log_file_dir = f"{config.resume_path_dir}/{config.dataset_name}"
                        os.makedirs(log_file_dir, exist_ok=True)
                        log_file_path = f"{log_file_dir}/log.txt"
                    else:
                        log_file_path = resume_path.replace('ckpt', 'txt').replace('pth', 'txt')

                        if log_file_path.find('/checkpoints') > -1:
                            log_file_path = log_file_path.replace('/checkpoints', '')

                    result_string_all += result_string

                    with open(log_file_path, 'w') as f:
                        f.write(result_string)
                        f.close()

                    print(result_string)

                # Mean over H_mean across different datasets
                if config.dataset_name == 'sdd_inter':
                    result_string = 'Average over all subsets.\n'

                    for key in mean_dict.keys():
                        s_value_list = mean_dict[key]

                        if len(s_value_list) > 0:
                            mean_s_value = np.mean(s_value_list)  # across different datasets for mpd
                            result_string += f'Dataset: {config.dataset_name}, {key}: {mean_s_value:.6f}.\n'

                    print(result_string)

                    # Write to a txt file for the mean values
                    if resume_path is None:
                        log_file_dir = f"{config.resume_path_dir}/{config.dataset_name}"
                        os.makedirs(log_file_dir, exist_ok=True)
                        log_file_path = f"{log_file_dir}/{file_name_key}_{config.expand_mode}.txt"
                    else:
                        save_dir_updated = f'{save_dir}/{config.dataset_name}/{dataset_file_names[0]}'
                        log_file_path = f'{save_dir_updated}/{file_name_key}_{config.expand_mode}.txt'

                    result_string_all += result_string

                    with open(log_file_path, 'w') as f:
                        f.write(result_string_all)
                        f.close()

            # Mean over H_mean across different datasets
            if config.dataset_name == 'mpd' and config.dataset_file_name is None:
                result_string = 'Average over all subsets.\n'

                for key in mean_dict.keys():
                    s_value_list = mean_dict[key]

                    if len(s_value_list) > 0:
                        mean_s_value = np.mean(s_value_list)  # across different datasets for mpd
                        result_string += f'Dataset: {config.dataset_name}, {key}: {mean_s_value:.6f}.\n'

                print(result_string)

                # Write to a txt file for the mean values
                if resume_path is None:
                    log_file_dir = f"{config.resume_path_dir}/{config.dataset_name}"
                    os.makedirs(log_file_dir, exist_ok=True)
                    log_file_path = f"{log_file_dir}/{file_name_key}.txt"
                else:
                    log_file_path = f'{save_dir}/{config.dataset_name}/{file_name_key}_{config.expand_mode}.txt'

                result_string_all += result_string

                with open(log_file_path, 'w') as f:
                    f.write(result_string_all)
                    f.close()