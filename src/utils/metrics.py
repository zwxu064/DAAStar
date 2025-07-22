# -------------------------------------------------------------------------------------------------------------
# File: metrics.py
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

import torch, os, yaml, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from third_party.similarity_measures import similaritymeasures as SM
from box import Box
from PIL import Image
from pytorch3d.loss import chamfer_distance as p3chamfer

# =============================================================================================================

def parse_args(
    config_file_path=None,
    config=None,
    dataset_name=None,
    method=None
):
    assert config_file_path is not None or config is not None

    if config is None:
        config = Box.from_yaml(
            filename=config_file_path,
            Loader=yaml.FullLoader
        )
    else:
        assert config.seed is not None
        assert config.dataset_name is not None

    if dataset_name is not None:
        config.dataset_name = dataset_name

    if method is not None:
        config.method = method

    # NOTE: distribute method and dataset config to some options.
    if config.dataset_name == "sdd_intra":
        config.sdd.test_scene_type = "intra"
    elif config.dataset_name == "sdd_inter":
        config.sdd.test_scene_type = "inter"
    else:
        config.sdd.test_scene_type = ""

    config.num_dirs = 4 if config.dataset_name.find("sdd") > -1 else 8
    config.enable_motion_planning_lib = False
    config.motion_planning_lib.method = ""
    config.enable_transpath = False
    config.enable_inv_rotation = None
    config.expand_mode = "max"
    config.transpath.loss_mode = "path+heat"
    config.transpath.path_loss_weight = 1.0
    config.transpath.mask = False
    config.enable_angle = False
    config.enable_train_rotation_const = False
    config.enable_train_g_ratio = False
    config.transpath.enable_diag_astar = False
    config.transpath.mode = 'f'

    if config.method in ["a_star", "theta_star", "dijkstra"]:
        assert config.dataset_name in ["mpd", "tmpd", "street", "aug_tmpd"]
        config.enable_motion_planning_lib = True
        config.motion_planning_lib.method = config.method
    elif config.method in ["randomwalk_3", "randomwalk_5"]:
        config.expand_mode = config.method
    elif config.method == "daa_min":
        config.enable_inv_rotation = False
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
    elif config.method == "daa_max":
        config.enable_inv_rotation = True
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
    elif config.method == "daa_mix":
        config.enable_inv_rotation = None
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True

        if config.dataset_name == "aug_tmpd":
            config.transpath.loss_mode = "path+heat"
            config.transpath.path_loss_weight = 1.0
    elif config.method == "transpath":
        assert config.dataset_name in ["aug_tmpd"], config.dataset_name
        assert os.path.exists(config.transpath.pretrained_model_path)
        config.enable_transpath = True
        config.transpath.enable_diag_astar = True
    elif config.method == "daa_path":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path"
    elif config.method == "daa_weight":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path+heat"
        config.transpath.path_loss_weight = 10.0
    elif config.method == "daa_mask":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path+heat"
        config.transpath.mask = True
    elif config.method != "neural_astar":
        assert False, f"Unknown method: {config.method}."

    config.loss_type = "l1"
    config.logdir = f'{config.log_root}/seed{config.seed}/{config.method}'
    config.resume_path_dir = f'{config.resume_root}/seed{config.seed}/{config.method}'

    num_epoch_dict = {
        'mpd': 400,
        'tmpd': 400,
        'sdd_intra': 150,
        'sdd_inter': 50,
        'street': 400,
        'aug_tmpd': 50,
        'warcraft': 100,
        'pkmn': 100
    }

    if config.params.batch_size is None: config.params.batch_size = 64
    config.params.num_epochs = num_epoch_dict[config.dataset_name]

    if config.dataset_name in [
        'sdd_inter',
        'sdd_intra',
        'warcraft',
        'pkmn'
    ]:
        config.encoder.input = 'rgb+'
    else:
        config.encoder.input = 'm+'

    if config.dataset_name in ['warcraft', 'pkmn']:
        config.encoder.arch = 'CNNDownSize'
    else:
        config.encoder.arch = 'Unet'

    if config.dataset_name in ['warcraft']:
        # warcraft from 96 to 12
        config.encoder.depth = 3
    else:
        # pkmn from 320 to 20
        config.encoder.depth = 4

    if config.dataset_name in [
        'sdd_inter',
        'sdd_intra',
        'aug_tmpd',
        'warcraft',
        'pkmn']:
        config.encoder.const = 10.0
        config.num_starts_valid = 1
        config.num_starts_test = 1
    else:
        config.encoder.const = 1.0
        config.num_starts_valid = 2
        config.num_starts_test = 5

    config.num_starts_train = 1

    if config.dataset_name in ['sdd_intra', 'sdd_inter']:
        test_scene_inter = [
            'bookstore',
            'coupa',
            'deathCircle',
            'gates',
            'hyang',
            'little',
            'nexus',
            'quad'
        ]
        test_scene_intra = ['video0']

        if not hasattr(config.sdd, 'test_scene_dict'):
            config.sdd.test_scene_dict = {
                'inter': test_scene_inter,
                'intra': test_scene_intra
            }

        if len(config.sdd.test_scene_dict['inter']) == 0:
            config.sdd.test_scene_dict['inter'] = test_scene_inter

        if len(config.sdd.test_scene_dict['intra']) == 0:
            config.sdd.test_scene_dict['intra'] = test_scene_intra

    if config.dataset_name in ['warcraft', 'pkmn']:
        config.params.lr = 3e-4  # 5e-4
    else:
        config.params.lr = 1e-3

    return config

# =============================================================================================================

# This is from the github repository of nA* paper on the icml2021 branch
def chamfer_distance(
    paths,
    traj_images
):
    cd = torch.zeros(len(paths))

    for i in range(len(paths)):
        pts1 = torch.stack(torch.where(paths[i][0])).T.float()
        pts2 = torch.stack(torch.where(traj_images[i][0])).T.float()
        cd[i] = p3chamfer(pts1.unsqueeze(0).cpu(), pts2.unsqueeze(0).cpu())[0]

    return cd

# =============================================================================================================

@torch.no_grad()
def cal_path_angle(path_stack):
    path_stack = path_stack.double()
    num_nodes = path_stack.shape[-1]
    num_samples = path_stack.shape[0]
    path_angles = torch.zeros(num_samples, dtype=torch.float64, device=path_stack.device)

    for idx in range(num_nodes):
        # Only for every 3 nodes, forming 2 vectors to calculate the angle
        if idx + 2 >= num_nodes: continue

        current_loc = path_stack[:, :, idx]
        next_loc = path_stack[:, :, idx + 1]
        next_next_loc = path_stack[:, :, idx + 2]

        # Only when the very remote one is valid, the vectors will be valid
        # For some batches, the path is shorter and the angle caculation will terminate earlier
        valid_indices = (next_next_loc[:, 0] >= 0) * (next_next_loc[:, 1] >= 0)

        if valid_indices.float().sum() == 0:
            break
        else:
            v_1 = (next_loc - current_loc)[valid_indices]
            v_2 = (next_next_loc - next_loc)[valid_indices]

            # To get the angle using unit vectors
            v_1_normed = v_1 / v_1.pow(2).sum(1, keepdim=True).sqrt()
            v_2_normed = v_2 / v_2.pow(2).sum(1, keepdim=True).sqrt()

            cos_v = torch.einsum('bn,bn->b', v_1_normed, v_2_normed)
            angles = torch.arccos(cos_v) * 180 / torch.pi
            path_angles[valid_indices] += angles

    return path_angles

# =============================================================================================================

# This took a while due to the batch-wise implemention while C style will be much easier
def generate_path_stack(
    path_maps,
    goal_maps
):
    path_map_shape = path_maps.shape

    if len(path_map_shape) == 3:
        num_samples, height, width = path_map_shape
    else:
        num_samples, _, height, width = path_map_shape

    num_nodes = height * width

    # 2D
    path_maps = path_maps.view(num_samples, height * width)
    goal_maps = goal_maps.view(num_samples, height * width)

    # 2D
    index_maps = torch.arange(height * width, dtype=torch.long, device=path_maps.device)
    index_maps = index_maps.view(1, height * width).repeat(num_samples, 1)

    # Non optimal paths are set as invalid with -1
    index_maps[path_maps == 0] = -1

    # Coordinates
    path_stack = -torch.ones(num_samples, 2, height * width, dtype=torch.float32, device=path_maps.device)
    current_index = index_maps[goal_maps == 1]

    # Offset with cross first, order matters, 8 neighbours
    neigh_offsets = torch.tensor([
        [-1, 0], [0, -1], [1, 0], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=torch.long, device=path_maps.device
    )
    neigh_offsets = neigh_offsets.view(1, 8, 2)

    # Backtracing from the end points by searching the neighbours
    for idx in range(num_nodes):
        # Calculate h, w to block those outside the image size
        current_index_h = (current_index / width).floor().float()
        current_index_w = (current_index % width).float()

        # Store the current nodes
        path_stack[:, 0, idx] = torch.where(current_index >= 0, current_index_h, -1)
        path_stack[:, 1, idx] = torch.where(current_index >= 0, current_index_w, -1)

        # Get coordinates of the 8 neighbours
        neigh_h = current_index_h.view(-1, 1) + neigh_offsets[:, :, 0]
        neigh_w = current_index_w.view(-1, 1) + neigh_offsets[:, :, 1]
        neigh_indices = neigh_h * width + neigh_w

        # Get the valid indices that within image size
        valid_range = (neigh_h >= 0) * (neigh_h < height) * (neigh_w >= 0) * (neigh_w < width)

        # Set (out of image size) points as invalid with -1
        neigh_indices = torch.where(valid_range, neigh_indices.long(), -1)

        # Set non-path with -1
        neigh_indices = torch.where(valid_range, torch.gather(index_maps, 1, neigh_indices.abs().long()), -1)

        # All neighbours' indices must be >= 0, invalid ones are all -1
        t = (neigh_indices >= 0).float()  # (batch * 8)

        # All batches reach the starting nodes, then stop backtracing
        if t.sum() == 0:
            break
        else:
            # Only get the first valid neighbour for each batch
            neigh_indices = torch.gather(neigh_indices, 1, torch.argmax(t, dim=1).view(-1, 1)).view(-1)

            # Remove the current nodes from the index_maps to avoid being counted as neighbours
            # This is ugly, well
            index_maps[current_index >= 0] = index_maps[current_index >= 0].scatter_(
                1, current_index[current_index >= 0].view(-1, 1), -1
            )

            # Update the current nodes with the closest (the first) neighbours
            current_index = torch.where(t.sum(1) > 0, neigh_indices, -1)

    return path_stack

# =============================================================================================================

def generate_gif(
    key_data,
    hist_index_list,
    save_path,
    image_size
):
    batch_size = key_data[0].shape[0]
    stack_indices = [int(v // batch_size) for v in hist_index_list]
    batch_indices = [int(v % batch_size) for v in hist_index_list]

    for img_idx, stack_index, batch_index in zip(hist_index_list, stack_indices, batch_indices):
        data_per = key_data[stack_index][batch_index, :, 1:].int()  # number of search steps * 2
        num_search_steps = data_per.shape[0]
        img = torch.zeros(num_search_steps, image_size[0], image_size[1], dtype=torch.float32)
        img[torch.arange(num_search_steps), data_per[:, 0], data_per[:, 1]] = 1
        img = torch.cumsum(img, dim=0)
        img = (img > 0).float().cpu().numpy() * 255
        img = [Image.fromarray(v) for v in img]
        save_img_path = save_path.replace('.gif', f'_{img_idx}.gif')
        img[0].save(
            save_img_path, format="GIF", append_images=img,
            save_all=True, duration=50, loop=0
        )

# =============================================================================================================

def viz_statistics(
    file_path_list,
    dataset_name,
    dataset_file_name,
    hist_index_list=None,
    image_size=(64, 64)
):
    key_list = [
        'num_steps',
        'path_angles',
        'path_costs',
        'histories',
        'hist_coordinates'
    ]
    bins = 20

    for key in key_list:
        _, axes = plt.subplots(2, 1)
        hist_list = []
        legend = []
        is_gt_shown = False

        for file_path in file_path_list:
            if os.path.exists(file_path):
                # Create folder to save gif and images
                save_dir = file_path.replace(file_path.split('/')[-1], '')
                os.makedirs(save_dir, exist_ok=True)

                method = file_path.split('/')[-2]
                data = torch.load(file_path)

                # GT
                if (key not in ['histories', 'hist_coordinates']) and (not is_gt_shown):
                    axes[0].plot(data['Unet']['GT'][key])
                    hist_list.append(data['Unet']['GT'][key])
                    legend.append('GT')
                    is_gt_shown = True

                # Vanilla A star
                if data['Unet']['vAstar'] is not None:
                    # As batch * num_search_steps * 3, 3: batch * h * w
                    key_data = data['Unet']['vAstar'][key]

                    if key == 'histories':
                        key_data = key_data.view(key_data.shape[0], -1).sum(1)
                    elif key == 'hist_coordinates':
                        save_path = save_dir + '/vAstar.gif'
                        generate_gif(key_data, hist_index_list, save_path, image_size)
                    else:
                        axes[0].plot(key_data)
                        hist_list.append(key_data)
                        legend.append(f'vAstar({method})')

                # Neural A star
                key_data = data['Unet']['nAstar'][key]

                if key == 'histories':
                    key_data = key_data.view(key_data.shape[0], -1).sum(1)
                elif key == 'hist_coordinates':
                    save_path = save_dir + '/nAstar.gif'
                    generate_gif(key_data, hist_index_list, save_path, image_size)
                else:
                    axes[0].plot(key_data)
                    hist_list.append(key_data)
                    legend.append(f'nAstar({method})')

        if key not in ['histories', 'hist_coordinates']:
            axes[0].set_title(f'{dataset_name}-{dataset_file_name}')
            axes[0].set_xlabel('sample index')
            axes[0].set_ylabel(f"{key.replace('_', ' ')}")
            axes[1].hist(hist_list, bins=bins)
            axes[1].legend(legend, fontsize=8)
            axes[1].set_xlabel(f"{key.replace('_', ' ')}")
            axes[1].set_ylabel('count per bin')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{key}.png')

# =============================================================================================================

def calculate_area_sim(
    gt_coordinates,
    predict_coordinates,
    grid_size
):
    # *_coordinates: (batch, 2, height*width)
    sim_list = []
    device = gt_coordinates.device
    gt_coordinates = gt_coordinates.permute(0, 2, 1).detach().cpu().numpy()
    predict_coordinates = predict_coordinates.permute(0, 2, 1).detach().cpu().numpy()

    for gt_per, pred_per in zip(gt_coordinates, predict_coordinates):
        gt_len = (gt_per.sum(1) >= 0).sum()
        pred_len = (pred_per.sum(1) >= 0).sum()

        # This is a patch for invalid test sample without continous labelling,
        # e.g., for sdd inter at batch 95 and 0th sample, in the gt labelling, the start node has no neighbours
        if gt_len <= 1: continue

        area_per = SM.area_between_two_curves(gt_per[:gt_len], pred_per[:pred_len])
        sim_per = 1 - area_per / grid_size
        sim_list.append(sim_per)

    sim_list = np.stack(sim_list, axis=0)
    sim_list = torch.from_numpy(sim_list).type(torch.FloatTensor).to(device)

    return sim_list