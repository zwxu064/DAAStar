# -------------------------------------------------------------------------------------------------------------
# File: compute_hmean.py
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

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")

from glob import glob
from src.utils.metrics import parse_args
from src.utils.notebook import parse_folder_path, get_dataset_file_names

# =============================================================================================================

def points_between_paths(
    path1,
    path2
):
    path1_set = set(path1)
    path2_set = set(path2)

    # Ensure paths are closed loops
    assert path1[0] == path2[0] and path1[-1] == path2[-1], "Paths must have the same start and end points."

    # Get bounding box
    min_x = min(p[0] for p in path1 + path2)
    max_x = max(p[0] for p in path1 + path2)
    min_y = min(p[1] for p in path1 + path2)
    max_y = max(p[1] for p in path1 + path2)

    enclosed_points = set()

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # Skip the boundary points
            if (x, y) in path1_set or (x, y) in path2_set:
                continue

            # Flood fill or point-in-polygon test
            # Check if (x, y) is inside the region defined by the paths
            crossings = 0

            for i in range(len(path1) - 1):
                x1, y1 = path1[i]
                x2, y2 = path1[i + 1]
                if y1 <= y < y2 or y2 <= y < y1:  # Crossing check
                    x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    if x < x_cross:
                        crossings += 1

            for i in range(len(path2) - 1):
                x1, y1 = path2[i]
                x2, y2 = path2[i + 1]
                if y1 <= y < y2 or y2 <= y < y1:
                    x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    if x < x_cross:
                        crossings += 1

            # Odd crossings mean the point is inside
            if crossings % 2 == 1:
                enclosed_points.add((x, y))

    return enclosed_points

# =============================================================================================================

def calculate_area_sim_union(
    predictions,
    reference,
    height=None,
    width=None
):
    enable_visual = (height != None) and (width != None)

    if enable_visual:
        plt.figure()
        num_plots = len(predictions)
        count = 1

    # Get the area between the prediction and reference, then union them
    unions = set()
    prediction_areas = []
    reference_set = set(reference)

    for prediction in predictions:
        prediction_set = set(prediction)
        points = points_between_paths(prediction, reference)
        points = points.union(prediction_set)
        points = points.union(reference_set)
        prediction_areas.append(points - reference_set)
        unions = unions.union(points - reference_set)

        if enable_visual:
            image = np.zeros((height, width), dtype=np.uint8)
            np_points = np.array(list(points))
            image[np_points[:, 0], np_points[:, 1]] = 255
            plt.subplot(1, num_plots, count)
            plt.imshow(image)
            count += 1

    # Exclude reference nodes
    union_area_size = len(unions)
    union_area_size = 1 if (union_area_size == 0) else union_area_size

    if enable_visual: plt.savefig('areas.png')

    # Area over union area
    prediction_area_sim = [1 - len(v) / union_area_size for v in prediction_areas]

    return union_area_size, prediction_area_sim

# =============================================================================================================

def compute_relative_asim(
    prediction_stack_list,
    reference_stack
):
    num_methods = len(prediction_stack_list)
    num_samples = len(reference_stack)
    union_area, relative_asims = [], []

    for idx_batch in range(num_samples):
        if idx_batch % 100 == 0: print(f"{idx_batch + 1}/{num_samples}.")
        
        # Remove -1 invalid coordinates.
        # Check calculate_area_sim() in src/utils/metrics.py.
        gt_per = reference_stack[idx_batch].astype(np.int32)
        gt_len = (gt_per.sum(1) >= 0).sum()

        # This is a patch for invalid test sample without continous labelling,
        # e.g., for sdd inter at batch 95 and 0th sample, in the gt labelling,
        # the start node has no neighbours
        if gt_len <= 1: continue
        reference = [tuple(v) for v in gt_per[:gt_len].tolist()]
        predictions = []

        for idx_method in range(num_methods):
            pred_per = prediction_stack_list[idx_method][idx_batch].astype(np.int32)
            pred_len = (pred_per.sum(1) >= 0).sum()
            prediction = [tuple(v) for v in pred_per[:pred_len].tolist()]

            # In case some prediction does not include starting and ending nodes
            if prediction[0] != reference[0]: prediction = [reference[0]] + prediction
            if prediction[-1] != reference[-1]: prediction = prediction + [reference[-1]]

            predictions.append(prediction)

        union_area_per, relative_asim_per = calculate_area_sim_union(predictions, reference)
        union_area.append(union_area_per)
        relative_asims.append(relative_asim_per)

    union_area = np.stack(union_area, axis=0)
    relative_asims = np.stack(relative_asims, axis=0)

    return union_area, relative_asims

# =============================================================================================================

def get_harmonic_mean_path(config):
    if config.enable_motion_planning_lib:
        if config.dataset_name in ["mpd", "street", "tmpd", "aug_tmpd", "warcraft", "pkmn"]:
            metric_file_path = f"{config.resume_path_dir}/{config.dataset_name}" \
                f"/{config.dataset_file_name}/path_stack_HarmonicMean.npy"
            metric_file_path = metric_file_path.replace("seed1", "seed0").replace("seed2", "seed0")
        else:
            metric_file_path = f"{config.resume_path_dir}/{config.dataset_name}/path_stack_HarmonicMean.npy"
    else:
        if config.dataset_name == "aug_tmpd" and config.enable_transpath:
            metric_file_path = config.transpath.pretrained_model_path.replace(".pth", "_HarmonicMean.npy")
        else:
            metric_file_paths = sorted(glob(
                f"{config.logdir}/{parse_folder_path(config)}_pid*/**/*_HarmonicMean.npy",
                recursive=True
            ))
            assert len(metric_file_paths) >= 1, \
                f"{config.logdir}/{parse_folder_path(config)}_pid*/**/*_HarmonicMean.npy not exist."
            metric_file_path = metric_file_paths[0]

    assert os.path.exists(metric_file_path), f"{metric_file_path} not exists."

    return metric_file_path

# =============================================================================================================

def get_methods(dataset_name):
    if dataset_name == "aug_tmpd":
        methods = [
            "a_star",
            "theta_star",
            "neural_astar",
            "randomwalk_3",
            "transpath",
            "daa_mix",
            "daa_path",
            "daa_mask",
            "daa_weight"
        ]
    elif dataset_name in ["sdd_intra", "sdd_inter", "warcraft", "pkmn"]:
        methods = [
            "neural_astar",
            "randomwalk_3",
            "daa_min",
            "daa_max",
            "daa_mix"
        ]
    else:
        methods = [
            "a_star",
            "theta_star",
            "neural_astar",
            "randomwalk_3",
            "daa_min",
            "daa_max",
            "daa_mix"
        ]

    return methods

# =============================================================================================================

def main():
    config_file_path = './config.yaml'
    config = parse_args(config_file_path)
    seeds = [0] if config.dataset_name == 'aug_tmpd' else [0, 1, 2]
    relative_asim_dict = {}

    #
    test_scenes = [None]
    dataset_file_names = [None]

    if config.dataset_name == "sdd_inter":
        test_scenes = config.sdd.test_scene_dict['inter']
    elif config.dataset_name in [
        "mpd",
        "street",
        "tmpd",
        "aug_tmpd",
        "warcraft",
        "pkmn",
        "sdd_intra"
    ]:
        dataset_file_names = get_dataset_file_names(config)

    #
    methods = get_methods(config.dataset_name)

    for seed in seeds:
        config.seed = seed
        relative_asim_dict.update({seed: {}})
        relative_asim_dict_obj = relative_asim_dict[seed]

        for dataset_file_name in dataset_file_names:
            if dataset_file_name is not None:
                config.dataset_file_name = dataset_file_name
                relative_asim_dict[seed].update({dataset_file_name: {}})
                relative_asim_dict_obj = relative_asim_dict[seed][dataset_file_name]

            for test_scene in test_scenes:
                if test_scene is not None:
                    config.sdd.test_scene = test_scene
                    relative_asim_dict[seed].update({test_scene: {}})
                    relative_asim_dict_obj = relative_asim_dict[seed][test_scene]

                nastar_path_stack_list = []

                for method in methods:
                    config = parse_args(config=config, method=method)
                    metric_file_path = get_harmonic_mean_path(config)

                    assert os.path.exists(metric_file_path), f'{metric_file_path} not exist.'

                    # Open file
                    data = np.load(metric_file_path, allow_pickle=True).item()
                    nastar_path_stack = data["nastar_path_stack"]
                    gt_path_stack = data["gt_path_stack"]
                    assert nastar_path_stack.shape[0] == gt_path_stack.shape[0]

                    nastar_path_stack_list.append(nastar_path_stack)

                _, relative_asims = compute_relative_asim(nastar_path_stack_list, gt_path_stack)

                for i, method in enumerate(methods):
                    relative_asim_dict_obj.update({method: relative_asims[:, i]})

    mean_metrics_dict = {}

    for method in methods:
        mean_metrics_dict.update({method: {}})
        config = parse_args(config=config, method=method)
        metrics_dict = {
            "spr": [],
            "psim": [],
            "asim": [],
            "relative_asim": [],
            "ep": [],
            "hist": [],
            "cd": [],
            "h_mean_old": [],
            "h_mean_new": []
        }

        for seed in seeds:
            config.seed = seed
            relative_asim_dict_obj = relative_asim_dict[seed]

            sub_metrics_dict = {
                "spr": [],
                "psim": [],
                "asim": [],
                "relative_asim": [],
                "hist": [],
                "cd": [],
                "h_mean_old": [],
                "h_mean_new": []
            }

            for dataset_file_name in dataset_file_names:
                if dataset_file_name is not None:
                    config.dataset_file_name = dataset_file_name
                    relative_asim_dict_obj = relative_asim_dict[seed][dataset_file_name]

                for test_scene in test_scenes:
                    if test_scene is not None:
                        config.sdd.test_scene = test_scene
                        relative_asim_dict_obj = relative_asim_dict[seed][test_scene]

                    config = parse_args(config=config)
                    metric_file_path = get_harmonic_mean_path(config)

                    assert os.path.exists(metric_file_path), f'{metric_file_path} not exist.'

                    # Open file
                    data = np.load(metric_file_path, allow_pickle=True).item()
                    vastar_path_stack = data["vastar_path_stack"]
                    spr = data["metrics"]["p_opt_nastar"]
                    psim = data["metrics"]["traj_sim"]
                    asim = data["metrics"]["area_sim"]
                    h_mean_old = data["metrics"]["h_mean"]
                    hist = data["metrics"]["hist_ratio_nastar"]
                    cd = data["metrics"]["c_distance"]
                    relative_asim = np.array(relative_asim_dict_obj[method])
                    support_vastar = vastar_path_stack is not None

                    sub_metrics_dict["spr"].append(np.mean(spr))
                    sub_metrics_dict["psim"].append(np.mean(psim))
                    sub_metrics_dict["asim"].append(np.mean(asim))
                    sub_metrics_dict["h_mean_old"].append(np.mean(h_mean_old))
                    sub_metrics_dict["relative_asim"].append(np.mean(relative_asim))
                    sub_metrics_dict["hist"].append(np.mean(hist))
                    sub_metrics_dict["cd"].append(np.mean(cd))

            spr = np.mean(sub_metrics_dict["spr"])
            psim = np.mean(sub_metrics_dict["psim"])
            asim = np.mean(sub_metrics_dict["asim"])
            relative_asim = np.mean(sub_metrics_dict["relative_asim"])

            metrics_dict["spr"].append(spr)
            metrics_dict["psim"].append(psim)
            metrics_dict["asim"].append(asim)
            metrics_dict["h_mean_old"].append(np.mean(sub_metrics_dict["h_mean_old"]))
            metrics_dict["relative_asim"].append(relative_asim)
            metrics_dict["hist"].append(np.mean(sub_metrics_dict["hist"]))
            metrics_dict["cd"].append(np.mean(sub_metrics_dict["cd"]))

            if support_vastar:
                h_mean_list = [spr, psim, relative_asim]
            elif config.dataset_name == 'aug_tmpd':
                h_mean_list = [spr, psim, relative_asim]
            else:
                h_mean_list = [psim, relative_asim]

            h_mean_inv_list = 1 / np.array(h_mean_list)
            h_mean_new = len(h_mean_inv_list) / np.sum(h_mean_inv_list)

            metrics_dict["h_mean_new"].append(h_mean_new)

        for key in metrics_dict.keys():
            value = metrics_dict[key]
            if len(value) == 0: continue
            mean_metrics_dict[method].update({key: np.mean(value)})

    print(mean_metrics_dict)

# =============================================================================================================

if __name__ == '__main__':
    main()