# -------------------------------------------------------------------------------------------------------------
# File: eval.py
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

import torch, os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")

from copy import deepcopy
from src.utils.notebook import run_statistics_all_methods
from src.utils.metrics import parse_args
from compute_hmean import get_methods

# =============================================================================================================

def eval(
    config,
    split='test',
    g_ratio_list=[0.5],
    batch_size=None,
    keep_dataset_file_name=False,
    enable_save_image=False,
    save_image_list=None,
    enable_save_path_stack=False,
):
    # This is important to keep external config unchanged
    config = deepcopy(config)

    if not keep_dataset_file_name:
        if config.dataset_name == 'mpd' \
            or (config.dataset_name == 'sdd_inter'):
            config.dataset_file_name = None

    if batch_size is not None: config.params.batch_size = batch_size
    config.enable_resume = True

    # Parse again to update checkpoint path with given seed otherwise seedNone
    config = parse_args(config=config)

    run_statistics_all_methods(
        config,
        [config.dataset_name],
        split,
        g_ratio_list,
        enable_save_image=enable_save_image,
        save_image_list=save_image_list,
        enable_save_path_stack=enable_save_path_stack,
        disable_compute_path_angle=False
    )

# =============================================================================================================

if __name__ == '__main__':
    config_file_path = './config.yaml'
    split = "test"  # test/val/train
    enable_save_path_stack = False  # NOTE this is used to compute relative ASIM, which is slow
    batch_size = 64

    config = parse_args(config_file_path)

    if config.dataset_name is not None:
        dataset_names = [config.dataset_name]
    else:
        dataset_names = [
            "mpd",
            "tmpd",
            "street",
            "warcraft",
            "pkmn",
            "aug_tmpd",
            "sdd_inter",
            "sdd_intra"
        ]

    for dataset_name in dataset_names:
        if config.method is not None:
            methods = [config.method]
        else:
            methods = get_methods(dataset_name)

        for method in methods:
            if method in ["a_star", "theta_star", "dijkstra"] or dataset_name == 'aug_tmpd':
                seeds = [0]
            else:
                # NOTE If train from scratch, set this as [0, 1, 2];
                # otherwise, as [0] because only this is given in OneDrive
                seeds = [0]  # [0, 1, 2]

            for seed in seeds:
                if seed == seeds[0]:
                    print(f"{'#' * 30}\ndataset: {dataset_name}\nmethod: {method}\n{'#' * 30}")

                config.seed = seed
                config = parse_args(config=config, method=method, dataset_name=dataset_name)
                eval(
                    config,
                    split=split,
                    batch_size=batch_size,
                    enable_save_image=False,
                    enable_save_path_stack=enable_save_path_stack
                )

        torch.cuda.empty_cache()