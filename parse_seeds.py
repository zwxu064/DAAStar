# -------------------------------------------------------------------------------------------------------------
# File: parse_seeds.py
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

import os, statistics, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")

from glob import glob
from src.utils.metrics import parse_args
from src.utils.notebook import parse_folder_path

# =============================================================================================================

if __name__ == '__main__':
    config_file_path = './config.yaml'
    config = parse_args(config_file_path)
    enable_mpd_multiple = config.dataset_name == 'mpd' and config.dataset_file_name is None
    enable_sdd_multiple = config.dataset_name == 'sdd_inter'
    seeds = [0] if config.dataset_name == 'aug_tmpd' else [0, 1, 2]
    mean_dict = {}

    for seed in seeds:
        config.seed = seed
        config = parse_args(config=config)

        if enable_mpd_multiple or enable_sdd_multiple:
            log_file_name = f'average_{config.expand_mode}.txt'

            if config.dataset_name.find('sdd') > -1:
                search_tar = f"{config.logdir}/{config.dataset_name}/s064_0.5_128_300/{log_file_name}"
            else:
                search_tar = f"{config.logdir}/{config.dataset_name}/{log_file_name}"
        else:
            search_tar = f"{config.logdir}/{parse_folder_path(config)}_pid*/**/**/*.txt"

        log_file_path = sorted(glob(search_tar,recursive=True))[0]

        if enable_mpd_multiple or enable_sdd_multiple:
            start_parse = False
        else:
            start_parse = True

        assert os.path.exists(log_file_path), f'{log_file_path} not exist.'

        # Open file
        f = open(log_file_path, 'r')
        print(f"Seed: {seed}, log: {log_file_path}.")

        for line in f:
            line_v = line.strip()

            if start_parse is False:
                start_parse = line_v.find('Average over all subsets.') > -1

            if start_parse:
                if (line_v.find(f'Method: {config.encoder.arch}') > -1 and not enable_mpd_multiple) \
                    or (
                        line_v.find(f'Dataset: {config.dataset_name}') > -1
                        and (enable_mpd_multiple or enable_sdd_multiple)
                    ):
                    split_v = line_v.split(':')
                    meta = split_v[1].split(',')[-1].replace(' ', '')
                    value = split_v[-1][:-1]

                    if meta not in mean_dict.keys():
                        mean_dict.update({meta: [float(value)]})
                    else:
                        mean_dict[meta].append(float(value))
                
                zip_key_v = zip(
                    ['G ratio:', 'rotation const:', 'rotation weight:'],
                    ['g', 'rotation_const', 'rotation_weight']
                )

                for key_str, meta in zip_key_v:
                    if line_v.find(key_str) > -1:
                        value = line_v.split(key_str)[-1].split(',')[0]
                        value = value[:-1] if value[-1] == '.' else value

                        if meta not in mean_dict.keys():
                            mean_dict.update({meta: [float(value)]})
                        else:
                            mean_dict[meta].append(float(value))

        f.close()

    mean_std_dict = {}

    for meta in mean_dict.keys():
        list_v = mean_dict[meta]

        if len(list_v) == len(seeds):
            mean_v = statistics.mean(list_v)
            std_v = statistics.stdev(list_v) if len(seeds) > 1 else 0
            mean_std_dict.update({meta: {'mean': mean_v, 'std': std_v}})

            print(f'{meta}: {mean_v:.4f}, {std_v:.4f}.')

    if enable_mpd_multiple or enable_sdd_multiple:
        print(
            f"lambda: {100 * (1 - mean_std_dict['mean_g_ratio']['mean']):.2f}" \
            f", {100 * mean_std_dict['mean_g_ratio']['std']:.2f}."
        )
        print(
            f"alpha: {100 * mean_std_dict['mean_rotation_weight']['mean']:.2f}" \
            f", {100 * mean_std_dict['mean_rotation_weight']['std']:.2f}."
        )
        print(
            f"beta: {100 * mean_std_dict['mean_g_ratio']['mean'] * mean_std_dict['mean_rotation_const']['mean']:.2f}" \
            f", {100 * mean_std_dict['mean_g_ratio']['std'] * mean_std_dict['mean_rotation_const']['std']:.2f}."
        )
        print(
            f"kappa: {100 * mean_std_dict['mean_rotation_const']['mean']:.2f}" \
            f", {100 * mean_std_dict['mean_rotation_const']['std']:.2f}."
        )
    else:
        print(f"lambda: {100 * (1 - mean_std_dict['g']['mean']):.2f}, {100 * mean_std_dict['g']['std']:.2f}.")
        print(
            f"alpha: {100 * mean_std_dict['rotation_weight']['mean']:.2f}" \
            f", {100 * mean_std_dict['rotation_weight']['std']:.2f}."
        )
        print(
            f"beta: {100 * mean_std_dict['g']['mean'] * mean_std_dict['rotation_const']['mean']:.2f}" \
            f", {100 * mean_std_dict['g']['std'] * mean_std_dict['rotation_const']['std']:.2f}."
        )
        print(
            f"kappa: {100 * mean_std_dict['rotation_const']['mean']:.2f}" \
            f", {100 * mean_std_dict['rotation_const']['std']:.2f}."
        )