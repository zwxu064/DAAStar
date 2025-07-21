# -------------------------------------------------------------------------------------------------------------
# File: train.py
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

import sys, wandb, pytz, os, torch
import pytorch_lightning as pl

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")

from copy import deepcopy
from src.planner import NeuralAstar, MotionPlanning
from src.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.metrics import parse_args
from src.utils.notebook import get_dataset_file_names, get_dataloaders, parse_folder_path
from src.utils.notebook import search_checkpoints
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from eval import eval

# =============================================================================================================

def main_unit(config):
    if config.dataset_name in ['aug_tmpd']:
        torch.set_float32_matmul_precision('high')

    set_global_seeds(config.seed)

    print(f'dataset_file_name:', config.dataset_file_name)

    # dataloaders
    train_loader, val_loader, _ = get_dataloaders(config)

    if config.enable_motion_planning_lib:
        motion_planner = MotionPlanning(config)
    else:
        motion_planner = NeuralAstar(config)

    # Empirical: monitor original: metrics/h_mean, but when valid seed is fixed, it is on epoch 1
    # which is bad, so use metrics/h_opt for the number of steps in the path
    if config.dataset_name == 'aug_tmpd':
        checkpoint_callback = ModelCheckpoint(
            save_weights_only=True,
            save_top_k=-1
        )
    else:
        mode = 'max'
        monitor_metrics = 'valid/h_mean'

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metrics,
            save_weights_only=True,
            mode=mode,
            save_top_k=1
        )

    pid = os.getpid()
    logdir = f'{config.logdir}/{parse_folder_path(config)}_pid{pid}'

    # Logger
    if config.logger == 'wandb' and not config.enable_motion_planning_lib:
        # Set wandb logger project and program names
        project_name = f"DAA_{config.dataset_name}"
        name = f"{config.now}_{config.method}_seed{config.seed}_pid{pid}"

        # Run wandb
        logger = WandbLogger(
            project=project_name,
            name=name,
            config={
                'learning_rate': config.params.lr,
                'epochs': config.params.num_epochs
            },
            save_dir=logdir
        )
    else:
        logger = None

    module = PlannerModule(motion_planner, config, wandb_logger=logger)

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0 if config.enable_motion_planning_lib else 2,
        check_val_every_n_epoch=config.params.num_epochs * 2 if config.enable_motion_planning_lib else 1
    )

    # Check config
    print(config)

    # Train and valid
    trainer.fit(module, train_loader, val_loader)

    # Test, !!!config will be changed globally if it is changed inside eval() but no deepcopy
    if config.dataset_name in ['tmpd', 'street', 'warcraft', 'pkmn', "sdd_intra"]:
        eval(config, split='test', batch_size=20)

    wandb.finish()

# =============================================================================================================

def main():
    config_file_path = './config.yaml'
    config = parse_args(config_file_path)

    # Set auto seeds and datasets
    if config.enable_motion_planning_lib:
        seeds = [0]
    else:
        if config.seed is None:
            seeds = [0] if (config.dataset_name == 'aug_tmpd') else [0, 1, 2]
        else:
            seeds = [config.seed]

    if config.dataset_name is None:
        dataset_names = ['street', 'tmpd', 'mpd', 'sdd_inter', 'sdd_intra', 'warcraft', 'pkmn']
    else:
        dataset_names = [config.dataset_name]
        sdd_test_scene_types = [config.sdd.test_scene_type]

    # Loop over seeds
    for seed in seeds:
        # Loop over datasets
        for dataset_name in dataset_names:
            config_modified = deepcopy(config)
            config_modified.dataset_name = dataset_name
            config_modified.seed = seed
            config_modified.now = pytz.utc.localize(datetime.now()).astimezone(
                pytz.timezone('Australia/Sydney')).strftime("%Y%m%d-%H%M%S")
            dataset_file_names = get_dataset_file_names(config_modified)

            for dataset_file_name in dataset_file_names:
                if dataset_name.find('sdd') > -1:
                    for test_scene_type in sdd_test_scene_types:
                        test_scene_list = config.sdd.test_scene_dict[test_scene_type]

                        # Loop over different scenes
                        for test_scene in test_scene_list:
                            config_modified.sdd.test_scene_type = test_scene_type
                            config_modified.sdd.test_scene = test_scene
                            config_modified.dataset_file_name = dataset_file_name

                            # Reset logdir for new seed, dataset_name, test_scene
                            config_modified = parse_args(config=config_modified)

                            # Check if already ran
                            ckpt_files = search_checkpoints(config_modified)

                            if len(ckpt_files) >= 1:
                                print(f'!!!Warning {ckpt_files[-1]} exists, skip.')
                                continue

                            main_unit(config_modified)

                        # Run to average over all sub-dataset
                        if test_scene_type == 'inter':
                            eval(config_modified, split='test', batch_size=20)
                else:
                    config_modified.dataset_file_name = dataset_file_name

                    # Reset logdir for new seed, dataset_name, test_scene
                    config_modified = parse_args(config=config_modified)

                    # Check if already ran
                    ckpt_files = search_checkpoints(config_modified)

                    if len(ckpt_files) >= 1:
                        print(f'!!!Warning {ckpt_files[-1]} exists, skip.')
                        continue

                    main_unit(config_modified)

            # Run to average over all sub-datasets
            if config.dataset_name == 'mpd':
                eval(config_modified, split='test', batch_size=20)

# =============================================================================================================

if __name__ == "__main__":
    main()