from models.autoencoder import DemAutoencoder
from data.dems import DemData

import cppimport.import_hook
from grid_planner import grid_planner

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
from data_utils import resize_image


def get_predictions(image_size, name='test', ckpt_path='./model.ckpt'):
    dataset = DemData(name, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    model = DemAutoencoder(resolution=(128, 128))
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()
    predictions_dem = []
    predictions_focal = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            # dem, rgb, sg, focal = batch
            dem = batch['maps']
            rgb = batch['rgb']
            sg = batch['sg']
            focal = batch['ppm']

            inputs = torch.cat([rgb, sg], dim=1)
            predictions = (model(inputs) + 1) / 2
            predictions_dem.append(predictions[:, 0].numpy())
            predictions_focal.append(predictions[:, 1].numpy())

    predictions_dem = np.stack(predictions_dem, axis=0)
    predictions_focal = np.stack(predictions_focal, axis=0)
    np.savez(name + '_predictions.npz', dem=predictions_dem, focal=predictions_focal)
    print('Saved predictions to ' + name + '_predictions.npz')

    return predictions_dem, predictions_focal


def convert_coord2map(height, width, coordinates):
    map = np.zeros((height, width)).astype(np.float32)

    for cor in coordinates:
        map[cor[0], cor[1]] = 1

    return map


def get_metrics(dataset_dir, image_size, name='test', ckpt_path='./model.ckpt', enable_save_gt_path_only=False):
    source_data = np.load(f'{dataset_dir}/{name}.npz')
    source_focal = np.load(f'{dataset_dir}/{name}_focal.npz')
    gt_dem = source_data['dem']
    starts = source_focal['start']
    goals = source_focal['goal']
    gt_focal = source_focal['focal']
    image_size_org = gt_dem.shape[-1]

    if image_size != image_size_org:
        gt_dem, gt_focal = resize_image([gt_dem, gt_focal], image_size)
        starts, goals = resize_image([starts, goals], image_size, mode='coordinate')

    if not enable_save_gt_path_only:
        if os.path.exists(f'{dataset_dir}/{name}_predictions.npz'):
            print('loading predictions')
            predictions = np.load(f'{dataset_dir}/{name}_predictions.npz')
            predictions_dem = predictions['dem']
            predictions_focal = predictions['focal']
        else:
            predictions_dem, predictions_focal = get_predictions(image_size, f'{dataset_dir}/{name}', ckpt_path)

    # save_dict = {'predicted_dem': predictions_dem, 'predicted_focal': predictions_focal}

    gt_dem_path_maps = []
    gt_focal_path_maps = []
    pred_dem_path_maps = []
    pred_focal_path_maps = []

    gt_dem_num = []
    pred_dem_num = []
    pred_focal_num = []
    save_half_index = len(gt_dem) // 3

    for i in tqdm(range(len(gt_dem))):
        if name == 'train':
            save_index = i // save_half_index
            save_path = f'{dataset_dir}/{name}_dem_ppm_part{save_index}.npz'
            if os.path.exists(save_path): continue

        for j in range(10):
            # search with A* and gt-dem
            planner = grid_planner(gt_dem[i][0].tolist())
            gt_dem_path = planner.find_path(starts[i][j], goals[i][j])
            gt_dem_num.append(planner.get_num_expansions())

            # search with A* and gt-focal
            planner = grid_planner(gt_focal[i][j][0].tolist())
            gt_focal_path = planner.find_path(starts[i][j], goals[i][j])

            gt_dem_path_map = convert_coord2map(image_size, image_size, gt_dem_path)
            gt_focal_path_map = convert_coord2map(image_size, image_size, gt_focal_path)
            gt_dem_path_maps.append(gt_dem_path_map)
            gt_focal_path_maps.append(gt_focal_path_map)

            # ====
            if not enable_save_gt_path_only:
                # search with A* and pred-dem
                planner = grid_planner((predictions_dem[i][j] * 255.).tolist())
                pred_dem_path = planner.find_path(starts[i][j], goals[i][j])
                pred_dem_num.append(planner.get_num_expansions())

                # focal search with predicted dem and focal values
                planner = grid_planner((predictions_dem[i][j] * 255.))
                pred_focal_path = planner.find_focal_path_reexpand(starts[i][j], goals[i][j], predictions_focal[i][j].tolist())
                pred_focal_num.append(planner.get_num_expansions())

                pred_dem_path_map = convert_coord2map(image_size, image_size, pred_dem_path)
                pred_focal_path_map = convert_coord2map(image_size, image_size, pred_focal_path)
                pred_dem_path_maps.append(pred_dem_path_map)
                pred_focal_path_maps.append(pred_focal_path_map)

            plt.figure()
            plt.subplot(2, 4, 1)
            plt.imshow(gt_dem[i, 0])
            plt.subplot(2, 4, 2)
            plt.imshow(gt_focal[i, j, 0])

            if not enable_save_gt_path_only:
                plt.subplot(2, 4, 3)
                plt.imshow(predictions_dem[i, j])
                plt.subplot(2, 4, 4)
                plt.imshow(predictions_focal[i, j])

            plt.subplot(2, 4, 5)
            plt.imshow(gt_dem_path_map)
            plt.subplot(2, 4, 6)
            plt.imshow(gt_focal_path_map)

            if not enable_save_gt_path_only:
                plt.subplot(2, 4, 7)
                plt.imshow(pred_dem_path_map)
                plt.subplot(2, 4, 8)
                plt.imshow(pred_focal_path_map)

            plt.savefig(f'{i}_{j}.png')
            plt.close()

        if name == 'train' and i > 0 and ((i + 1) % save_half_index == 0 or i == len(gt_dem) - 1):
            gt_dem_paths = np.stack(gt_dem_path_maps, axis=0)
            # predicted_dem_paths = np.stack(pred_dem_path_maps, axis=0)
            # predicted_focal_paths = np.stack(pred_focal_path_maps, axis=0)

            print(f"Saved {save_index}.")
            np.savez(
                save_path,
                # gt_dem_paths=gt_dem_paths,
                # predicted_dem_paths=predicted_dem_paths,
                # predicted_focal_paths=predicted_focal_paths
            )

            del gt_dem_path_maps, pred_dem_path_maps, pred_focal_path_maps
            torch.cuda.empty_cache()

            gt_dem_path_maps = []
            pred_dem_path_maps = []
            pred_focal_path_maps = []

    if name != 'train':
        gt_dem_paths = np.stack(gt_dem_path_maps, axis=0)
        # predicted_dem_paths = np.stack(pred_dem_path_maps, axis=0)
        # predicted_focal_paths = np.stack(pred_focal_path_maps, axis=0)

        np.savez(
            f'{dataset_dir}/{name}_dem_ppm.npz',
            # gt_dem_paths=gt_dem_paths,
            # predicted_dem_paths=predicted_dem_paths,
            # predicted_focal_paths=predicted_focal_paths
        )

        torch.cuda.empty_cache()

    gt_dem_num = np.array(gt_dem_num)
    pred_dem_num = np.array(pred_dem_num)
    pred_focal_num = np.array(pred_focal_num)

    focal2pred_ratio_mean = (pred_focal_num / pred_dem_num).mean()
    pred2gt_ratio_mean = (pred_dem_num / gt_dem_num).mean()
    general_ratio_mean = (pred_focal_num / gt_dem_num).mean()

    print(f'Focal2pred ratio: {focal2pred_ratio_mean:.3f}')
    print(f'Pred2gt ratio: {pred2gt_ratio_mean:.3f}')
    print(f'General ratio:{general_ratio_mean:.3f}')