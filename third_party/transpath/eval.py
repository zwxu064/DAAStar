from models.autoencoder import Autoencoder
from data.hmaps import GridData
from modules.planners import DifferentiableDiagAstar, get_diag_heuristic

import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse


def main(mode, state_dict_path, dataset_dir='.', split='test', hardness_limit=1.05):
    device = 'cuda'
    
    test_data = GridData(
        path=f'{dataset_dir}/{split}',
        mode=mode
    )
    test_dataloader = DataLoader(test_data, batch_size=256,
                        shuffle=False, num_workers=0, pin_memory=True)
    model = Autoencoder(mode=mode)
    model.load_state_dict(torch.load(state_dict_path)['state_dict'])
    model.to(device)
    model.eval()
    
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1)

    if mode == 'cf':
        learnable_planner = DifferentiableDiagAstar(mode='k')
    else:
        learnable_planner = DifferentiableDiagAstar(mode=mode, f_w=100)

    vanilla_planner.to(device)
    learnable_planner.to(device)    
        
    expansions_ratio = []
    cost_ratio = []
    hardness = []

    gt_ppm = []
    predicted_ppm = []
    vanilla_path = []
    learn_path = []

    save_half_index = len(test_dataloader) // 3
    
    for idx, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            # map_design, start, goal, gt_heatmap = batch
            map_design = batch['maps']
            start = batch['starts']
            goal = batch['goals']
            gt_heatmap = batch['ppm']

            inputs = torch.cat([map_design, start + goal], dim=1) if mode == 'f' else torch.cat([map_design, goal], dim=1)
            inputs = inputs.to(device)

            predictions = (model(inputs) + 1) / 2

            learn_outputs = learnable_planner(
                predictions.to(device),
                start.to(device),
                goal.to(device),
                ((map_design == 0)*1.).to(device)
            )

            vanilla_outputs = vanilla_planner(
                ((map_design == 0)*1.).to(device),
                start.to(device),
                goal.to(device),
                ((map_design == 0)*1.).to(device)
            )

            gt_ppm.append(gt_heatmap)
            predicted_ppm.append(predictions)
            vanilla_path.append(vanilla_outputs.paths)
            learn_path.append(learn_outputs.paths)

            # plt.figure()
            # plt.subplot(2, 2, 1)
            # plt.imshow(gt_heatmap[0, 0].cpu().numpy())
            # plt.subplot(2, 2, 2)
            # plt.imshow(predictions[0, 0].cpu().numpy())
            # plt.subplot(2, 2, 3)
            # plt.imshow(vanilla_outputs.paths[0, 0].cpu().numpy())
            # plt.subplot(2, 2, 4)
            # plt.imshow(learn_outputs.paths[0, 0].cpu().numpy())
            # plt.savefig(f'grid_{idx}.png')

            if split == 'train' and idx > 0 and ((idx + 1) % save_half_index == 0 or idx == len(test_dataloader) - 1):
                gt_ppm = torch.stack(gt_ppm, dim=0).detach().cpu().numpy()
                predicted_ppm = torch.stack(predicted_ppm, dim=0).detach().cpu().numpy()
                vanilla_paths = torch.stack(vanilla_path, dim=0).detach().cpu().numpy()
                learn_paths = torch.stack(learn_path, dim=0).detach().cpu().numpy()

                save_index = idx // save_half_index
                save_path = f'{dataset_dir}/{split}_grid_ppm_part{save_index}.npz'
                print(f"Saved {save_index}.")
                np.savez(
                    save_path,
                    gt_ppm=gt_ppm,
                    predicted_ppm=predicted_ppm,
                    vanilla_paths=vanilla_paths,
                    learn_paths=learn_paths
                )

                del gt_ppm, predicted_ppm, vanilla_path, learn_path
                torch.cuda.empty_cache()

                gt_ppm = []
                predicted_ppm = []
                vanilla_path = []
                learn_path = []

            expansions_ratio.append(((learn_outputs.histories).sum((-1, -2, -3))) / ((vanilla_outputs.histories).sum((-1, -2, -3))))
            learn_costs = (learn_outputs.g * goal.to(device)).sum((-1, -2, -3))
            vanilla_costs = (vanilla_outputs.g * goal.to(device)).sum((-1, -2, -3))
            cost_ratio.append(learn_costs / vanilla_costs)
            start_heur = (get_diag_heuristic(goal[:, 0].to(device)) * start[:, 0].to(device)).sum((-1, -2))
            hardness.append(vanilla_costs / start_heur)

    # NOTE
    if split != 'train':
        gt_ppm = torch.stack(gt_ppm, dim=0).detach().cpu().numpy()
        predicted_ppm = torch.stack(predicted_ppm, dim=0).detach().cpu().numpy()
        vanilla_paths = torch.stack(vanilla_path, dim=0).detach().cpu().numpy()
        learn_paths = torch.stack(learn_path, dim=0).detach().cpu().numpy()

        np.savez(
            f'{dataset_dir}/{split}_grid_ppm.npz',
            gt_ppm=gt_ppm,
            predicted_ppm=predicted_ppm,
            vanilla_paths=vanilla_paths,
            learn_paths=learn_paths
        )

    expansions_ratio = torch.cat(expansions_ratio, dim=0)
    cost_ratio = torch.cat(cost_ratio, dim=0)
    hardness = torch.cat(hardness, dim=0)
    mask = torch.where(hardness >= hardness_limit, torch.ones_like(hardness), torch.zeros_like(hardness))
    n = mask.sum()
    expansions_ratio = (expansions_ratio * mask).sum() / n
    cost_ratio = (cost_ratio * mask).sum() / n
    torch.cuda.empty_cache()
    
    print(f'expansions_ratio: {expansions_ratio}, cost_ratio: {cost_ratio}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--weights_path', type=str, default='./weights/focal.pth')
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    args.seed = 1234
    args.weights_path = './checkpoints/f/epoch=159-val_loss=0.0030.ckpt'
    dataset_dir = './datasets/transpath'

    for split in ['train']:
        print(split)
        main(dataset_dir=dataset_dir, mode=args.mode, state_dict_path=args.weights_path, split=split)