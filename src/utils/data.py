# -------------------------------------------------------------------------------------------------------------
# File: data.py
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

import numpy as np
import torch, sys, os
import torch.utils.data as data

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from PIL import Image
from torchvision.utils import make_grid
from src.utils.sdd import SDD

# =============================================================================================================

def visualize_results(
    map_designs,
    planner_outputs,
    scale=1
):
    if type(planner_outputs) == dict:
        histories = planner_outputs["histories"]
        paths = planner_outputs["paths"]
    else:
        histories = planner_outputs.histories
        paths = planner_outputs.paths
    results = make_grid(map_designs).permute(1, 2, 0)
    h = make_grid(histories).permute(1, 2, 0)
    p = make_grid(paths).permute(1, 2, 0).float()
    results[h[..., 0] == 1] = torch.tensor([0.2, 0.8, 0])
    results[p[..., 0] == 1] = torch.tensor([1.0, 0.0, 0])

    results = ((results.numpy()) * 255.0).astype("uint8")

    if scale > 1:
        results = Image.fromarray(results).resize(
            [x * scale for x in results.shape[:2]], resample=Image.NEAREST
        )
        results = np.asarray(results)

    return results

# =============================================================================================================

def create_dataloader(
    filename,
    split,
    batch_size,
    eval_seed,
    num_starts=1,
    shuffle=False,
    test_scene='gates',
    hardness_threshold=None,
    disable_random_start=False
):
    if filename.find('sdd') > -1:
        if split == 'train':
            hardness = SDD(
                filename,
                is_train=True,
                test_scene=test_scene,
                load_hardness=True
            )

            hardness = np.array([x for x in hardness])

            train_dataset = SDD(
                filename,
                is_train=True,
                test_scene=test_scene,
                load_hardness=False
            )

            dataset = data.Subset(
                train_dataset,
                np.where(hardness <= float(hardness_threshold))[0]
            )
        else:
            dataset = SDD(
                filename,
                is_train=False,
                test_scene=test_scene,
                load_hardness=False
            )
    else:
        dataset = MazeDataset(
            filename,
            split,
            eval_seed=eval_seed,
            num_starts=num_starts,
            disable_random_start=disable_random_start
        )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

# =============================================================================================================

class MazeDataset(data.Dataset):
    def __init__(
        self,
        filename,
        split,
        eval_seed,
        pct1=0.55,
        pct2=0.70,
        pct3=0.85,
        num_starts=1,
        disable_random_start=False
    ):
        self.disable_random_start = disable_random_start
        assert filename.endswith("npz")
        self.filename = filename
        self.dataset_type = split  # train, valid, test
        self.pcts = np.array([pct1, pct2, pct3, 1.0])
        self.num_starts = num_starts
        self.eval_seed = eval_seed

        (
            self.map_designs,
            self.goal_maps,
            self.opt_policies,
            self.opt_dists,
        ) = self._process(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(
        self,
        filename
    ):
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]
            map_designs = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]
            opt_dists = f["arr_" + str(idx + 3)]

        # Set proper datatypes
        map_designs = map_designs.astype(np.float32)
        goal_maps = goal_maps.astype(np.float32)
        opt_policies = opt_policies.astype(np.float32)
        opt_dists = opt_dists.astype(np.float32)

        print(
            f"Number of {self.dataset_type} samples: {map_designs.shape[0]}" \
            f", size: {map_designs.shape[1]}x{map_designs.shape[2]}."
        )
        return map_designs, goal_maps, opt_policies, opt_dists

    def __getitem__(
        self,
        index
    ):
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]
        start_maps, opt_trajs = [], []
        map_designs, goal_maps = [], []

        # NOTE, set a seed, otherwise cannot used for visualization
        # Generally, when dataset_type=train, num_starts=1;
        # this varys when use python_motion_planning to generate optimal path
        # for trainset with num_start>1.
        if self.dataset_type in ['valid', 'test'] or self.num_starts > 1:
            np.random.seed(self.eval_seed)

        for _ in range(self.num_starts):
            if self.dataset_type in ['valid', 'test']:
                if self.disable_random_start:
                    indices = [0]
                else:
                    indices = [0, 1, 2]
            else:
                indices = [None]

            for start_idx in indices:
                # NOTE 2024-Oct-18, this start_map may not be good, see my comments
                # in get_random_start_map().
                # For instance, TMPD index=37, the 4nd sample (start_map[4]).
                # One can save it as a figure and visualize it.
                start_map = self.get_random_start_map(opt_dist, index=start_idx)
                opt_traj = self.get_opt_traj(start_map, goal_map, opt_policy)
                start_maps.append(start_map)
                opt_trajs.append(opt_traj)

                map_designs.append(map_design)
                goal_maps.append(goal_map)

        start_maps = np.concatenate(start_maps)
        opt_trajs = np.concatenate(opt_trajs)

        if len(map_designs) > 0:
            map_designs = np.concatenate(map_designs)
            goal_maps = np.concatenate(goal_maps)

        return {
            'maps': map_designs,
            'starts': start_maps,
            'goals': goal_maps,
            'optimal_paths': opt_trajs
        }

    def __len__(self):
        return self.map_designs.shape[0]

    def get_opt_traj(
        self,
        start_map,
        goal_map,
        opt_policy
    ):
        opt_traj = np.zeros_like(start_map)
        opt_policy = opt_policy.transpose((1, 2, 3, 0))
        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())

        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self.next_loc(current_loc, opt_policy[current_loc])
            assert (
                opt_traj[next_loc] == 0.0
            ), "Revisiting the same position while following the optimal policy"
            current_loc = next_loc

        # NOTE GT path does not include goal_map, so sum them up.
        opt_traj = ((opt_traj + start_map + goal_map) > 0).astype(opt_traj.dtype)

        return opt_traj

    def get_random_start_map(
        self,
        opt_dist,
        index=None
    ):
        # NOTE 2024-Oct-18
        # This function has a problem that it should not go cross diagonal when it is blocked.
        # But this start_idx can be within a block, which should not.
        # This is caused by the wrong opt_dist (predefined from download)
        # which has no large cost within this block.
        # Using empirical Dijkstra policy provided by NeuralA* paper can go cross this block,
        # but this policy is not good due to zig-zag. From a reviewer's comment.
        # Such start_idx cannot be used to generate path by python_motion_planning lib.
        # To the present, just keep this sample, but when the generated path above is empty,
        # then ignore this sample in the evalution.

        od_vct = opt_dist.flatten()
        od_vals = od_vct[od_vct > od_vct.min()]
        od_th = np.percentile(od_vals, 100.0 * (1 - self.pcts))

        if index is None:
            r = np.random.randint(0, len(od_th) - 1)
        else:
            r = index

        start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        start_idx = np.random.choice(np.where(start_candidate)[0])
        start_map = np.zeros_like(opt_dist)
        start_map.ravel()[start_idx] = 1.0

        return start_map

    def next_loc(
        self,
        current_loc,
        one_hot_action
    ):
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        move = action_to_move[np.argmax(one_hot_action)]
        return tuple(np.add(current_loc, move))

# =============================================================================================================

def create_game_dataloader(
    dirname,
    split,
    batch_size,
    shuffle=False
):
    dataset = GameDataset(dirname, split)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

# =============================================================================================================

class GameDataset(data.Dataset):
    def __init__(
        self,
        dirname,
        split
    ):
        data = np.load(f'{dirname}/{split}.npz', mmap_mode='r')

        maps = data['maps'].astype(np.float32)  # batch,h,w,c
        starts = data['sources']                # batch,2,2,2
        goals = data['targets']                 # batch,2,2
        paths = data['paths']                   # batch,2,2,h,w
        costs = data['costs']                   # batch,h,w

        self.num_augs = 4

        self.map_designs = maps.transpose(0, 3, 1, 2)
        num_samples = self.map_designs.shape[0]

        if dirname.find('warcraft') > -1:
            h, w = 12, 12
        elif dirname.find('pkmn') > -1:
            h, w = 20, 20
        else:
            assert False

        starts = starts.reshape(num_samples, 4, 2)[:, :self.num_augs].reshape(num_samples * self.num_augs, 2)
        goals = np.repeat(goals.reshape(num_samples, 2, 1, 2), 2, axis=2).reshape(num_samples, 4, 2)
        goals = goals[:, :self.num_augs].reshape(num_samples * self.num_augs, 2)
        self.paths = paths.reshape(num_samples, 4, h, w)[:, :self.num_augs]
        self.cost_maps = (costs - costs.min()) / (costs.max() - costs.min())

        print(
            f'Loading {split} of size {self.map_designs.shape}' \
            f' augmented by {self.num_augs} times, target size {paths.shape}.'
        )

        starts = starts.astype(np.int64)
        start_maps = np.zeros((num_samples * self.num_augs, h, w)).astype(np.float32)
        start_maps[np.arange(num_samples * self.num_augs), starts[:, 0], starts[:, 1]] = 1
        self.start_maps = start_maps.reshape(num_samples, self.num_augs, h, w)

        goals = goals.astype(np.int64)
        goal_maps = np.zeros((num_samples * self.num_augs, h, w)).astype(np.float32)
        goal_maps[np.arange(num_samples * self.num_augs), goals[:, 0], goals[:, 1]] = 1
        self.goal_maps = goal_maps.reshape(num_samples, self.num_augs, h, w)

    def __len__(self):
        return self.map_designs.shape[0] * self.num_augs

    def __getitem__(
        self,
        index
    ):
        map_idx, task_idx = index // self.num_augs, index % self.num_augs
        h, w = self.paths.shape[-2:]

        map_design = self.map_designs[map_idx] / 255
        start_map = self.start_maps[map_idx][task_idx].reshape(1, h, w)
        goal_map = self.goal_maps[map_idx][task_idx].reshape(1, h, w)
        gt_path = self.paths[map_idx][task_idx].reshape(1, h, w)
        cost_map = self.cost_maps[map_idx].reshape(1, h, w)

        return {
            'maps': map_design,
            'starts': start_map,
            'goals': goal_map,
            'optimal_paths': gt_path,
            'ppm': cost_map
        }