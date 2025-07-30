# -------------------------------------------------------------------------------------------------------------
# File: pq_astar.py
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

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from pqdict import pqdict
from src.planner.differentiable_astar import AstarOutput

# =============================================================================================================

def get_neighbor_indices(
    idx,
    H,
    W
):
    neighbor_indices = []
    if idx % W - 1 >= 0:
        neighbor_indices.append(idx - 1)
    if idx % W + 1 < W:
        neighbor_indices.append(idx + 1)
    if idx // W - 1 >= 0:
        neighbor_indices.append(idx - W)
    if idx // W + 1 < H:
        neighbor_indices.append(idx + W)
    if (idx % W - 1 >= 0) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W - 1)
    if (idx % W + 1 < W) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W + 1)
    if (idx % W - 1 >= 0) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W - 1)
    if (idx % W + 1 < W) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W + 1)

    return np.array(neighbor_indices)

# =============================================================================================================

def compute_chebyshev_distance(
    idx,
    goal_idx,
    W
):
    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    dxdy = np.abs(loc - goal_loc)
    h = dxdy.sum() - dxdy.min()
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return h + 0.001 * euc

# =============================================================================================================

def get_history(
    close_list,
    H,
    W
):
    history = np.array([[idx % W, idx // W] for idx in close_list.keys()])
    history_map = np.zeros((H, W))
    history_map[history[:, 1], history[:, 0]] = 1

    return history_map

# =============================================================================================================

def backtrack(
    parent_list,
    goal_idx,
    H,
    W
):
    current_idx = goal_idx
    path = []

    while current_idx != None:
        path.append([current_idx % W, current_idx // W])
        current_idx = parent_list[current_idx]

    path = np.array(path)
    path_map = np.zeros((H, W))
    path_map[path[:, 1], path[:, 0]] = 1

    return path_map

# =============================================================================================================

def pq_astar(
    pred_costs,
    start_maps,
    goal_maps,
    map_designs,
    store_intermediate_results=False,
    g_ratio=0.5,
):
    assert (
        store_intermediate_results == False
    ), "store_intermediate_results = True is currently supported only for differentiable A*"

    pred_costs_np = pred_costs.detach().numpy()
    start_maps_np = start_maps.detach().numpy()
    goal_maps_np = goal_maps.detach().numpy()
    map_designs_np = map_designs.detach().numpy()
    histories = np.zeros_like(goal_maps_np)
    path_maps = np.zeros_like(goal_maps_np, np.int64)
    for n in range(len(pred_costs)):
        histories[n, 0], path_maps[n, 0] = solve_single(
            pred_costs_np[n, 0],
            start_maps_np[n, 0],
            goal_maps_np[n, 0],
            map_designs_np[n, 0],
            g_ratio,
        )

    return AstarOutput(
        torch.tensor(histories),
        torch.tensor(path_maps)
    )

# =============================================================================================================

def solve_single(
    pred_cost,
    start_map,
    goal_map,
    map_design,
    g_ratio=0.5,
):
    H, W = map_design.shape
    start_idx = np.argwhere(start_map.flatten()).item()
    goal_idx = np.argwhere(goal_map.flatten()).item()
    map_design_vct = map_design.flatten()
    pred_cost_vct = pred_cost.flatten()
    open_list = pqdict()
    close_list = pqdict()
    open_list.additem(start_idx, 0)
    parent_list = dict()
    parent_list[start_idx] = None
    num_steps = 0

    while goal_idx not in close_list:
        if len(open_list) == 0:
            print("goal not found")
            return np.zeros_like(goal_map), np.zeros_like(goal_map)
        num_steps += 1
        idx_selected, f_selected = open_list.popitem()
        close_list.additem(idx_selected, f_selected)
        for idx_nei in get_neighbor_indices(idx_selected, H, W):

            if map_design_vct[idx_nei] == 1:
                f_new = (
                    f_selected
                    - (1 - g_ratio)
                    * compute_chebyshev_distance(idx_selected, goal_idx, W)
                    + g_ratio * pred_cost_vct[idx_nei]
                    + (1 - g_ratio) * compute_chebyshev_distance(idx_nei, goal_idx, W)
                )

                # conditions for the nodes not yet in the open list nor closed list
                cond = (idx_nei not in open_list) & (idx_nei not in close_list)

                # condition for the nodes already in the open list but with larger f value
                if idx_nei in open_list:
                    cond = cond | (open_list[idx_nei] > f_new)

                if cond:
                    try:
                        open_list.additem(idx_nei, f_new)
                    except:
                        open_list[idx_nei] = f_new
                    parent_list[idx_nei] = idx_selected

    history_map = get_history(close_list, H, W)
    path_map = backtrack(parent_list, goal_idx, H, W)

    return history_map, path_map