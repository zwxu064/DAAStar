# -------------------------------------------------------------------------------------------------------------
# File: motion_planning_wrapper.py
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

import os, sys, torch, time
import numpy as np

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from multiprocessing import Pool
from src.planner.astar import VanillaAstar
from third_party.python_motion_planning.example.global_example import single_process
from src.planner.differentiable_astar import AstarOutput
from src.utils.metrics import generate_path_stack, cal_path_angle
from bresenham import bresenham

# =============================================================================================================

def motion_planning_thread(
    idx,
    start,
    goal,
    obstacle_map,
    method
):
    path_map, history_map, path_stack, _ = single_process(
        start,
        goal,
        obstacle_map,
        method
    )

    if method == 'theta_star':
        # Find path segments between each two corners.
        num_points = (path_stack.sum(1) >= 0).sum()

        for idx_point in range(num_points - 1):
            start_point = path_stack[idx_point]
            end_point = path_stack[idx_point + 1]
            segment_points = list(bresenham(start_point[0], start_point[1], end_point[0], end_point[1]))
            segment_point_x = [v[0] for v in segment_points]
            segment_point_y = [v[1] for v in segment_points]
            path_map[segment_point_x, segment_point_y] = 1

    return idx, path_map, history_map

# =============================================================================================================

class MotionPlanning(VanillaAstar):
    def __init__(
        self,
        config,
        *args,
        **kwargs
    ):
        super().__init__(
            config,
            *args,
            **kwargs
        )
        self.config = config
        self.method = self.config.motion_planning_lib.method
        self.init_path_angles = torch.tensor([-1], dtype=torch.float32)

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        prob_maps: torch.tensor = None,
        store_intermediate_results: bool = False,
        store_hist_coordinates: bool = False,
        disable_heuristic: bool = False,
        gt_paths = None,
        disable_compute_path_angle=False
    ):
        _, obstacle_maps, _ = self.data_preprocessing(map_designs, start_maps, goal_maps)

        device = start_maps.device
        num_samples, _, h, w = start_maps.shape
        path_maps = np.zeros((num_samples, 1, h, w), dtype=np.float32)
        history_maps = np.zeros((num_samples, 1, h, w), dtype=np.float32)

        start_maps_np = start_maps.cpu().numpy()
        goal_maps_np = goal_maps.cpu().numpy()
        obstacle_maps_np = obstacle_maps.cpu().numpy()

        max_processes = 20
        time_start = time.time()

        for i in range(0, num_samples, max_processes):
            x1 = i
            x2 = min(i + max_processes, num_samples)
            processes = []

            with Pool(processes=max_processes) as pool:
                for j in range(x1, x2):
                    process = pool.apply_async(
                        func=motion_planning_thread,
                        args=(
                            j,
                            start_maps_np[j, 0],
                            goal_maps_np[j, 0],
                            obstacle_maps_np[j, 0],
                            self.method
                        )
                    )

                    processes.append(process)

                pool.close()
                pool.join()

            for process in processes:
                idx, path_map, history_map = process.get()
                path_maps[idx, 0] = path_map
                history_maps[idx, 0] = history_map

        duration = time.time() - time_start
        print(f"Motion planning {self.method}: {duration:.2f}s.")

        path_maps = torch.from_numpy(path_maps).to(device)
        history_maps = torch.from_numpy(history_maps).to(device)
        cost_maps = 1 - map_designs #.new_zeros((num_samples, 1, h, w))
        path_stacks = generate_path_stack(path_maps, goal_maps)

        if disable_compute_path_angle:
            if self.init_path_angles.device != start_maps.device:
                self.init_path_angles = self.init_path_angles.to(start_maps.device)

            path_angles = self.init_path_angles
        else:
            path_angles = cal_path_angle(path_stacks)

        results = AstarOutput(
            history_maps,
            path_maps,
            None,
            cost_maps,
            obstacle_maps,
            None,
            path_stacks,
            path_angles,
            None,
            None,
            None
        )

        return results, None