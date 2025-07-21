import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(os.path.join(__file__))}/../../..")

from third_party.python_motion_planning.utils import Grid, Map, SearchFactory


def single_process(start, goal, obstacle_map, method):
    search_factory = SearchFactory()

    h, w = start.shape

    start_padded = np.zeros((h+2, w+2))
    goal_padded = np.zeros((h+2, w+2))
    obstacle_map_padded = np.zeros((h+2, w+2))

    start_padded[1:-1, 1:-1] = start
    goal_padded[1:-1, 1:-1] = goal
    obstacle_map_padded[1:-1, 1:-1] = obstacle_map

    start_cor = np.nonzero(start_padded)
    goal_cor = np.nonzero(goal_padded)

    start = (start_cor[0].item(), start_cor[1].item())
    goal = (goal_cor[0].item(), goal_cor[1].item())

    env = Grid(h+2, w+2)
    obstacles = set()
    obstacle_cor = np.nonzero(obstacle_map_padded==0)

    for obs_x, obs_y in zip(obstacle_cor[0], obstacle_cor[1]):
        obstacles.add((obs_x, obs_y))

    env.update(obstacles)

    # creat planner
    # planner = search_factory("a_star", start=start, goal=goal, env=env)
    # planner = search_factory("dijkstra", start=start, goal=goal, env=env)
    # planner = search_factory("gbfs", start=start, goal=goal, env=env)
    # planner = search_factory("theta_star", start=start, goal=goal, env=env)
    # planner = search_factory("lazy_theta_star", start=start, goal=goal, env=env)
    # planner = search_factory("jps", start=start, goal=goal, env=env)
    # planner = search_factory("d_star", start=start, goal=goal, env=env)
    # planner = search_factory("lpa_star", start=start, goal=goal, env=env)
    # planner = search_factory("d_star_lite", start=start, goal=goal, env=env)
    # planner = search_factory("voronoi", start=start, goal=goal, env=env, n_knn=4, max_edge_len=10.0, inflation_r=1.0)
    planner = search_factory(method, start=start, goal=goal, env=env)

    # animation
    path_cor, history_cor = planner.run()

    def convert_cor2map(h, w, coordinates):
        path_map = np.zeros((h, w))
        path_coordinates = np.ones((h * w, 2), dtype=np.int32) * -1
        path_cor_x, path_cor_y = [], []

        for path_per in coordinates:
            path_map[path_per[0]-1, path_per[1]-1] = 1
            path_cor_y.append(path_per[0]-1)
            path_cor_x.append(path_per[1]-1)

        num_points = len(path_cor_x)
        path_coordinates[:num_points, 0] = path_cor_y
        path_coordinates[:num_points, 1] = path_cor_x

        return path_map, path_coordinates

    path_map, path_coordinates = convert_cor2map(h, w, path_cor)
    history_map, history_coordinates = convert_cor2map(h, w, history_cor)

    return path_map, history_map, path_coordinates, history_coordinates


if __name__ == '__main__':
    for split in ['test', 'val', 'train']:
        dataset_dir = f'../../datasets/aug_tmpd/{split}'
        assert os.path.exists(dataset_dir), f"!!!Error, {dataset_dir} not exist."
        save_dir = f'{dataset_dir}/viz'
        os.makedirs(save_dir, exist_ok=True)

        start_path = f'{dataset_dir}/starts.npy'
        goal_path = f'{dataset_dir}/goals.npy'
        map_path = f'{dataset_dir}/maps.npy'
        focal_path = f'{dataset_dir}/focal.npy'

        methods = ['dijkstra']

        for method in methods:
            starts = np.load(start_path)
            goals = np.load(goal_path)
            maps = np.load(map_path)
            focals = np.load(focal_path)

            num_samples, _, h, w = starts.shape
            count = 0  # manually set count, as automatic one will be killed, unknown
            paths = []  # np.zeros((num_samples, h, w), dtype=bool)

            for idx in range(num_samples):
                if split == 'train':
                    left = 50000 * count
                    right = min(left + 49999, num_samples - 1)

                    if not (idx >= left and idx <= right):
                        continue

                if idx % 100 == 0 or idx == num_samples - 1:
                    print(f"{split}: {idx} / {num_samples}")

                start_per = starts[idx][0]
                goal_per = goals[idx][0]
                map_per = maps[idx][0]
                focal_per = focals[idx][0]

                if False:
                    obstacle_map_per = 1 - map_per[0]
                else:  # 20240617 Use PPM to find path instead of map
                    obstacle_map_per = focal_per >= 0.95

                path_map, history_map, _, _ = single_process(start_per, goal_per, obstacle_map_per, method)
                paths.append(path_map)

                if split == 'train' and idx == right:
                    paths = np.stack(paths, axis=0)
                    np.save(f'{dataset_dir}/{method}_part{count}.npy', paths)
                    paths = []

                if idx <= 5:
                    plt.figure()
                    plt.subplot(2, 3, 1)
                    plt.imshow(map_per)
                    plt.subplot(2, 3, 2)
                    plt.imshow(start_per)
                    plt.subplot(2, 3, 3)
                    plt.imshow(goal_per)
                    plt.subplot(2, 3, 4)
                    plt.imshow(map_per+focal_per+path_map)
                    plt.subplot(2, 3, 5)
                    plt.imshow(focal_per >= 0.95)
                    plt.subplot(2, 3, 6)
                    plt.imshow(history_map)

                    plt.savefig(f'{save_dir}/{method}_{idx}.png')
                    plt.close()

            if split in ['val', 'test']:
                paths = np.stack(paths, axis=0)
                np.save(f'{dataset_dir}/{method}.npy', paths)