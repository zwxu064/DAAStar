from torch.utils.data import Dataset
import numpy as np
import os
from data_utils import resize_image


def sg2img_map(start, goal):
    img = np.stack((start, goal), axis=0)

    return img


def sg2img(start, goal, img_size=128):
    img = np.zeros((2, img_size, img_size))
    img[0, start[0], start[1]] = 1
    img[1, goal[0], goal[1]] = 1

    return img.astype(np.float32)


class DemData(Dataset):
    def __init__(self, filename='train', enable_use_map=False, image_size=None, num_augs=None):
        if image_size is None: image_size = 128
        data = np.load(filename + '.npz')
        data_focal = np.load(filename + '_focal.npz')

        # NOTE
        if num_augs is None:
            if filename.find('/train') > -1:
                self.num_augs = 1
            else:
                self.num_augs = 1
        else:
            self.num_augs = num_augs

        self.starts = data_focal['start'][:, :self.num_augs]
        self.goals = data_focal['goal'][:, :self.num_augs]
        self.rgb = data['rgb'].astype(np.float32) / 255.
        dems = data['dem'].astype(np.float32)
        focal = data_focal['focal'][:, :self.num_augs].astype(np.float32)
        org_img_size = self.rgb.shape[-1]

        if org_img_size != image_size:
            self.rgb, dems, focal = resize_image([self.rgb, dems, focal], image_size)
            self.starts, self.goals = resize_image([self.starts, self.goals], image_size, mode='coordinate')

        self.img_size = self.rgb.shape[-1]
        num_samples, _, h, w = self.rgb.shape
        self.enable_use_map = enable_use_map

        self.dems = dems
        self.focal = focal

        if enable_use_map:
            starts = self.starts.reshape(num_samples * self.num_augs, 2).astype(np.int64)
            start_maps = np.zeros((num_samples * self.num_augs, h, w)).astype(np.float32)
            start_maps[np.arange(num_samples * self.num_augs), starts[:, 0], starts[:, 1]] = 1

            goals = self.goals.reshape(num_samples * self.num_augs, 2).astype(np.int64)
            goal_maps = np.zeros((num_samples * self.num_augs, h, w)).astype(np.float32)
            goal_maps[np.arange(num_samples * self.num_augs), goals[:, 0], goals[:, 1]] = 1

            self.starts = start_maps.reshape(num_samples, self.num_augs, h, w)
            self.goals = goal_maps.reshape(num_samples, self.num_augs, h, w)

        self.gt_paths = None
        self.trans_focal_paths = None

        ppm_path_filename_list_dict = {
            'train': [
                'dem_part0.npz',
                'dem_part1.npz',
                'dem_part2.npz',
                'dem_part3.npz'
            ],
            'val': ['dem.npz'],
            'test': ['dem.npz']
        }

        ppm_path_filename_list = ppm_path_filename_list_dict[filename.split('/')[-1]]
        ppm_path_filename_list = [f'{filename}_{v}' for v in ppm_path_filename_list]

        if image_size != 128:
            postfix = f'_{image_size}'
        else:
            postfix = None

        if postfix is not None: ppm_path_filename_list = [v.replace('.npz', f'{postfix}.npz') for v in ppm_path_filename_list]
        gt_paths = []

        for ppm_path_filename in ppm_path_filename_list:
            if not os.path.exists(ppm_path_filename): continue

            data_ppm = np.load(ppm_path_filename)
            gt_paths.append(data_ppm['gt_dem_paths'])
            print(ppm_path_filename, gt_paths[-1].shape)

            # if enable_trained_ppm:
                # {'gt_dem_paths': gt_dem_path_maps, 'predicted_dem_paths': pred_dem_path_maps, 'predicted_focal_paths': pred_focal_path_maps}
                # 'predicted_dem': predictions_dem, 'predicted_focal': predictions_focal

                # self.dems = data_ppm['predicted_dem'].astype(np.float32)
                # self.focal = data_ppm['predicted_focal'].astype(np.float32)

        if len(gt_paths) > 0:
            gt_paths = np.concatenate(gt_paths, axis=0)
            gt_paths = gt_paths.reshape(num_samples, -1, h, w)[:, :self.num_augs]
            self.gt_paths = gt_paths.astype(np.float32)
    
    def __len__(self):
        return len(self.dems) * self.num_augs
    
    def __getitem__(self, idx):
        map_idx, task_idx = idx // self.num_augs, idx % self.num_augs
        rgb = self.rgb[map_idx]
        start = self.starts[map_idx][task_idx]
        goal = self.goals[map_idx][task_idx]
        focal = self.focal[map_idx][task_idx]
        dem = self.dems[map_idx]

        _, h, w = rgb.shape

        if len(dem.shape) != 3:
            dem = dem[task_idx].reshape(1, h, w)

        dem = dem - dem.min()
        dem = dem / dem.max()

        if self.enable_use_map:
            sg = sg2img_map(start, goal)
            start = start.reshape(1, h, w)
            goal = goal.reshape(1, h, w)
        else:
            sg = sg2img(start, goal, img_size=self.img_size)

        if self.gt_paths is None:
            gt_path = -1
        else:
            gt_path = self.gt_paths[map_idx][task_idx].reshape(1, h, w)

        return {
            'dem': dem,
            'maps': rgb,
            'rgb': rgb,
            'starts': start,
            'goals': goal,
            'sg': sg,
            'ppm': 1 - dem,  # focal.reshape(1, h, w),  # the calculation of focal a bit makes no sense
            'optimal_paths': gt_path,
        }

        # return dem, rgb, sg, focal
