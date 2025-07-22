# -------------------------------------------------------------------------------------------------------------
# File: sdd.py
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

import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset

# =============================================================================================================

class SDD(Dataset):
    def __init__(
        self,
        dirname,
        test_scene='gates',
        is_train=True,
        load_hardness=False,
        load_label=False
    ):
        self.data = sorted(glob('%s/*/*/*.npz' % dirname))

        if is_train:
            self.data = [x for x in self.data if test_scene not in x]
        else:
            self.data = [x for x in self.data if test_scene in x]

        self.load_hardness = load_hardness
        self.load_label = load_label
        print(f'SSD test scene: {test_scene}.')

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx
    ):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = np.load(self.data[idx])

        if (self.load_hardness):
            return s['length_ratio']

        if (self.load_label):
            return s['label']

        sample = {
            'image': s['image'],
            'traj_image': s['traj_image'],
            'start_image': s['start_image'],
            'goal_image': s['goal_image'],
            'label': s['label'],
            'length_ratio': s['length_ratio']
            # 'traj': s['traj'],
        }

        sample = CustomToTensor()(sample)

        return {
            'maps': sample['image'],
            'starts': sample['start_image'],
            'goals': sample['goal_image'],
            'optimal_paths': sample['traj_image']
        }

# =============================================================================================================

class CustomToTensor(object):
    def __call__(
        self,
        sample
    ):
        image, traj_image, start_image, goal_image, label, length_ratio = \
            sample['image'], sample['traj_image'], sample['start_image'], \
            sample['goal_image'], sample['label'], sample['length_ratio']
        image = torch.from_numpy(np.array(image)).type(torch.float32).permute(
            2, 0, 1) / 255.
        traj_image = torch.from_numpy(traj_image).type(torch.float32)
        start_image = torch.from_numpy(start_image).type(torch.float32)
        goal_image = torch.from_numpy(goal_image).type(torch.float32)

        return {
            'image': image,
            'traj_image': traj_image.unsqueeze(0),
            'start_image': start_image.unsqueeze(0),
            'goal_image': goal_image.unsqueeze(0),
            'label': label,
            'length_ratio': length_ratio,
        }

# =============================================================================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader, Subset
    import matplotlib.pyplot as plt

    data_dir = 'datasets/sdd/s064_0.5_128_300'
    test_scene = 'video0'
    batch_size = 64
    hardness_factor = 1.0

    hardness = SDD(
        data_dir,
        is_train=True,
        test_scene=test_scene,
        load_hardness=True
    )

    hardness = np.array([x for x in hardness])

    train_dataset = SDD(
        data_dir,
        is_train=True,
        test_scene=test_scene,
        load_hardness=False
    )

    train_dataset = Subset(
        train_dataset,
        np.where(hardness <= float(hardness_factor))[0]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0
    )

    test_dataset = SDD(
        data_dir,
        is_train=False,
        test_scene=test_scene,
        load_hardness=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0
    )

    for idx, data in enumerate(test_loader):
        if idx < 10:
            print(
                data['image'].shape,
                data['traj_image'].shape,
                data['start_image'].shape,
                data['goal_image'].shape,
                data['label'].shape,
                data['length_ratio'].shape
            )

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((
                data['image'][0]+ data['traj_image'][0] \
                + data['start_image'][0] \
                + data['goal_image'][0]
            ).permute(1, 2, 0) .cpu().numpy())
            plt.subplot(2, 2, 2)
            plt.imshow(data['traj_image'][0, 0].cpu().numpy())
            plt.subplot(2, 2, 3)
            plt.imshow(data['start_image'][0, 0].cpu().numpy())
            plt.subplot(2, 2, 4)
            plt.imshow(data['goal_image'][0, 0].cpu().numpy())
            plt.savefig(f'tmp/sdd_{idx}.png')
            print(data['label'][0], data['length_ratio'][0])