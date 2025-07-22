# -------------------------------------------------------------------------------------------------------------
# File: viz.py
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

import torch, PIL
import matplotlib.pyplot as plt

from torchvision.transforms import Resize as tresize
from matplotlib.patches import Circle, Rectangle

# =============================================================================================================

def get_method_from_path(
    image_dir,
    return_viz_method=False,
    is_reference=False
):
    seed = None
    image_dir_split = image_dir.split('/')

    for v in image_dir_split:
        if v.find('seed') > -1:
            seed = v.replace('seed', '')

    method = image_dir.split(f"seed{seed}/")[-1].split('/')[0]
    viz_method = method

    if image_dir.find('neural_astar/') > -1:
        viz_method = r"Neural A$^\ast$"
    elif image_dir.find('randomwalk_3/') > -1:
        viz_method = 'Random-walk'
    elif image_dir.find('daa_mix/') > -1:
        viz_method = r"DAA$^\ast$"
    elif image_dir.find('daa_min/') > -1:
        viz_method = r"DAA$^\ast$-min"
    elif image_dir.find('daa_max/') > -1:
        viz_method = r"DAA$^\ast$-max"

    if image_dir.find('aug_tmpd') > -1:
        if image_dir.find('daa_path') > -1:
            method = 'daa_path'
            viz_method = r"DAA$^\ast$-path"
        elif image_dir.find('daa_weight') > -1:
            method = 'daa_weight'
            viz_method = r"DAA$^\ast$-weight"
        elif image_dir.find('daa_mask') > -1:
            method = 'daa_mask'
            viz_method = r"DAA$^\ast$-mask"
        else:
            method = 'daa'
            viz_method = r"DAA$^\ast$"

    if image_dir.find('transpath/weights') > -1:
        method = 'transpath'
        viz_method = 'TransPath'

    if viz_method == r"DAA$^\ast$-weight": viz_method = r"DAA$^\ast$"
    if is_reference: viz_method = 'Reference'

    if return_viz_method:
        return method, viz_method
    else:
        return method

# =============================================================================================================

def add_start_end_mark(
    axes,
    start_map,
    goal_map,
    left_viz_string=None,
    right_viz_string=None,
    fontsize=10,
    zorder=None
):
    axes.set_aspect('equal')

    # Use loop because for warcraft and pokemon, the image size and grid size are different
    # For visualization, one needs resize the grid,
    # so does the start and target maps with one pixel enlarge to several
    corr_starts = torch.nonzero(start_map == 1).reshape(-1, 2)
    corr_ends = torch.nonzero(goal_map == 1).reshape(-1, 2) - 1

    for i in range(corr_starts.shape[0]):
        y_start = corr_starts[i, 0]
        x_start = corr_starts[i, 1]

        y_end = corr_ends[i, 0]
        x_end = corr_ends[i, 1]

        if zorder is None:
            axes.add_patch(Circle(
                (x_start, y_start),
                radius=2,
                color='red'
            ))
            axes.add_patch(Rectangle(
                (x_end.cpu(), y_end.cpu()),
                width=3,
                height=3,
                color='violet'
            ))
        else:
            axes.add_patch(Circle(
                (x_start, y_start),
                radius=2,
                color='red',
                zorder=zorder
            ))
            axes.add_patch(Rectangle(
                (x_end.cpu(), y_end.cpu()),
                width=3,
                height=3,
                color='violet',
                zorder=zorder
            ))

    if left_viz_string is not None:
        axes.text(
            0.01, 0.01, left_viz_string, size=fontsize, horizontalalignment='left',
            verticalalignment='bottom', transform=axes.transAxes, color='white',
            weight='bold'
        )

    if right_viz_string is not None:
        axes.text(
            1.0, 1.0, right_viz_string, size=fontsize, horizontalalignment='right',
            verticalalignment='top', transform=axes.transAxes, color='white',
            weight='bold'
        )

# =============================================================================================================

def change_input_color(
    inputs,
    color='gray'
):
    # All batch*c*h*w
    outputs = (inputs * 1.0).permute(0, 2, 3, 1)
    c = outputs.shape[-1]

    if c == 3: return outputs.permute(0, 3, 1, 2)
    if c == 1: outputs = outputs.repeat(1, 1, 1, 3)

    # Gray
    if color == 'gray':
        outputs[outputs[:, :, :, 0] > 0] = torch.tensor(
            [150, 150, 150], dtype=inputs.dtype, device=inputs.device) / 255
    else:
        outputs[outputs[:, :, :, 0] > 0] = torch.tensor(
            [255, 255, 255], dtype=inputs.dtype, device=inputs.device) / 255

    return outputs.permute(0, 3, 1, 2)

# =============================================================================================================

def add_path(
    inputs,
    paths,
    mode='optimal'
):
    assert mode in ['optimal', 'hist']

    if mode == 'optimal':
        color_map = [1, 1, 0]
    elif mode == 'hist':
        color_map = [0, 1, 0]

    # All batch*c*h*w
    outputs = (inputs * 1.0).permute(0, 2, 3, 1)
    if len(paths.shape) == 4: paths = paths.squeeze(1)

    c = outputs.shape[-1]
    if c == 1: outputs = outputs.repeat(1, 1, 1, 3)

    # Yellow
    outputs[paths > 0] = torch.tensor(
        color_map,
        dtype=inputs.dtype,
        device=inputs.device
    )

    return outputs.permute(0, 3, 1, 2)

# =============================================================================================================

def save_figure_mixed(
    map,
    start,
    goal,
    save_path,
    path=None,
    path_angle=None,
    hist=None,
    cost_map=None,
    is_reference=False
):
    img_h, img_w = map.shape[-2:]
    h, w = start.shape[-2:]

    if path is not None: path_length = path.sum()

    # Move to cpu
    map = map.cpu()
    start = start.cpu()
    goal = goal.cpu()
    if path is not None: path = path.cpu()
    if hist is not None: hist = hist.cpu()
    if cost_map is not None: cost_map = cost_map.cpu()

    # For that is downsized so that the map has a different image size with path
    if img_h != h or img_w != w:
        resize_obj = tresize((img_h, img_w), interpolation=PIL.Image.NEAREST)
        start = resize_obj(start)
        goal = resize_obj(goal)
        if path is not None: path = resize_obj(path)
        if hist is not None: hist = resize_obj(hist)
        if cost_map is not None: cost_map = resize_obj(cost_map)

    # Get fixed viz colors
    map = change_input_color(map)

    if False and hist is not None: map = add_path(map, hist, mode='hist')
    if path is not None: map = add_path(map, path, mode='optimal')

    # Save path
    fig = plt.figure()
    axes_sub = fig.add_subplot(1, 1, 1)
    viz_method_string = get_method_from_path(
        save_path,
        return_viz_method=True,
        is_reference=is_reference
    )[1]
    viz_length_string = None

    if path is not None:
        if False:
            viz_length_string = f'({int(path_length)}'

            if path_angle is not None:
                viz_length_string += f', {path_angle:.1f})'
        else:
            viz_length_string = f'{int(path_length)}'

    map = map[0].permute(1, 2, 0)
    start = start[0, 0]
    goal = goal[0, 0]

    add_start_end_mark(
        axes_sub,
        start,
        goal,
        left_viz_string=viz_length_string,
        right_viz_string=viz_method_string,
        fontsize=35
    )

    axes_sub.imshow(map.numpy())
    axes_sub.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save cost maps
    if cost_map is not None:
        save_path = save_path.replace('.png', '_cost.png')
        fig = plt.figure()
        axes_sub = fig.add_subplot(1, 1, 1)
        axes_sub.imshow(cost_map[0, 0].numpy())
        if cost_map.min() != cost_map.max():
            axes_sub.text(
                0.99,
                0.99,
                viz_method_string,
                size=35,
                horizontalalignment='right',
                verticalalignment='top',
                transform=axes_sub.transAxes,
                color='red',
                weight='bold'
            )
        axes_sub.axis('off')
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
