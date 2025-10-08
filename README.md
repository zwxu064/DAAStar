# **DAA\*: Deep Angular A Star for Image-based Path Planning**

[![arXiv](https://img.shields.io/badge/ICCV-2025-oliver.svg)](https://www.arxiv.org/abs/2507.09305)
[![video](https://img.shields.io/badge/YouTube-DAA*-8b0000.svg)](https://www.youtube.com/watch?v=_DbkEj3yjoI)
[![poster](https://img.shields.io/badge/Poster-DAA*-magenta.svg)](assets/poster.png)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

This is the **official** implementation of [deep angular A star (DAA*)](https://www.arxiv.org/abs/2507.09305).\
We have made things easy with user-friendly source files for demonstration, evaluation, and training.\
Please read this document carefully for the instructions.

## Environment
Build the conda environment with pip.\
All our GPU-related experiments were conducted on a single 3090 GPU (24GB).
```Python
conda create -n DAAStar python==3.8.0
source activate DAAStar
pip install -r requirements.txt
```

**Note**
  1. If you encounter issues installing pytorch3d, please comment out Line 23 in requirements.txt, re-run the "pip install" command above, then uncomment this line and re-run the "pip install" again.\
    Installing pytorch3d may take a while; however, it often works if you succeed at installing it in another way.

## Datasets
- Motion planning dataset (MPD)
- Tiled MPD (TMPD)
- City and street maps (CSM)
- Augmented TMPD (Aug-TMPD)
- Warcraft
- Pokémon
- SDD-intra
- SDD-inter

**Note**
  1. MPD, TMPD, CSM, SDD-intra, and SDD-inter can be downloaded from the repository of [neural A*](https://github.com/omron-sinicx/neural-astar.git).
  2. [Aug-TMPD](https://disk.360.yandex.ru/d/xLeW_jrUpTVnCA) can be downloaded from the repository of [TransPath](https://github.com/AIRI-Institute/TransPath.git).\
 Unzip it, rename it as "aug_tmpd", and move it to ./datasets.\
    **Ensure** the directory structure is "aug_tmpd/{split}" where "split" is train/val/test.
  3. Warcraft and Pokémon can be downloaded from [tile-based navigation datasets](https://github.com/archettialberto/tilebased_navigation_datasets.git).
  4. The reference paths for MPD, TMPD, and CSM are computed on the fly using empirical policy because the start nodes are randomly sampled during training.\
 The reference paths for Aug-TMPD, Warcraft, and Pokémon are computed using Dijkstra's algorithm given the reference cost maps (equivalent to PPMs).\
 The reference paths for SDD-intra and SDD-inter are provided by human labelling in the datasets.

**Download**
  - For the ease of use, users can download the datasets once from [DAAStar/datasets](https://drive.google.com/drive/folders/1sT5UhJ0QnuYPqlA2UfgSBjPtBFPS_Tr8?usp=sharing) (~2.3GB), *excluding* Aug-TMPD as it is large.\
 Please download Aug-TMPD according to Note 2 above, particularly the train and validation sets.
  - Meanwhile, for Aug-TMPD, we use Dijkstra's algorithm from [motion planning library in Python](https://github.com/ai-winter/python_motion_planning.git) to compute the reference paths.\
 We provide the pre-computed paths, named as *dijkstra.npy*, for the test set.\
 For the train and validation sets used in model training, please compute the paths as follows, giving *datasets/train/dijkstra_part{ID}.npy* and *datasets/val/dijkstra.npy*.\
 These paths move diagonally at the obstacle's corners as we slightly modified the function *self.isCollision(...)* for the model training.
    ```python
    cd third_party/python_motion_planning
    python example/global_example.py  # ensure the dataset directory is correct; visualization is in datasets/aug_tmpd/{split}/viz
    ```
  - The folder of datasets should be structured as below for training and evaluation automation, see [datasets.txt](./assets/datasets.txt).
    ```
    datasets
    ├── aug_tmpd
    │   ├── test
    │   │   ├── focal.npy
    │   │   ├── goals.npy
    │   │   ├── maps.npy
    │   │   ├── starts.npy
    |   │   └── dijkstra.npy
    |   ├── train
    |   │   ├── focal.npy
    ...
    ├── mpd
    │   ├── alternating_gaps_032_moore_c8.npz
    │   ├── bugtrap_forest_032_moore_c8.npz
    │   ├── forest_032_moore_c8.npz
    │   ├── gaps_and_forest_032_moore_c8.npz
    │   ├── mazes_032_moore_c8.npz
    │   ├── multiple_bugtraps_032_moore_c8.npz
    │   ├── shifting_gaps_032_moore_c8.npz
    │   └── single_bugtrap_032_moore_c8.npz
    ├── pkmn
    │   ├── test.npz
    │   ├── train.npz
    │   └── val.npz
    ├── sdd
    │   └── s064_0.5_128_300
    │       ├── bookstore
    │       │   ├── video0
    │       │   │   ├── 00000000_01.npz
    │       │   │   ├── 00000001_01.npz
    ...
    ├── street
    │   └── mixed_064_moore_c16.npz
    ├── tmpd
    │   └── all_064_moore_c16.npz
    └── warcraft
        ├── test.npz
        ├── train.npz
        └── val.npz
    ```

## Algorithms
- [A*](https://github.com/ai-winter/python_motion_planning.git)
- [Theta*](https://github.com/ai-winter/python_motion_planning.git)
- [Neural A*](https://github.com/omron-sinicx/neural-astar.git)
- [TransPath](https://github.com/AIRI-Institute/TransPath.git) (only for Aug-TMPD)
- Randomwalk-*k* (typically *k*=3)
- DAA* series (ours)
  - DAA*-min ($\alpha=1$ in Eq.(5))
  - DAA*-max ($\alpha=0$ in Eq.(5))
  - DAA*-mix ($\alpha$ in Eq.(5) is learned)
  - DAA*-path (only for Aug-TMPD)
  - DAA*-mask (only for Aug-TMPD)
  - DAA*-weight (only for Aug-TMPD)

**Note**
  - For conventional methods, including Dijkstra's algorithm, A*, and Theta*, we use [motion planning library in Python](https://github.com/ai-winter/python_motion_planning.git) to compute their paths.
  - We also use Dijkstra's algorithm from this library to compute the reference paths for Aug-TMPD for supervised learning.

**Download**
  - Alternatively, users can download all the model weights once from [DAAStar/model_weights](https://drive.google.com/drive/folders/1skUyLfS2U3v1FsCj3cgwg5DopQNEs_fX?usp=sharing), which provides *single-seed* models and takes ~8.3GB.\
 Since the numbers in our paper were averaged over 3 seeds except Aug-TMPD, it is reasonable to have marginal evaluation gaps over the single seed.
 - The folder of model weights should be structured as below for automatically evaluating the models, see [model_weight.txt](./assets/model_weights.txt).\
 This also aligns with the wandb checkpoint configuration during the training.
    ```
    model_weights
    ├── seed0
    │   ├── daa_mask
    │   │   └── aug_tmpd
    │   │       └── transpath
    │   │           └── Unet
    │   │               └── Enconst10.000_pid180960
    │   │                   └── DAA_aug_tmpd
    │   │                       └── 6j43bvj6
    │   │                           └── checkpoints
    │   │                               └── epoch=45-step=354752.ckpt
    │   ├── daa_max
    │   │   ├── aug_tmpd
    │   │   │   └── transpath
    │   │   │       └── Unet
    │   │   │           └── Enconst10.000_pid74747
    │   │   │               └── DAA_aug_tmpd
    │   │   │                   └── 5elzupfu
    │   │   │                       └── checkpoints
    │   │   │                           └── epoch=43-step=339328.ckpt
    │   │   ├── mpd
    │   │   │   ├── alternating_gaps_032_moore_c8
    │   │   │   │   └── Unet
    │   │   │   │       └── Enconst1.000_pid2105007
    │   │   │   │           └── DAA_mpd
    │   │   │   │               └── yg9k5st2
    │   │   │   │                   └── checkpoints
    │   │   │   │                       └── epoch=112-step=1469.ckpt
    ...
    └── transpath
        └── weights
            └── focal.pth
    ```

## Demonstration
Please refer to [demo.ipynb](demo.ipynb) where the dataset, denoted as *dataset_name*, can be changed to visualise all the results of the methods involved.

## Evaluation
For a specific dataset and method, set the dataset (*dataset_name*) and method (*method*) in [config.yaml](config.yaml), then run below.\
To evaluate **all datasets and methods** automatically, set *dataset_name: null* and *method: null* in [config.yaml](config.yaml).
```python
CUDA_VISIBLE_DEVICES="0" python eval.py
```
To get the averaged metrics over **multiple** seeds, run
```python
python parse_seeds.py
```

**Note**
  1. The area similarities (ASIMs) reported in our paper are *relative* values that exclude common areas in the path maps, which are shared by all methods.\
    Thus, all methods should be evaluated to compute these common areas by running [eval.py](eval.py) with *enable_save_path_stack=True* (which will be very slow) and then [compute_hmean.py](compute_hmean.py).\
    Alternatively and preferrly, one can simply use the *absolute* values by running [eval.py](eval.py) only with *enable_save_path_stack=False* as default.\
    These ASIMs are generally higher than the relative value because the shared areas, that is the areas excluding the interaction of the predicted and reference paths, are often large.

## Training
Set the dataset (*dataset_name*) and method (*method*) in [config.yaml](config.yaml), then run below.\
The training process will be recorded in wandb.
```python
CUDA_VISIBLE_DEVICES="0" python train.py
```

## Reference
If you find this paper or code useful, please cite our work as follows,
```
@inproceedings{Xu:ICCV25,
 author    = {Zhiwei Xu},
 title     = {DAA*: Deep Angular A Star for Image-based Path Planning},
 booktitle = {International Conference on Computer Vision},
 year      = {2025}
}
```
This code is built on [neural A*](https://github.com/omron-sinicx/neural-astar.git), [TransPath](https://github.com/AIRI-Institute/TransPath.git), and [motion planning library](https://github.com/ai-winter/python_motion_planning.git).\
If you use parts of their code from either the official repositories or our repository, please also cite them appropriately.

## License
This code is distributed under the MIT License.

## Contact
For any enquiries, please contact [Danny Xu](mailto:zwxu064@gmail.com) or open an issue.
