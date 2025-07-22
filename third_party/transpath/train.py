from models.autoencoder import Autoencoder, PathLogger, DemAutoencoder, DemPathLogger
from data.hmaps import GridData
from data.dems import DemData

import pytorch_lightning as pl
import wandb, random
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import argparse
import multiprocessing


def main(mode, run_name, proj_name, batch_size, max_epochs):
    train_data = GridData(
        path='./datasets/transpath/train',
        mode=mode
    ) if mode != 'dem' else DemData(filename='./datasets/transpath/dem_rgb/train')

    val_data = GridData(
        path='./datasets/transpath/val',
        mode=mode
    ) if mode != 'dem' else DemData(filename='./datasets/transpath/dem_rgb/val')

    resolution = (train_data.img_size, train_data.img_size)

    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # multiprocessing.cpu_count(),
        pin_memory=True
    )

    samples = next(iter(val_dataloader))

    model = Autoencoder(mode=mode, resolution=resolution) if mode != 'dem' else DemAutoencoder(resolution=resolution)
    callback = PathLogger(samples, mode=mode) if mode != 'dem' else DemPathLogger(samples)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'./checkpoints/{mode}',
        filename='{epoch:02d}-{val_loss:.4f}',
        mode='min',
        every_n_epochs=1,
        save_weights_only=True,
        save_top_k=-1,
    )

    wandb_logger = WandbLogger(project=proj_name, name=f'{run_name}_{mode}', log_model='all')

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=max_epochs,
        deterministic=False,
        callbacks=[callback, checkpoint_callback]
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()


def set_global_seeds(seed: int):
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf', 'dem'], default='dem')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--proj_name', type=str, default='TransPath_runs')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=160)

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') #fix for tesor blocks warning with new video card
    set_global_seeds(args.seed)
    main(
        mode=args.mode,
        run_name=args.run_name,
        proj_name=args.proj_name,
        batch_size=args.batch,
        max_epochs=args.epoch,
    )