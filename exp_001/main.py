from omegaconf import DictConfig
import hydra
import pandas as pd

from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import wandb


@hydra.main(config_path='.', config_name='config', )
def main(cfg: DictConfig):
    sys.path.append(os.path.join('.', cfg.path.src_dir, cfg.wandb.exp_name))
    from models import QAModel
    from lightning_datamodule import FAQDataModule

    output_path = os.path.join(os.getcwd(), cfg.path.checkpoint_path)
    target_dir = os.path.join(
        cfg.path.project_path, 'data', cfg.path.client_name, 'target')

    target_csv = os.listdir(target_dir)[0]

    targets = pd.read_csv(target_csv)[
        cfg.data.answer_column].tolist()
    data_module = FAQDataModule(cfg, targets)
    model = QAModel(cfg, targets)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch}',
        verbose=True,
        monitor='val_loss',
        save_last=True,
        mode='min',
    )
    wb_logger = pl_loggers.WandbLogger(
        name=cfg.wandb.exp_name, project=cfg.wandb.project)
    wb_logger.log_hyperparams(cfg)

    trainer = pl.Trainer(max_epochs=cfg.training.num_epochs,
                         gpus=1,
                         progress_bar_refresh_rate=5,
                         callbacks=[checkpoint_callback],
                         logger=wb_logger)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
