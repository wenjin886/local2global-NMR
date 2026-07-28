"""Hydra entry point for standalone 3D2Shift training."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .data import Shift3DDataModule
from .lit_module import Shift3DModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="shift3d")
def main(config: DictConfig) -> None:
    pl.seed_everything(int(config.seed), workers=True)
    print(OmegaConf.to_yaml(config, resolve=True))
    datamodule = Shift3DDataModule(
        **OmegaConf.to_container(config.data, resolve=True, throw_on_missing=True)
    )
    model = Shift3DModule(
        **OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
    )
    output_dir = Path(config.output_dir)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=int(config.checkpoint.save_top_k),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint]
    logger = False
    if bool(config.logger.enabled):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=str(config.logger.project),
            name=None if config.logger.name is None else str(config.logger.name),
            save_dir=str(output_dir),
            mode=str(config.logger.mode),
            log_model=bool(config.logger.log_model),
        )
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **OmegaConf.to_container(
            config.trainer, resolve=True, throw_on_missing=True
        ),
    )
    trainer.fit(model, datamodule=datamodule)
    if bool(config.run_test):
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
