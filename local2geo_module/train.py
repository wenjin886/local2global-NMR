"""Hydra entry point for standalone 2D-only topology-prior training."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from .data import ParquetLocal2GeoDataModule
from .lit_module import HybridLocal2GeoModule


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="local2geo_hybrid",
)
def main(config: DictConfig) -> None:
    pl.seed_everything(int(config.seed), workers=True)
    print(OmegaConf.to_yaml(config, resolve=True))

    datamodule = ParquetLocal2GeoDataModule(
        **OmegaConf.to_container(
            config.data, resolve=True, throw_on_missing=True
        )
    )
    model = HybridLocal2GeoModule(
        **OmegaConf.to_container(
            config.model, resolve=True, throw_on_missing=True
        )
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
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

        try:
            logger = WandbLogger(
                project=str(config.logger.project),
                name=(
                    None
                    if config.logger.name is None
                    else str(config.logger.name)
                ),
                save_dir=str(output_dir),
                mode=str(config.logger.mode),
                log_model=bool(config.logger.log_model),
            )
            logger.log_hyperparams(
                OmegaConf.to_container(config, resolve=True)
            )
            callbacks.append(
                LearningRateMonitor(logging_interval="epoch")
            )
        except Exception as error:
            print(
                "W&B initialization failed; continuing without a logger: "
                f"{error}"
            )
            logger = False

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **OmegaConf.to_container(
            config.trainer, resolve=True, throw_on_missing=True
        ),
    )
    trainer.fit(model, datamodule=datamodule)
    if bool(config.run_test):
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
        )
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Last checkpoint: {checkpoint.last_model_path}")


if __name__ == "__main__":
    main()
