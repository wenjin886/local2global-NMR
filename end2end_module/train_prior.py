"""Hydra entry point for frozen-NMR, prior-only graph correction."""

from __future__ import annotations

from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_prior_only",
)
def main(config: DictConfig) -> Optional[float]:
    pl.seed_everything(int(config.seed), workers=True)
    print(OmegaConf.to_yaml(config, resolve=True))
    datamodule = hydra.utils.instantiate(config.datamodule)
    module = hydra.utils.instantiate(config.lit_module)
    callbacks = [
        hydra.utils.instantiate(callback)
        for callback in config.get("callbacks", {}).values()
    ]
    loggers = (
        [
            hydra.utils.instantiate(logger)
            for logger in config.get("logger", {}).values()
        ]
        if config.get("logger")
        else False
    )
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))
    if bool(config.get("test", False)):
        trainer.test(module, datamodule=datamodule, ckpt_path="best")
    metric = config.get("optimized_metric")
    value = trainer.callback_metrics.get(metric) if metric else None
    return float(value) if value is not None else None


if __name__ == "__main__":
    main()
