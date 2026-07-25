from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch


@hydra.main(config_path="../configs", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    lit_module = hydra.utils.instantiate(cfg.lit_module)
    callbacks = [
        hydra.utils.instantiate(callback)
        for callback in cfg.get("callbacks", {}).values()
    ]
    logger = (
        [hydra.utils.instantiate(logger) for logger in cfg.get("logger", {}).values()]
        if cfg.get("logger")
        else False
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    if cfg.get("only_load_weights", False):
        lit_module.load_state_dict(
            torch.load(cfg.ckpt_path)["state_dict"]
        )
        trainer.fit(lit_module, datamodule=datamodule)
    else:
        trainer.fit(lit_module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if cfg.get("test", False):
        trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")
    metric = cfg.get("optimized_metric")
    if metric is None:
        return None
    value = trainer.callback_metrics.get(metric)
    return float(value) if value is not None else None


if __name__ == "__main__":
    main()

