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
        if not cfg.get("ckpt_path"):
            raise ValueError(
                "only_load_weights=true requires ckpt_path to a previous "
                "stage checkpoint"
            )
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        incompatible = lit_module.load_state_dict(
            checkpoint["state_dict"],
            strict=cfg.get("load_weights_strict", True),
        )
        if not cfg.get("load_weights_strict", True):
            print(
                "Loaded checkpoint weights non-strictly: "
                f"{len(incompatible.missing_keys)} missing keys, "
                f"{len(incompatible.unexpected_keys)} unexpected keys"
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
