from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(cfg.lit_module)
    callbacks = [
        hydra.utils.instantiate(callback)
        for callback in cfg.get("callbacks", {}).values()
    ]
    loggers = (
        [hydra.utils.instantiate(logger) for logger in cfg.logger.values()]
        if cfg.get("logger") else False
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if cfg.get("test", False):
        trainer.test(module, datamodule=datamodule, ckpt_path="best")
    name = cfg.get("optimized_metric")
    value = trainer.callback_metrics.get(name) if name else None
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
