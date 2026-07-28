"""Evaluate a 3D2Shift checkpoint on one dataset split."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl

from .data import Shift3DDataModule
from .lit_module import Shift3DModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datamodule = Shift3DDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = Shift3DModule.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
    )
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
