import abc
from collections import Counter
from typing import Sequence, Any

import ase
import numpy as np

import json

def read_json(json_path: str):
    with open(json_path, encoding="utf-8", mode="r") as fp:
        return json.load(fp)


def save_json(json_dict: dict, json_path: str):
    def _fix_dict():
        for key in json_dict:
            if isinstance(json_dict[key], np.ndarray):
                json_dict[key] = json_dict[key].tolist()

    _fix_dict()
    with open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp, indent=4)

def discrete_histogram(values: Sequence[Any], encoder: dict[Any, int], norm: bool = False) -> np.ndarray:
    counter = Counter(values)
    histogram = np.zeros(max(encoder.values()) + 1)
    for key in counter:
        histogram[encoder[key]] = counter[key]

    if norm:
        histogram /= np.sum(histogram)

    return histogram


class Metrics(abc.ABC):
    def __call__(self, atoms: list[ase.Atoms] | ase.Atoms):
        return self.update(atoms)

    def update(self, atoms: list[ase.Atoms] | ase.Atoms):
        raise NotImplementedError()

    def summarize(self) -> dict:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class ContextMetrics(abc.ABC):
    def update(self, atoms: list[ase.Atoms] | ase.Atoms, targets: np.ndarray):
        raise NotImplementedError()

    def summarize(self) -> dict:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
