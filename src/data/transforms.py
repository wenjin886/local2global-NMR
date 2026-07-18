import json

import torch
try:
    from torch_geometric.data import Data
    from torch_geometric.data.datapipes import functional_transform
    from torch_geometric.transforms import BaseTransform
    from torch_geometric.utils import one_hot, dense_to_sparse
except ImportError:  # NMR-only tests and lightweight inference do not need PyG.
    Data = object

    def functional_transform(_name):
        return lambda cls: cls

    class BaseTransform:
        def __call__(self, data):
            return self.forward(data)

    def one_hot(value, num_classes):
        return torch.nn.functional.one_hot(value, num_classes=num_classes)

    def dense_to_sparse(matrix):
        indices = matrix.nonzero(as_tuple=False).t().contiguous()
        return indices, matrix[indices[0], indices[1]]



def center(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


class NormalizeNMR:
    """Apply train-set z-score statistics while preserving missing values.

    Continuous values are normalized in place. Availability masks ensure that
    a missing integration or padded J value remains exactly zero.
    """

    def __init__(
            self,
            stats_path: str,
            normalize_h_shift: bool = True,
            normalize_c_shift: bool = True,
            normalize_h_integration: bool = True,
            normalize_h_j: bool = True,
            eps: float = 1e-6,
    ):
        with open(stats_path, encoding="utf-8") as handle:
            self.stats = json.load(handle)
        self.enabled = {
            "h_nmr": normalize_h_shift,
            "c_nmr": normalize_c_shift,
            "h_nmr_integration": normalize_h_integration,
            "h_nmr_j": normalize_h_j,
        }
        self.stat_names = {
            "h_nmr": "hnmr_shift",
            "c_nmr": "cnmr_shift",
            "h_nmr_integration": "hnmr_integration",
            "h_nmr_j": "hnmr_j",
        }
        self.masks = {
            "h_nmr_integration": "h_nmr_integration_mask",
            "h_nmr_j": "h_nmr_j_mask",
        }
        self.eps = eps

    def _stat(self, name: str, suffix: str) -> float:
        for key in (f"{name}_{suffix}", f"_{name}_{suffix}"):
            if key in self.stats:
                return float(self.stats[key])
        raise KeyError(f"Missing normalization statistic: {name}_{suffix}")

    def __call__(self, data):
        for key, enabled in self.enabled.items():
            if not enabled or not hasattr(data, key):
                continue
            value = getattr(data, key)
            mean = self._stat(self.stat_names[key], "mean")
            std = max(self._stat(self.stat_names[key], "std"), self.eps)
            normalized = (value - mean) / std
            mask_name = self.masks.get(key)
            if mask_name is not None and hasattr(data, mask_name):
                mask = getattr(data, mask_name).bool()
                normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
            setattr(data, key, normalized)
        return data


@functional_transform('one_hot')
class OneHot(BaseTransform):
    def __init__(
            self,
            values: list[int],
            key: str = "h",
            scale: float = 1.0,
            noise_std: float = 0.0,
            dtype: torch.dtype = torch.get_default_dtype()
    ) -> None:
        self.mapping = {v: i for (i, v) in enumerate(values)}
        self.key = key
        self.dtype = dtype
        self.noise_std = noise_std
        self.scale = scale

    def forward(self, data: Data) -> Data:
        data_key = getattr(data, self.key)
        assert data_key.ndim == 1

        x = torch.as_tensor([self.mapping[xi.item()] for xi in data_key])
        x = self.scale * one_hot(x, num_classes=len(self.mapping)).to(self.dtype)

        if self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std

        setattr(data, self.key, x)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mapping})'


@functional_transform('fully_connected')
class FullyConnected(BaseTransform):
    def __init__(
            self,
            key: str = "edge_index",
    ) -> None:
        self.key = key

    def forward(self, data: Data) -> Data:
        n = len(data.pos)
        fc_graph = torch.ones(n, n) - torch.eye(n)
        fc_edges, _ = dense_to_sparse(fc_graph)

        setattr(data, self.key, fc_edges)

        return data


@functional_transform('zero_cog')
class ZeroCoG(BaseTransform):
    def __init__(
            self,
            key: str = "pos",
    ) -> None:
        self.key = key

    def forward(self, data: Data) -> Data:
        pos = getattr(data, self.key)
        centered_pos = center(pos)
        setattr(data, self.key, centered_pos)

        return data


    
@functional_transform('hnmr')
class HNMR(BaseTransform):
    def __init__(
            self,
            nmr_std: float,
            nmr_mean: float,
            key_in: str = "h_nmr",
            key_out: str = "h_nmr_norm",
            dtype: torch.dtype = torch.get_default_dtype()
    ) -> None:
        self.nmr_std = nmr_std
        self.nmr_mean = nmr_mean
        self.key_in = key_in
        self.key_out = key_out
        self.dtype = dtype

    def forward(self, data: Data) -> Data:
        data_key_in = getattr(data, self.key_in)
        assert data_key_in.ndim == 1
        out_nmr = (data_key_in - self.nmr_mean) / self.nmr_std

        setattr(data, self.key_out, out_nmr)
        return data

@functional_transform('cnmr')
class CNMR(BaseTransform):
    def __init__(
            self,
            nmr_std: float,
            nmr_mean: float,
            key_in: str = "c_nmr",
            key_out: str = "c_nmr_norm",
            dtype: torch.dtype = torch.get_default_dtype()
    ) -> None:
        self.nmr_std = nmr_std
        self.nmr_mean = nmr_mean
        self.key_in = key_in
        self.key_out = key_out
        self.dtype = dtype

    def forward(self, data: Data) -> Data:
        data_key_in = getattr(data, self.key_in)
        assert data_key_in.ndim == 1
        out_nmr = (data_key_in - self.nmr_mean) / self.nmr_std

        setattr(data, self.key_out, out_nmr)
        return data
