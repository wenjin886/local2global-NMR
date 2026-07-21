from typing import Any

import torch


COMPACT_STORAGE_VERSION = 1


def _delete_field(data: Any, key: str) -> None:
    if not hasattr(data, key):
        return
    try:
        del data[key]
    except (KeyError, TypeError, AttributeError):
        delattr(data, key)


def _cast_integer_field(data: Any, key: str, dtype: torch.dtype) -> None:
    if not hasattr(data, key):
        return
    value = torch.as_tensor(getattr(data, key))
    limits = torch.iinfo(dtype)
    if value.numel() and (
        value.min().item() < limits.min or value.max().item() > limits.max
    ):
        raise ValueError(
            f"{key} values do not fit in {dtype}: "
            f"[{value.min().item()}, {value.max().item()}]"
        )
    setattr(data, key, value.to(dtype=dtype))


def compact_sample_storage(data: Any) -> Any:
    """Remove superseded fields and use lossless compact on-disk dtypes."""
    isomeric_smiles = getattr(data, "isomeric_smiles", None)
    if not isomeric_smiles:
        smiles = getattr(data, "smiles", None)
        if not smiles:
            raise ValueError("Cannot compact a sample without an isomeric SMILES")
        from src.data.dataset import canonicalize_smiles_with_stereo
        isomeric_smiles = canonicalize_smiles_with_stereo(smiles)
    data.isomeric_smiles = isomeric_smiles

    # These fields belonged to the previous local-label pipeline. Aromaticity
    # remains derivable from the factorized fragment or edge bond-type targets.
    for key in (
        "canno_h",
        "hydrogen_neighbors",
        "is_aromatic_heavy_atoms",
        "heavy_atom_local_labels",
        "smiles",
        "canonical_smiles",
        "original_smiles",
    ):
        _delete_field(data, key)

    for key in ("h", "bond_types"):
        _cast_integer_field(data, key, torch.uint8)
    for key in (
        "heavy_fragment_labels",
        "h_parent_fragment_labels",
        "h_parent_types",
    ):
        _cast_integer_field(data, key, torch.int8)
    for key in (
        "h_attachment",
        "h_nmr_multiplicity",
        "smiles_token_ids",
    ):
        _cast_integer_field(data, key, torch.int16)
    for key in (
        "h_nmr_integration_mask",
        "h_nmr_multiplicity_mask",
        "h_nmr_j_mask",
    ):
        if hasattr(data, key):
            setattr(data, key, torch.as_tensor(getattr(data, key)).bool())
    return data
