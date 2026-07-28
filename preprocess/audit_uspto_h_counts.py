"""Audit USPTO proton/carbon peak cardinality against molecular atom counts.

The per-molecule output distinguishes the number of peak rows from the number
of proton labels after expanding MestreNova ``nH`` integrations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm

from preprocess.uspto_3d_nmr import (
    _as_peak_list,
    _integer_integration,
    canonical_smiles,
    iter_parquet_rows,
    load_nmr_split_index,
)


OUTPUT_SCHEMA = pa.schema(
    [
        ("row_index", pa.int64()),
        ("smiles", pa.string()),
        ("isomeric_smiles", pa.string()),
        ("split", pa.string()),
        ("valid_smiles", pa.bool_()),
        ("valid_integrations", pa.bool_()),
        # Number of distinct peak entries before nH expansion.
        ("h_peak_count", pa.int64()),
        # Sum(nH): number of labels after expanding every shift by nH.
        ("h_shift_atom_count", pa.int64()),
        ("zero_integration_peak_count", pa.int64()),
        ("invalid_integration_peak_count", pa.int64()),
        ("oh_h_count", pa.int64()),
        ("total_h_count", pa.int64()),
        ("non_oh_h_count", pa.int64()),
        # total explicit H - Sum(nH)
        ("total_minus_shift", pa.int64()),
        # (total explicit H - O-bound H) - Sum(nH)
        ("non_oh_minus_shift", pa.int64()),
        ("matches_all_h", pa.bool_()),
        ("missing_equals_oh", pa.bool_()),
        # Strictly positive missing count equal to the number of O-bound H.
        ("positive_missing_equals_oh", pa.bool_()),
        ("accept_exact_or_missing_oh", pa.bool_()),
        ("h_shift_count_gt_total_h", pa.bool_()),
        ("h_environment_count", pa.int64()),
        ("h_peaks_minus_environments", pa.int64()),
        ("h_peak_count_gt_environments", pa.bool_()),
        # Carbon entries are raw simulated lines and may include heteronuclear
        # splitting; both atom and graph-equivalence baselines are reported.
        ("c_peak_line_count", pa.int64()),
        ("total_c_count", pa.int64()),
        ("c_environment_count", pa.int64()),
        ("c_lines_minus_total_c", pa.int64()),
        ("c_lines_minus_environments", pa.int64()),
        ("c_line_count_gt_total_c", pa.bool_()),
        ("c_line_count_gt_environments", pa.bool_()),
    ]
)


def _molecular_counts(smiles: str) -> Optional[Dict[str, int]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    ranks = Chem.CanonicalRankAtoms(
        molecule, breakTies=False, includeChirality=True
    )
    carbon_ranks = {
        int(rank)
        for atom, rank in zip(molecule.GetAtoms(), ranks)
        if atom.GetAtomicNum() == 6
    }
    total_c = sum(
        atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
    )
    molecule = Chem.AddHs(molecule)
    explicit_ranks = Chem.CanonicalRankAtoms(
        molecule, breakTies=False, includeChirality=True
    )
    hydrogen_ranks = {
        int(rank)
        for atom, rank in zip(molecule.GetAtoms(), explicit_ranks)
        if atom.GetAtomicNum() == 1
    }
    total_h = 0
    oh_h = 0
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        total_h += 1
        neighbours = list(atom.GetNeighbors())
        if len(neighbours) == 1 and neighbours[0].GetAtomicNum() == 8:
            oh_h += 1
    return {
        "total_h_count": total_h,
        "oh_h_count": oh_h,
        "non_oh_h_count": total_h - oh_h,
        "total_c_count": total_c,
        "c_environment_count": len(carbon_ranks),
        "h_environment_count": len(hydrogen_ranks),
    }


def audit_row(
    row: Mapping[str, Any],
    row_index: int,
    split_for_smiles: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    smiles = str(row.get("smiles", ""))
    isomeric = canonical_smiles(smiles, isomeric=True)
    molecule_counts = _molecular_counts(smiles)
    peaks = _as_peak_list(row.get("h_nmr_peaks"))
    integrations: List[int] = []
    invalid_count = 0
    zero_count = 0
    for peak in peaks:
        integration = _integer_integration(peak)
        if integration is None:
            invalid_count += 1
            continue
        integrations.append(integration)
        zero_count += int(integration == 0)

    integrations_valid = invalid_count == 0
    shift_count = sum(integrations) if integrations_valid else None
    total_h = (
        None if molecule_counts is None else molecule_counts["total_h_count"]
    )
    oh_h = None if molecule_counts is None else molecule_counts["oh_h_count"]
    non_oh_h = (
        None if molecule_counts is None else molecule_counts["non_oh_h_count"]
    )
    total_minus_shift = (
        None
        if total_h is None or shift_count is None
        else total_h - shift_count
    )
    non_oh_minus_shift = (
        None
        if non_oh_h is None or shift_count is None
        else non_oh_h - shift_count
    )
    missing_equals_oh = (
        None
        if total_minus_shift is None or oh_h is None
        else total_minus_shift == oh_h
    )
    positive_missing_equals_oh = (
        None
        if missing_equals_oh is None
        else missing_equals_oh and total_minus_shift > 0
    )
    matches_all_h = (
        None if total_minus_shift is None else total_minus_shift == 0
    )
    carbon_peaks = _as_peak_list(row.get("c_nmr_peaks"))
    c_line_count = len(carbon_peaks)
    total_c = (
        None if molecule_counts is None else molecule_counts["total_c_count"]
    )
    c_environment_count = (
        None
        if molecule_counts is None
        else molecule_counts["c_environment_count"]
    )
    c_lines_minus_total = (
        None if total_c is None else c_line_count - total_c
    )
    c_lines_minus_environments = (
        None
        if c_environment_count is None
        else c_line_count - c_environment_count
    )
    h_environment_count = (
        None
        if molecule_counts is None
        else molecule_counts["h_environment_count"]
    )
    h_peaks_minus_environments = (
        None
        if h_environment_count is None
        else len(peaks) - h_environment_count
    )
    return {
        "row_index": row_index,
        "smiles": smiles,
        "isomeric_smiles": isomeric,
        "split": (
            None
            if split_for_smiles is None or isomeric is None
            else split_for_smiles.get(isomeric)
        ),
        "valid_smiles": molecule_counts is not None,
        "valid_integrations": integrations_valid,
        "h_peak_count": len(peaks),
        "h_shift_atom_count": shift_count,
        "zero_integration_peak_count": zero_count,
        "invalid_integration_peak_count": invalid_count,
        "oh_h_count": oh_h,
        "total_h_count": total_h,
        "non_oh_h_count": non_oh_h,
        "total_minus_shift": total_minus_shift,
        "non_oh_minus_shift": non_oh_minus_shift,
        "matches_all_h": matches_all_h,
        "missing_equals_oh": missing_equals_oh,
        "positive_missing_equals_oh": positive_missing_equals_oh,
        "accept_exact_or_missing_oh": (
            None
            if matches_all_h is None or positive_missing_equals_oh is None
            else matches_all_h or positive_missing_equals_oh
        ),
        "h_shift_count_gt_total_h": (
            None
            if total_minus_shift is None
            else total_minus_shift < 0
        ),
        "h_environment_count": h_environment_count,
        "h_peaks_minus_environments": h_peaks_minus_environments,
        "h_peak_count_gt_environments": (
            None
            if h_peaks_minus_environments is None
            else h_peaks_minus_environments > 0
        ),
        "c_peak_line_count": c_line_count,
        "total_c_count": total_c,
        "c_environment_count": c_environment_count,
        "c_lines_minus_total_c": c_lines_minus_total,
        "c_lines_minus_environments": c_lines_minus_environments,
        "c_line_count_gt_total_c": (
            None if c_lines_minus_total is None else c_lines_minus_total > 0
        ),
        "c_line_count_gt_environments": (
            None
            if c_lines_minus_environments is None
            else c_lines_minus_environments > 0
        ),
    }


def _update_summary(summary: Counter, record: Mapping[str, Any]) -> None:
    summary["rows"] += 1
    split = record["split"]
    summary[f"split/{split if split is not None else 'unassigned'}"] += 1
    if not record["valid_smiles"]:
        summary["invalid_smiles"] += 1
    if not record["valid_integrations"]:
        summary["invalid_integrations"] += 1
    summary[
        f"zero_integration_peak_count/"
        f"{record['zero_integration_peak_count']}"
    ] += 1
    if record["total_minus_shift"] is not None:
        summary[
            f"total_minus_shift/{record['total_minus_shift']}"
        ] += 1
    if record["non_oh_minus_shift"] is not None:
        summary[
            f"non_oh_minus_shift/{record['non_oh_minus_shift']}"
        ] += 1
    if record["matches_all_h"] is not None:
        summary[f"matches_all_h/{record['matches_all_h']}"] += 1
    if record["missing_equals_oh"] is not None:
        summary[
            f"missing_equals_oh/{record['missing_equals_oh']}"
        ] += 1
    if record["positive_missing_equals_oh"] is not None:
        summary[
            "positive_missing_equals_oh/"
            f"{record['positive_missing_equals_oh']}"
        ] += 1
    if record["accept_exact_or_missing_oh"] is not None:
        summary[
            "accept_exact_or_missing_oh/"
            f"{record['accept_exact_or_missing_oh']}"
        ] += 1
    for key in (
        "h_shift_count_gt_total_h",
        "h_peak_count_gt_environments",
        "c_line_count_gt_total_c",
        "c_line_count_gt_environments",
    ):
        if record[key] is not None:
            summary[f"{key}/{record[key]}"] += 1
    if record["c_lines_minus_total_c"] is not None:
        summary[
            f"c_lines_minus_total_c/{record['c_lines_minus_total_c']}"
        ] += 1
    if record["h_peaks_minus_environments"] is not None:
        summary[
            "h_peaks_minus_environments/"
            f"{record['h_peaks_minus_environments']}"
        ] += 1
    if record["c_lines_minus_environments"] is not None:
        summary[
            "c_lines_minus_environments/"
            f"{record['c_lines_minus_environments']}"
        ] += 1


def audit_hydrogen_counts(
    parquet_paths: Sequence[str | Path],
    output_path: str | Path,
    nmr_dir: str | Path | None = None,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_for_smiles = None
    if nmr_dir is not None:
        split_for_smiles, _, _ = load_nmr_split_index(nmr_dir)

    writer = pq.ParquetWriter(
        str(output_path), OUTPUT_SCHEMA, compression="zstd"
    )
    summary: Counter = Counter()
    buffer: List[Dict[str, Any]] = []
    try:
        rows: Iterator[Dict[str, Any]] = iter_parquet_rows(
            parquet_paths, batch_size=batch_size
        )
        for row_index, row in enumerate(tqdm(rows, desc="Auditing H counts")):
            record = audit_row(row, row_index, split_for_smiles)
            _update_summary(summary, record)
            buffer.append(record)
            if len(buffer) >= batch_size:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA)
                )
                buffer = []
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA)
            )
    finally:
        writer.close()

    report = {
        "parquet_paths": [str(Path(path)) for path in parquet_paths],
        "nmr_dir": None if nmr_dir is None else str(Path(nmr_dir)),
        "per_molecule_output": str(output_path),
        "counts": dict(sorted(summary.items())),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--nmr-dir",
        default=None,
        help=(
            "Optional directory containing train.pt/val.pt/test.pt; when "
            "provided, each molecule also receives its authoritative split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_hydrogen_counts(
        parquet_paths=args.parquet,
        output_path=args.output,
        nmr_dir=args.nmr_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
