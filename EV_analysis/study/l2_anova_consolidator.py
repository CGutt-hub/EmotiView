"""Consolidate modality-specific L2 ANOVA tables into one long-format summary.

Usage:
    python l2_anova_consolidator.py <anova1.parquet> <anova2.parquet> ... <out_basename>

Input tables are expected to come from anova_analyzer.py and contain one row with:
- x_data: column names (e.g., DV, F, df1, df2, p, η², sig)
- y_data: column vectors aligned with x_data

Output is a plain long-format parquet with one row per (modality, DV), including
eta-squared interpretation for quick hypothesis readout.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import polars as pl


def log_info(msg: str) -> None:
    print(f"[l2_consolidator] INFO: {msg}")


def log_warning(msg: str) -> None:
    print(f"[l2_consolidator] WARNING: {msg}")


def modality_from_path(path: str) -> str:
    name = os.path.basename(path).lower()
    if "eeg_frontal" in name:
        return "eeg_frontal"
    if "eeg_parietal" in name:
        return "eeg_parietal"
    if "eda" in name:
        return "eda"
    if "hrv" in name:
        return "hrv"
    if "fai" in name:
        return "fai"
    return "unknown"


def normalize_column(name: str) -> str:
    raw = str(name).strip()
    mapping = {
        "DV": "dv",
        "dv": "dv",
        "F": "F",
        "df1": "df1",
        "df2": "df2",
        "p": "p",
        "p (FDR-corrected)": "p",
        "η²": "eta_sq",
        "eta_sq": "eta_sq",
        "sig": "sig",
    }
    return mapping.get(raw, raw.lower().replace(" ", "_"))


def eta_sq_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "medium"
    return "large"


def parse_anova_table(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df.height == 0:
        log_warning(f"Skipping empty file: {path}")
        return pl.DataFrame([])

    row = df.to_dicts()[0]
    x_data = row.get("x_data", [])
    y_data = row.get("y_data", [])

    if not isinstance(x_data, list) or not isinstance(y_data, list) or len(x_data) == 0:
        log_warning(f"Skipping unexpected format: {path}")
        return pl.DataFrame([])

    columns: Dict[str, List] = {}
    for idx, col_name in enumerate(x_data):
        key = normalize_column(col_name)
        if idx < len(y_data) and isinstance(y_data[idx], list):
            columns[key] = y_data[idx]

    if "dv" not in columns:
        log_warning(f"Skipping table without DV column: {path}")
        return pl.DataFrame([])

    table = pl.DataFrame(columns)

    # Keep only expected numeric/stat columns when present, plus DV/significance fields.
    keep = [c for c in ["dv", "F", "df1", "df2", "p", "eta_sq", "sig"] if c in table.columns]
    table = table.select(keep)

    # Cast known fields to stable dtypes.
    for col, dtype in [("F", pl.Float64), ("p", pl.Float64), ("eta_sq", pl.Float64), ("df1", pl.Int64), ("df2", pl.Int64)]:
        if col in table.columns:
            table = table.with_columns(pl.col(col).cast(dtype, strict=False))

    modality = modality_from_path(path)
    table = table.with_columns([
        pl.lit(modality).alias("modality"),
        pl.lit(os.path.basename(path)).alias("source_file"),
    ])

    if "eta_sq" in table.columns:
        table = table.with_columns([
            pl.col("eta_sq").map_elements(eta_sq_label, return_dtype=pl.String).alias("eta_sq_label")
        ])

    if "p" in table.columns:
        table = table.with_columns([
            (pl.col("p") < 0.05).alias("reject_h0_p_lt_0_05")
        ])

    return table.select([
        c
        for c in [
            "modality",
            "dv",
            "F",
            "df1",
            "df2",
            "p",
            "sig",
            "eta_sq",
            "eta_sq_label",
            "reject_h0_p_lt_0_05",
            "source_file",
        ]
        if c in table.columns
    ])


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(
            "Usage: l2_anova_consolidator.py <anova1.parquet> <anova2.parquet> ... <out_basename>"
        )
        return 1

    in_files = [a for a in argv[1:-1] if a.endswith(".parquet") and os.path.exists(a)]
    out_base = argv[-1]

    if not in_files:
        log_warning("No input parquet files found.")
        return 1

    frames = [parse_anova_table(path) for path in in_files]
    frames = [f for f in frames if f.height > 0]

    if not frames:
        log_warning("No valid ANOVA tables could be parsed.")
        return 1

    result = pl.concat(frames, how="diagonal")
    if "modality" in result.columns and "dv" in result.columns:
        result = result.sort(["modality", "dv"])

    out_path = os.path.join(os.getcwd(), f"{out_base}.parquet")
    result.write_parquet(out_path, compression="snappy")
    log_info(f"Wrote consolidated L2 ANOVA summary: {out_path}")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
