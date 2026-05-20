"""Export trial-level FAI values as a labels-style table for correlation_analyzer.

Input: parquet with condition/epoch_id and one or more fai_* columns.
Output: <base>_fai_labels.parquet with columns trial_id, fai.
"""
import os
import sys
import polars as pl


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: EV2_fai_label_export.py <fai_epochs.parquet>")
        return 1

    ip = sys.argv[1]
    df = pl.read_parquet(ip)

    # Normalize trial id.
    if 'trial_id' not in df.columns:
        if 'condition' in df.columns:
            df = df.with_columns(pl.col('condition').cast(pl.Utf8).alias('trial_id'))
        else:
            raise ValueError(f"No trial_id/condition column in {ip}")

    fai_cols = [c for c in df.columns if c.startswith('fai_')]
    if not fai_cols:
        # Fallback for unexpected naming.
        numeric = [c for c in df.columns if c not in ('trial_id', 'condition', 'epoch_id') and df[c].dtype.is_numeric()]
        if not numeric:
            raise ValueError(f"No FAI numeric column found in {ip}")
        fai_col = numeric[0]
    else:
        fai_col = fai_cols[0]

    out = (
        df.select([
            pl.col('trial_id').cast(pl.Utf8),
            pl.col(fai_col).cast(pl.Float64).alias('fai')
        ])
        .group_by('trial_id')
        .agg(pl.col('fai').mean().alias('fai'))
        .sort('trial_id')
    )

    base = os.path.splitext(os.path.basename(ip))[0]
    out_path = os.path.join(os.getcwd(), f"{base}_fai_labels.parquet")
    out.write_parquet(out_path, compression='snappy')
    print(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
