"""Study L2 analysis for EV2.

This post-processor scans the finished EV2 L1 folders, copies the relevant
participant-level raw summaries into each participant's results/ folder, and
builds a second-level ANOVA table for valence/arousal bins across modalities.

Inputs:
    argv[1:] = ignored completion-barrier file list emitted by Nextflow
    argv[-5] = path to the EV2_l1 root folder
    argv[-4] = project name (e.g. EV2)
    argv[-3] = sampling rate (unused, kept for compatibility)
    argv[-2] = valence threshold
    argv[-1] = arousal threshold

Outputs:
    Writes participant summaries into each participant's results/ folder.
    Writes L2 ANOVA parquets into the current working directory so Nextflow
    publishes them to EV2_l2/results/ via the wrapper's `result` token.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import f_oneway, pearsonr


def _first_match(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _load_labels(labels_path: Path, valence_threshold: float, arousal_threshold: float) -> pl.DataFrame:
    df = pl.read_parquet(labels_path)
    if 'trial_id' not in df.columns and 'condition' in df.columns:
        df = df.rename({'condition': 'trial_id'})
    if 'trial_id' not in df.columns:
        raise ValueError(f"Labels file missing trial_id column: {labels_path}")
    if 'valence' not in df.columns or 'arousal' not in df.columns:
        raise ValueError(f"Labels file missing valence/arousal columns: {labels_path}")
    return df.select([
        pl.col('trial_id').cast(pl.Utf8),
        pl.col('valence').cast(pl.Float64),
        pl.col('arousal').cast(pl.Float64),
    ]).with_columns([
        (pl.col('valence') >= valence_threshold).cast(pl.Int8).alias('valence_bin'),
        (pl.col('arousal') >= arousal_threshold).cast(pl.Int8).alias('arousal_bin'),
    ])


def _normalize_metric_frame(df: pl.DataFrame, source: str) -> pl.DataFrame:
    cols = set(df.columns)
    trial_col = 'trial_id' if 'trial_id' in cols else 'condition' if 'condition' in cols else None
    if trial_col is None:
        raise ValueError(f"No trial identifier found in {source}: {list(df.columns)}")

    value_col = 'value' if 'value' in cols else next((c for c in df.columns if c.startswith('fai_')), None)
    if value_col is None:
        numeric_cols = [c for c in df.columns if c not in {'trial_id', 'condition', 'epoch_id', 'metric', 'source'} and df[c].dtype.is_numeric()]
        value_col = numeric_cols[0] if numeric_cols else None
    if value_col is None:
        raise ValueError(f"No numeric value column found in {source}: {list(df.columns)}")

    keep_cols = [trial_col]
    if 'epoch_id' in cols:
        keep_cols.append('epoch_id')
    keep_cols.append(value_col)

    out = df.select([pl.col(c).alias('trial_id') if c == trial_col else pl.col(c) for c in keep_cols])
    if value_col != 'value':
        out = out.rename({value_col: 'value'})
    return out.with_columns(pl.lit(source).alias('source'))


def _collect_participant_metrics(participant_dir: Path, valence_threshold: float, arousal_threshold: float) -> pl.DataFrame | None:
    labels_path = _first_match(participant_dir, ['**/*labels*.parquet'])
    if labels_path is None:
        print(f"[EV2 L2] No labels file for {participant_dir.name}, skipping")
        return None

    labels = _load_labels(labels_path, valence_threshold, arousal_threshold)

    results_dir = participant_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    participant_rows: list[pl.DataFrame] = []
    pid = participant_dir.name

    # FAI: one raw epoch table per participant.
    fai_path = _first_match(participant_dir, ['**/*fai_epochs.parquet', '**/*_fai.parquet'])
    if fai_path is not None:
        fai_df = _normalize_metric_frame(pl.read_parquet(fai_path), 'FAI')
        fai_df = fai_df.join(labels, on='trial_id', how='inner')
        fai_df.write_parquet(results_dir / f'{pid}_eeg_fai_result.parquet', compression='snappy')
        participant_rows.append(fai_df)

    # EDA: aggregate all raw amplitude files for the participant.
    eda_files = sorted(participant_dir.rglob('*amp*.parquet'))
    eda_frames = []
    for path in eda_files:
        try:
            frame = _normalize_metric_frame(pl.read_parquet(path), 'EDA')
            eda_frames.append(frame)
        except Exception:
            continue
    if eda_frames:
        eda_df = pl.concat(eda_frames, how='diagonal')
        eda_df = eda_df.join(labels, on='trial_id', how='inner')
        eda_df.write_parquet(results_dir / f'{pid}_eda_result.parquet', compression='snappy')
        participant_rows.append(eda_df)

    # HRV: aggregate all interval files for the participant.
    hrv_files = sorted(participant_dir.rglob('*interv*.parquet'))
    hrv_frames = []
    for path in hrv_files:
        try:
            frame = _normalize_metric_frame(pl.read_parquet(path), 'HRV')
            hrv_frames.append(frame)
        except Exception:
            continue
    if hrv_frames:
        hrv_df = pl.concat(hrv_frames, how='diagonal')
        hrv_df = hrv_df.join(labels, on='trial_id', how='inner')
        hrv_df.write_parquet(results_dir / f'{pid}_hrv_result.parquet', compression='snappy')
        participant_rows.append(hrv_df)

    if not participant_rows:
        return None

    return pl.concat(participant_rows, how='diagonal')


def _anova_rows(df: pl.DataFrame, between: str) -> tuple[list[str], list[list[str]]]:
    numeric_cols = [c for c in df.columns if c not in {'trial_id', 'epoch_id', 'source', 'valence', 'arousal', 'valence_bin', 'arousal_bin'} and df[c].dtype.is_numeric()]
    sources = sorted(df['source'].drop_nulls().unique().to_list())

    rows = []
    pvals = []
    for source in sources:
        sub = df.filter(pl.col('source') == source)
        for col in numeric_cols:
            groups = [sub.filter(pl.col(between) == lvl)[col].drop_nulls().to_numpy() for lvl in sorted(sub[between].drop_nulls().unique().to_list())]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            F, p = f_oneway(*groups)
            all_vals = np.concatenate(groups)
            grand_mean = all_vals.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
            rows.append([source, col, float(F), len(groups) - 1, int(sum(len(g) for g in groups) - len(groups)), float(p), float(eta_sq)])
            pvals.append(float(p))
    return ["source", "DV", "F", "df1", "df2", "p", "eta_sq"], rows


def _write_anova(df: pl.DataFrame, between: str, out_name: str) -> None:
    cols, rows = _anova_rows(df, between)
    if not rows:
        raise ValueError(f"No ANOVA rows produced for {between}")
    table = pl.DataFrame({
        'x_data': [cols],
        'y_data': [[list(map(str, row)) for row in rows]],
        'y_var': [None],
        'plot_type': ['table'],
        'x_label': [between],
        'y_label': ['p'],
        'y_ticks': [None],
    })
    table.write_parquet(out_name, compression='snappy')
    print(f"[EV2 L2] Wrote {out_name}")


def _write_wp3_baseline_plv_correlation(
    l1_root: Path,
    valence_threshold: float,
    out_name: str,
) -> None:
    """WP3: correlate participant baseline RMSSD with negative-condition EEG-HRV PLV.

    Negative condition is operationalized as low-valence trials (valence < threshold).
    """
    rows = []
    for participant_dir in sorted(p for p in l1_root.iterdir() if p.is_dir()):
        pid = participant_dir.name
        results_dir = participant_dir / 'results'

        baseline_path = _first_match(results_dir, [f"{pid}_baseline_rmssd.parquet"])
        plv_path = _first_match(results_dir, [f"{pid}_plv_table.parquet"])
        labels_path = _first_match(participant_dir, ['**/*labels*.parquet'])
        if baseline_path is None or plv_path is None or labels_path is None:
            continue

        try:
            baseline_df = pl.read_parquet(baseline_path)
            if 'value' not in baseline_df.columns:
                continue
            baseline_rmssd = float(baseline_df['value'].drop_nulls().mean())

            labels = _load_labels(labels_path, valence_threshold, 5.0).select(['trial_id', 'valence'])
            labels = labels.with_columns((pl.col('valence') < valence_threshold).alias('is_negative'))

            plv_df = pl.read_parquet(plv_path)
            if 'trial_id' not in plv_df.columns and 'condition' in plv_df.columns:
                plv_df = plv_df.with_columns(pl.col('condition').cast(pl.Utf8).alias('trial_id'))
            if 'trial_id' not in plv_df.columns:
                continue

            merged = plv_df.join(labels, on='trial_id', how='inner').filter(pl.col('is_negative'))
            eeg_hrv_cols = [
                c for c in merged.columns
                if c not in ('condition', 'epoch_id', 'trial_id', 'valence', 'is_negative')
                and merged[c].dtype.is_numeric()
                and 'bvp' in c.lower()
            ]
            if not eeg_hrv_cols or len(merged) == 0:
                continue

            neg_plv = float(merged.select([pl.mean(c).alias(c) for c in eeg_hrv_cols]).row(0)[0])
            # If multiple EEG-HRV columns exist, average their means.
            if len(eeg_hrv_cols) > 1:
                neg_plv = float(np.mean([float(merged[c].drop_nulls().mean()) for c in eeg_hrv_cols]))

            rows.append({'participant_id': pid, 'baseline_rmssd': baseline_rmssd, 'negative_eeg_hrv_plv': neg_plv})
        except Exception as e:
            print(f"[EV2 L2] WP3 skip {pid}: {e}")

    if len(rows) < 3:
        raise ValueError("Insufficient participants for WP3 baseline-PLV correlation")

    wp3_df = pl.DataFrame(rows)
    r, p = pearsonr(wp3_df['baseline_rmssd'].to_numpy(), wp3_df['negative_eeg_hrv_plv'].to_numpy())

    table = pl.DataFrame({
        'x_data': [['metric', 'r', 'p', 'N']],
        'y_data': [[["baseline_rmssd_vs_negative_eeg_hrv_plv", f"{float(r):.4f}", f"{float(p):.6f}", str(len(wp3_df))]]],
        'y_var': [None],
        'plot_type': ['table'],
        'x_label': ['WP3'],
        'y_label': ['Pearson correlation'],
        'y_ticks': [None],
    })
    table.write_parquet(out_name, compression='snappy')
    print(f"[EV2 L2] Wrote {out_name}")


def main(argv: list[str]) -> int:
    if len(argv) < 6:
        print('Usage: EV2_l2_analysis.py <ignored...> <l1_root> <project_name> <sfreq> <valence_threshold> <arousal_threshold>')
        return 1

    l1_root = Path(argv[-5])
    project_name = argv[-4]
    valence_threshold = float(argv[-2])
    arousal_threshold = float(argv[-1])

    l2_root = l1_root.parent / f'{project_name}_l2'
    l2_results = l2_root / 'results'
    l2_results.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pl.DataFrame] = []
    for participant_dir in sorted(p for p in l1_root.iterdir() if p.is_dir()):
        participant_df = _collect_participant_metrics(participant_dir, valence_threshold, arousal_threshold)
        if participant_df is not None and len(participant_df) > 0:
            combined_rows.append(participant_df)

    if not combined_rows:
        print('[EV2 L2] No participant rows found')
        return 1

    combined = pl.concat(combined_rows, how='diagonal')
    tidy_path = l2_results / f'{project_name}_l2_tidy_result.parquet'
    combined.write_parquet(tidy_path, compression='snappy')

    _write_anova(combined, 'valence_bin', f'{project_name}_l2_valence_anova_result.parquet')
    _write_anova(combined, 'arousal_bin', f'{project_name}_l2_arousal_anova_result.parquet')
    _write_wp3_baseline_plv_correlation(l1_root, valence_threshold, f'{project_name}_l2_baseline_plv_correlation_result.parquet')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))