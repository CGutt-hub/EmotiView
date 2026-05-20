"""EV2 Bootstrap — Download EV2 dataset via kagglehub and emit per-participant triggers.

Reads a config parquet containing the Kaggle dataset ID, calls
kagglehub.dataset_download() to populate the local cache (idempotent — skips
download if already cached), then writes one trigger parquet per participant
(DEAP_01_dat.parquet … DEAP_32_dat.parquet) each containing the resolved
absolute path to the corresponding s0N.dat file.

These trigger parquets are collected by IOInterface and fanned out by
Nextflow's .flatten() operator, one per participant, into deap_ingestor.

Usage:
    deap_bootstrap.py <kaggle_config.parquet> [n_participants=32]
"""
import polars as pl, sys, os, re, kagglehub


def bootstrap(config_path: str, n_participants: int = 32) -> list[str]:
    cfg = pl.read_parquet(config_path)
    kaggle_id = str(cfg['kaggle_id'][0])
    print(f"[deap_bootstrap] Dataset: {kaggle_id}")

    dataset_root = kagglehub.dataset_download(kaggle_id)
    print(f"[deap_bootstrap] Dataset root: {dataset_root}")

    # Locate data_preprocessed_python/ — may be at root or one level deeper
    dat_dir = None
    for root, dirs, files in os.walk(dataset_root):
        if 'data_preprocessed_python' in dirs:
            dat_dir = os.path.join(root, 'data_preprocessed_python')
            break
        # Also accept if we're already inside it
        dat_files = [f for f in files if re.match(r'^s\d+\.dat$', f)]
        if dat_files:
            dat_dir = root
            break

    if dat_dir is None or not os.path.isdir(dat_dir):
        print(f"[deap_bootstrap] ERROR: could not find data_preprocessed_python under {dataset_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[deap_bootstrap] .dat directory: {dat_dir}")

    trigger_paths = []
    for n in range(1, n_participants + 1):
        dat_name = f"s{n:02d}.dat"
        dat_path = os.path.join(dat_dir, dat_name)
        if not os.path.exists(dat_path):
            print(f"[deap_bootstrap] WARNING: {dat_name} not found — skipping", file=sys.stderr)
            continue
        pid = f"DEAP_{n:02d}"
        trigger = os.path.join(os.getcwd(), f"{pid}_dat.parquet")
        pl.DataFrame({'participant_id': [pid], 'dat_path': [os.path.abspath(dat_path)]}) \
            .write_parquet(trigger, compression='snappy')
        print(trigger)
        trigger_paths.append(trigger)

    print(f"[deap_bootstrap] Emitted {len(trigger_paths)} trigger parquets", file=sys.stderr)
    return trigger_paths


if __name__ == '__main__':
    a = sys.argv
    if len(a) < 2:
        print('[deap_bootstrap] Usage: deap_bootstrap.py <kaggle_config.parquet> [n_participants=32]')
        sys.exit(1)
    n = int(a[2]) if len(a) > 2 and a[2] not in ('None', '') else 32
    bootstrap(a[1], n)
