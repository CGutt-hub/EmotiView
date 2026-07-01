"""EV2 Dataset Ingestor

Converts preprocessed EV2 .dat files into pipeline-compatible parquets.

EV2 data layout per file (s01.dat .. s32.dat):
  data   : float32 (40 trials × 40 channels × 8064 samples @ 128 Hz)
           ch 0-31  EEG (10-20 system, see DEAP_EEG_CHANNELS)
           ch 36    GSR / EDA
           ch 37    BVP (blood volume pulse → HRV via peak detection)
  labels : float32 (40 trials × 4)  [valence, arousal, dominance, liking]  1-9 scale

Accepts two input formats:
  1. Direct .dat path: the raw EV2 .dat file
  2. Trigger parquet:  a parquet with columns {participant_id, dat_path}
     produced by deap_bootstrap.py — used when the pipeline downloads via kagglehub.

Pre-stimulus baseline: first 3 s = 384 samples; this ingestor strips it by default,
leaving 60 s = 7680 samples per trial.

Outputs a signal-pointer parquet (folder_path → subfolder) so file_finder can extract:
  {base}_eeg_epochs.parquet    condition, epoch_id, time, Fp1…Cz (32 ch)
  {base}_eda_epochs.parquet    condition, epoch_id, time, eda
  {base}_bvp_epochs.parquet    condition, epoch_id, time, bvp
  {base}_labels.parquet        trial_id, valence, arousal, dominance, liking

Usage:
    deap_ingestor.py <s0X.dat | DEAP_NN_dat.parquet> [strip_baseline=true] [sfreq=128]
"""
import pickle, numpy as np, polars as pl, sys, os, re

DEAP_EEG_CHANNELS = [
    'Fp1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',
    'CP5', 'CP1', 'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',
    'P4',  'P8',  'PO4', 'O2',  'T8',  'CP2', 'CP6', 'C4',
    'FC2', 'FC6', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz',
]

EDA_CH = 36   # GSR / EDA  (DEAP 1-based ch 37, 0-based index 36)
BVP_CH = 38   # Plethysmograph / BVP  (DEAP 1-based ch 39, 0-based index 38)
              # Note: index 37 (ch 38) is the Respiration belt — not used here.
N_EEG  = 32   # number of EEG channels


def log_info(msg):    print(f"[EV2] INFO: {msg}")
def log_warning(msg): print(f"[EV2] WARNING: {msg}")


def ingest(dat_path: str, strip_baseline: bool = True, sfreq: int = 128) -> str:
    """Ingest a EV2 .dat file (or trigger parquet) and write epoch parquets + signal pointer.

    Returns the path to the signal pointer parquet.
    """
    # ── Resolve trigger parquet → actual .dat path ────────────────────
    if dat_path.endswith('.parquet'):
        trigger = pl.read_parquet(dat_path)
        if 'dat_path' in trigger.columns:
            dat_path = str(trigger['dat_path'][0])
            log_info(f"Resolved trigger → {dat_path}")
        else:
            raise ValueError(f"Trigger parquet {dat_path} has no 'dat_path' column")

    log_info(f"Loading {dat_path}")
    with open(dat_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    data_full = np.array(raw['data'], dtype=np.float32)  # (40, 40, 8064)
    labels = np.array(raw['labels'], dtype=np.float32)   # (40, 4)

    n_trials, _, _ = data_full.shape
    baseline_samples = 3 * sfreq if strip_baseline else 0
    data = data_full[:, :, baseline_samples:]          # → (40, 40, 7680)
    n_samples = data.shape[2]
    times = (np.arange(n_samples) / sfreq).tolist()
    baseline_times = (np.arange(3 * sfreq) / sfreq).tolist()

    # Normalise ID to pipeline convention: s01 → DEAP_01 (matches ^[A-Za-z]+_[0-9]+ pattern
    # used by IOInterface for participant folder routing)
    raw_base       = os.path.splitext(os.path.basename(dat_path))[0]  # e.g. "s01"
    participant_num = re.sub(r'^[a-zA-Z]+', '', raw_base).zfill(2)    # e.g. "01"
    base           = f"EV2_{participant_num}"                          # e.g. "EV2_01"
    outdir = os.path.join(os.getcwd(), f"{base}_deap")
    os.makedirs(outdir, exist_ok=True)

    trial_ids = [f"trial_{t+1:02d}" for t in range(n_trials)]

    # ── EEG epochs ────────────────────────────────────────────────────
    # Wide format: one row per sample, columns = channels
    frames = []
    for t, cond in enumerate(trial_ids):
        row: dict = {
            'condition': [cond] * n_samples,
            'epoch_id':  [cond] * n_samples,
            'time':      times,
        }
        for ci, ch in enumerate(DEAP_EEG_CHANNELS):
            row[ch] = data[t, ci, :].tolist()
        frames.append(pl.DataFrame(row))
    eeg_path = os.path.join(outdir, f"{base}_eeg_epochs.parquet")
    pl.concat(frames).write_parquet(eeg_path, compression='gzip')
    log_info(f"EEG epochs → {eeg_path}  ({n_trials} trials × {n_samples} samples)")

    # ── EDA epochs ────────────────────────────────────────────────────
    frames = []
    for t, cond in enumerate(trial_ids):
        frames.append(pl.DataFrame({
            'condition': [cond] * n_samples,
            'epoch_id':  [cond] * n_samples,
            'time':      times,
            'eda':       data[t, EDA_CH, :].tolist(),
        }))
    eda_path = os.path.join(outdir, f"{base}_eda_epochs.parquet")
    pl.concat(frames).write_parquet(eda_path, compression='gzip')
    log_info(f"EDA epochs → {eda_path}")

    # ── BVP epochs ────────────────────────────────────────────────────
    frames = []
    for t, cond in enumerate(trial_ids):
        frames.append(pl.DataFrame({
            'condition': [cond] * n_samples,
            'epoch_id':  [cond] * n_samples,
            'time':      times,
            'bvp':       data[t, BVP_CH, :].tolist(),
        }))
    bvp_path = os.path.join(outdir, f"{base}_bvp_epochs.parquet")
    pl.concat(frames).write_parquet(bvp_path, compression='gzip')
    log_info(f"BVP epochs → {bvp_path}")

    # ── BVP baseline epochs (first 3 seconds before stimulus) ───────────
    baseline_frames = []
    for t, cond in enumerate(trial_ids):
        baseline_frames.append(pl.DataFrame({
            'condition': [cond] * (3 * sfreq),
            'epoch_id':  [cond] * (3 * sfreq),
            'time':      baseline_times,
            'bvp':       data_full[t, BVP_CH, :3 * sfreq].tolist(),
        }))
    bvp_baseline_path = os.path.join(outdir, f"{base}_bvp_baseline_epochs.parquet")
    pl.concat(baseline_frames).write_parquet(bvp_baseline_path, compression='gzip')
    log_info(f"BVP baseline epochs → {bvp_baseline_path}")

    # ── Labels ────────────────────────────────────────────────────────
    labels_path = os.path.join(outdir, f"{base}_labels.parquet")
    pl.DataFrame({
        'trial_id':  trial_ids,
        'valence':   labels[:, 0].tolist(),
        'arousal':   labels[:, 1].tolist(),
        'dominance': labels[:, 2].tolist(),
        'liking':    labels[:, 3].tolist(),
    }).write_parquet(labels_path, compression='gzip')
    log_info(f"Labels → {labels_path}")

    # ── Signal pointer ─────────────────────────────────────────────────
    # file_finder reads folder_path from this parquet to locate the outputs above
    signal_path = os.path.join(os.getcwd(), f"{base}_deap.parquet")
    pl.DataFrame({
        'signal':      [1],
        'folder_path': [os.path.abspath(outdir)],
    }).write_parquet(signal_path, compression='gzip')
    log_info(f"Signal pointer → {signal_path}")
    print(signal_path)
    return signal_path


if __name__ == '__main__':
    a = sys.argv
    if len(a) < 2:
        print('[EV2] Usage: deap_ingestor.py <s0X.dat | DEAP_NN_dat.parquet> [strip_baseline=true] [sfreq=128]')
        sys.exit(1)
    strip = len(a) <= 2 or a[2].lower() not in ('0', 'false', 'no')
    sfreq = int(a[3]) if len(a) > 3 else 128
    ingest(a[1], strip_baseline=strip, sfreq=sfreq)
