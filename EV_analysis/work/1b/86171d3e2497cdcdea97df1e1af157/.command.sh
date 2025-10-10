#!/bin/bash -ue
# Check venv directory
if [ ! -d "~/EV_venv" ]; then
    echo "[ENV CHECK] ERROR: venv not found at ~/EV_venv. Please create it manually."
    exit 1
fi
# Check pip executable
if [ ! -x "${params.venv_path}/bin/pip" ]; then
    echo "[ENV CHECK] ERROR: pip not found or not executable at ${params.venv_path}/bin/pip."
    exit 1
fi
# Check python executable
if [ ! -x "${params.venv_path}/bin/python3" ]; then
    echo "[ENV CHECK] ERROR: python not found or not executable at ${params.venv_path}/bin/python3."
    exit 1
fi
# Check input_dir
if [ ! -d "/mnt/emotiview/rawData" ]; then
    echo "[ENV CHECK] ERROR: input_dir not found at /mnt/emotiview/rawData."
    exit 1
fi
# Check output_dir (create if missing)
if [ ! -d "/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results" ]; then
    echo "[ENV CHECK] output_dir not found at /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results, creating it."
    mkdir -p "/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results"
fi
# Check required Python packages
echo "[ENV CHECK] Checking and installing required Python packages..."
if ! ${params.venv_path}/bin/pip show polars numpy scipy mne mne_nirs pyxdf matplotlib pikepdf > /dev/null 2>&1; then
    echo "[ENV CHECK] Installing missing packages: polars numpy scipy mne mne_nirs pyxdf matplotlib pikepdf"
    ${params.venv_path}/bin/pip install polars numpy scipy mne mne_nirs pyxdf matplotlib pikepdf
else
    echo "[ENV CHECK] All required packages already installed."
fi
echo "[ENV CHECK] All environment checks passed."
