#!/bin/bash -ue
# Check required Python packages in venv (also validates venv and pip existence)
echo "[ENV CHECK] Checking required Python packages in venv..."
if ! ${params.venv_path}/bin/pip show polars numpy scipy mne mne_nirs pyxdf matplotlib pikepdf > /dev/null 2>&1; then
    echo "[ENV CHECK] ERROR: One or more required packages are missing in the venv or venv/pip does not exist."
    exit 1
fi
echo "[ENV CHECK] venv, python, and all required packages found."
