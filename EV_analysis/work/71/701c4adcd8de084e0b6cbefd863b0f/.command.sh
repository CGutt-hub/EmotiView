#!/bin/bash -ue
# Check venv directory
if [ ! -d /home/gutt/EV_venv ]; then
    echo "[ENV CHECK] ERROR: venv not found at /home/gutt/EV_venv. Please create it manually."
    exit 1
fi
# Check python executable
if [ ! -x ${params.venv_path}/bin/python3 ]; then
    echo "[ENV CHECK] ERROR: python not found or not executable at ${params.venv_path}/bin/python3."
    exit 1
fi
echo "[ENV CHECK] venv and python found."
