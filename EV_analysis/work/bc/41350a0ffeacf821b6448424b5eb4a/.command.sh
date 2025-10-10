#!/bin/bash -ue
# Check venv directory
if [ ! -d !{params.venv_path} ]; then
    echo "[ENV CHECK] ERROR: venv not found at !{params.venv_path}. Please create it manually."
    exit 1
fi
# Check python executable
if [ ! -x !{params.python_exe} ]; then
    echo "[ENV CHECK] ERROR: python not found or not executable at !{params.python_exe}."
    exit 1
fi
echo "[ENV CHECK] venv and python found."
