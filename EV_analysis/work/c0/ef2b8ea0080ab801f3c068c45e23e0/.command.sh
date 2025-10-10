#!/bin/bash -ue
# Check venv directory
    if [ ! -d "!{params.venv_path}" ]; then
        echo "[ENV CHECK] ERROR: venv not found at !{params.venv_path}. Please create it manually."
        exit 1
    fi
    # Check pip executable
    if [ ! -x "!{params.pip_exe}" ]; then
        echo "[ENV CHECK] ERROR: pip not found or not executable at !{params.pip_exe}."
        exit 1
    fi
    # Check python executable
    if [ ! -x "!{params.python_exe}" ]; then
        echo "[ENV CHECK] ERROR: python not found or not executable at !{params.python_exe}."
        exit 1
    fi
    # Check input_dir
    if [ ! -d "!{params.input_dir}" ]; then
        echo "[ENV CHECK] ERROR: input_dir not found at !{params.input_dir}."
        exit 1
    fi
    # Check output_dir (create if missing)
    if [ ! -d "!{params.output_dir}" ]; then
        echo "[ENV CHECK] output_dir not found at !{params.output_dir}, creating it."
        mkdir -p "!{params.output_dir}"
    fi
# Check required Python packages
echo "[ENV CHECK] Checking and installing required Python packages..."
    if ! !{params.pip_exe} show !{params.python_requirements} > /dev/null 2>&1; then
        echo "[ENV CHECK] Installing missing packages: !{params.python_requirements}"
        !{params.pip_exe} install !{params.python_requirements}
    else
        echo "[ENV CHECK] All required packages already installed."
    fi
    echo "[ENV CHECK] All environment checks passed."
