#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/txt_reader.py with /mnt/emotiview/rawData/EV_004/EV_004.txt"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/txt_reader.py /mnt/emotiview/rawData/EV_004/EV_004.txt utf-16
echo "[DISPATCHER] Python script completed with exit code: $LASTEXITCODE"
