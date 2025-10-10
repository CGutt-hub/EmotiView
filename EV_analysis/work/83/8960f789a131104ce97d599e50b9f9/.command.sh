#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/xdf_reader.py with /mnt/emotiview/rawData/EV_003/EV_003.xdf"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/xdf_reader.py /mnt/emotiview/rawData/EV_003/EV_003.xdf .
echo "[DISPATCHER] Python script completed with exit code: $?"
