#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/fe/e3c362c6627a5c84e5e3c0939331a5/EV_004_txt_quest.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/fe/e3c362c6627a5c84e5e3c0939331a5/EV_004_txt_quest.parquet bar '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results/EV_001'
echo "[DISPATCHER] Python script completed with exit code: $?"
