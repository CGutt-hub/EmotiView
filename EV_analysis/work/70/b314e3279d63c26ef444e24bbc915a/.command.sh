#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/fd/5cb4251c74e2a46c2f19d98a6f200f/EV_003_txt_quest.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/fd/5cb4251c74e2a46c2f19d98a6f200f/EV_003_txt_quest.parquet bar '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results/EV_004'
echo "[DISPATCHER] Python script completed with exit code: $?"
