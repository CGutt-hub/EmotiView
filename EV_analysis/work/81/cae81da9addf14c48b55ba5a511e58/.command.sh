#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/48/15bc7b5bff880df255ff63a8ce1b66/EV_002_txt_quest.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/48/15bc7b5bff880df255ff63a8ce1b66/EV_002_txt_quest.parquet bar '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results/EV_002'
echo "[DISPATCHER] Python script completed with exit code: $?"
