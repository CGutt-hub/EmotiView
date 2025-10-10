#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/51/1fdda757952a0f32592d28f66c9bfc/EV_004_txt_quest.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/51/1fdda757952a0f32592d28f66c9bfc/EV_004_txt_quest.parquet bar '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results/EV_001'
echo "[DISPATCHER] Python script completed with exit code: $?"
