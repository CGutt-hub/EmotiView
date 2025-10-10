#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/66/d40f57ce2ceb7395c1d958153bbd31/EV_003_txt_quest.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/66/d40f57ce2ceb7395c1d958153bbd31/EV_003_txt_quest.parquet bar '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results/EV_003'
echo "[DISPATCHER] Python script completed with exit code: $?"
