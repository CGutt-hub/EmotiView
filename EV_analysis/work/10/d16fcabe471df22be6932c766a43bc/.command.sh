#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/39/fee6df3d87185ed9485fc5d0bbac6f/EV_003_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/39/fee6df3d87185ed9485fc5d0bbac6f/EV_003_txt.parquet ^ea11 ^ea11
echo "[DISPATCHER] Python script completed with exit code: $?"
