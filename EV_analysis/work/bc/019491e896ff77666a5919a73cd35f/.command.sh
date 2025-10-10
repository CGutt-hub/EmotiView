#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/45/b82672ea0704dd113b6e7d8ae65770/EV_004_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/45/b82672ea0704dd113b6e7d8ae65770/EV_004_txt.parquet ^ea11 ^ea11
echo "[DISPATCHER] Python script completed with exit code: $?"
