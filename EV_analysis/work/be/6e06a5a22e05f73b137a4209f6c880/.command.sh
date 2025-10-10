#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/21/69f86f6519f7b9f4e4ebf889bc9df7/EV_001_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/21/69f86f6519f7b9f4e4ebf889bc9df7/EV_001_txt.parquet ^be7 ^be7
echo "[DISPATCHER] Python script completed with exit code: $?"
