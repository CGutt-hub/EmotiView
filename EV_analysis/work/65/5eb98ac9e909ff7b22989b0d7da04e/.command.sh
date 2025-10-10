#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/0f/602dd1a7eaac10a5259ac24a608ab4/EV_001_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/0f/602dd1a7eaac10a5259ac24a608ab4/EV_001_txt.parquet ^panas ^panas
echo "[DISPATCHER] Python script completed with exit code: $?"
