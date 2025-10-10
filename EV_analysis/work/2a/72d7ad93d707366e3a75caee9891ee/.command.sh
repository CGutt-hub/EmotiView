#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/20/351e09b511a78d08355c365cfb2bb6/EV_002_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/20/351e09b511a78d08355c365cfb2bb6/EV_002_txt.parquet ^ea11 ^ea11
echo "[DISPATCHER] Python script completed with exit code: $?"
