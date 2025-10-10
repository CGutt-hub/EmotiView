#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/08/cd0da26f1f894fb5b54b4bda702b69/EV_003_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/08/cd0da26f1f894fb5b54b4bda702b69/EV_003_txt.parquet ^be7 ^be7
echo "[DISPATCHER] Python script completed with exit code: $?"
