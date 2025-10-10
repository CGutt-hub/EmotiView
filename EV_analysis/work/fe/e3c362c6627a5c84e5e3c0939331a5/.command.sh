#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/c2/1def56f39f8780f07f05d718b5b69f/EV_004_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/c2/1def56f39f8780f07f05d718b5b69f/EV_004_txt.parquet ^bisBas ^bisBas
echo "[DISPATCHER] Python script completed with exit code: $?"
