#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/64/0d3cc65ac2d1c22925ab5194e739f3/EV_003_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/64/0d3cc65ac2d1c22925ab5194e739f3/EV_003_txt.parquet ^bisBas ^bisBas
echo "[DISPATCHER] Python script completed with exit code: $?"
