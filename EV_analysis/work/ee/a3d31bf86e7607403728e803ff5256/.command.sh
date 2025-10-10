#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/b0/85df8a1a409bb48a4cdceb983220ff/EV_004_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/b0/85df8a1a409bb48a4cdceb983220ff/EV_004_txt.parquet ^sam ^sam
echo "[DISPATCHER] Python script completed with exit code: $?"
