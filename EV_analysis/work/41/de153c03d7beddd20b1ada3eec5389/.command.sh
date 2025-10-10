#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/96/3b39abaaf482d831b9de27b9fa1f2e/EV_001_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/96/3b39abaaf482d831b9de27b9fa1f2e/EV_001_txt.parquet ^bisBas ^bisBas
echo "[DISPATCHER] Python script completed with exit code: $?"
