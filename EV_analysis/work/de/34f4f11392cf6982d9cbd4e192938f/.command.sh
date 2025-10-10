#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/d7/e70fca2d8e849598ca612c1e7a57df/EV_002_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/d7/e70fca2d8e849598ca612c1e7a57df/EV_002_txt.parquet ^be7 ^be7
echo "[DISPATCHER] Python script completed with exit code: $?"
