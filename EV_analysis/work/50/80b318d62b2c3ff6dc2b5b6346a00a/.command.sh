#!/bin/bash -ue
echo "[DISPATCHER] Starting: /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py with /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/85/8eb9222bfa59d7f5aa40bbf5d464dd/EV_002_txt.parquet"
/home/gutt/EV_venv/bin/python3 -u /mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py /mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/work/85/8eb9222bfa59d7f5aa40bbf5d464dd/EV_002_txt.parquet ^bisBas ^bisBas
echo "[DISPATCHER] Python script completed with exit code: $?"
