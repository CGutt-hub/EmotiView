#!/bin/bash -ue
echo "Checking all Python modules."
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/readers/xdf_reader.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/readers/txt_reader.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/preprocessors/eeg_preprocessor.py 
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/preprocessors/ecg_preprocessor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/preprocessors/eda_preprocessor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/preprocessors/fnirs_preprocessor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/preprocessors/event_preprocessor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/processors/epoching_processor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/processors/merging_processor.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/ic_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/hrv_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/scr_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/glm_analyzer.py 
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/psd_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/fai_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/plv_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/utils/plotter.py
${params.venv_path}/bin/python3 -c ../../PsyAnalysisToolbox/Python/utils/git_sync.py
${params.venv_path}/bin/python3 -c null
echo "All Python modules checked."
