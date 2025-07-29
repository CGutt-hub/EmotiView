import os
import sys
import platform
import datetime
import hashlib
import json
import importlib

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
toolbox_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'PsyAnalysisToolbox', 'Python'))
if toolbox_path not in sys.path:
    sys.path.insert(0, toolbox_path)

import logging
from typing import Dict, Any, List
import pandas as pd
from configparser import ConfigParser

# --- Import EmotiView Components ---
try:
    from analyzers.anova_analyzer import ANOVAAnalyzer
    from analyzers.groupLevel_analyzer import GroupLevelAnalyzing
    from reporters.xml_reporter import XMLReporter
    from analyzers.fdr_corrector import FDRCorrector
    from PsyAnalysisToolbox.Python.utils.group_aggregator import GroupAggregator
except ImportError as e:
    print(f"FATAL: Could not import necessary EmotiView modules. Make sure paths are correct. Error: {e}")
    sys.exit(1)

# --- PATCH: Robustly handle new questionnaire DataFrame format from E-Prime parser ---
# The expected DataFrame should have columns: participant_id, item_id, item_text, response_value
# Add checks and debug logging to ensure compatibility
import logging
import pandas as pd

def robust_load_questionnaire_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the questionnaire DataFrame has the required columns and is in the correct format.
    Logs debug info and returns a cleaned DataFrame.
    """
    required_cols = ['participant_id', 'item_id', 'response_value']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.warning(f"Questionnaire DataFrame is missing columns: {missing}. Columns found: {list(df.columns)}")
        # Try to infer columns if possible
        if 'Subject' in df.columns:
            df = df.rename(columns={'Subject': 'participant_id'})
        if 'bisBasList' in df.columns:
            df = df.rename(columns={'bisBasList': 'item_id'})
        if 'bisBas.Choice1.Value' in df.columns:
            df = df.rename(columns={'bisBas.Choice1.Value': 'response_value'})
        # Re-check
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Cannot process questionnaire DataFrame: missing columns {missing}")
    # Ensure correct types
    df['participant_id'] = df['participant_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['response_value'] = pd.to_numeric(df['response_value'], errors='coerce')
    logging.info(f"Loaded questionnaire DataFrame with shape {df.shape} and columns {list(df.columns)}")
    return df

def hash_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def load_participant_results(config, logger):
    base_output_dir = config['Data']['base_output_dir']
    all_participant_csv_data = []
    seen_participant_ids = set()
    logger.info(f"Scanning for participant result directories in: {base_output_dir}")
    for dir_name in os.listdir(base_output_dir):
        participant_dir_path = os.path.join(base_output_dir, dir_name)
        if dir_name.startswith("EV_") and os.path.isdir(participant_dir_path):
            participant_id = dir_name
            if participant_id in seen_participant_ids:
                logger.warning(f"Duplicate participant ID found: {participant_id}")
                continue
            seen_participant_ids.add(participant_id)
            participant_data = {'participant_id': participant_id}
            for file_name in os.listdir(participant_dir_path):
                file_path = os.path.join(participant_dir_path, file_name)
                if file_name == f"{participant_id}_plv_results.xml":
                    try:
                        df = pd.read_xml(file_path)
                        participant_data['plv_results_df'] = df
                    except Exception as e:
                        logger.error(f"Failed to read PLV xml for {participant_id} at {file_path}: {e}")
                elif file_name == f"{participant_id}_fai_results.xml":
                    try:
                        df = pd.read_xml(file_path)
                        participant_data['fai_results_df'] = df
                    except Exception as e:
                        logger.error(f"Failed to read FAI xml for {participant_id} at {file_path}: {e}")
                elif file_name == f"{participant_id}_questionnaire_results.xml":
                    try:
                        df = pd.read_xml(file_path)
                        participant_data['questionnaire_results_df'] = df
                    except Exception as e:
                        logger.error(f"Failed to read Questionnaire xml for {participant_id} at {file_path}: {e}")
            if len(participant_data) > 1:
                all_participant_csv_data.append(participant_data)
    logger.info(f"Loaded data from {len(all_participant_csv_data)} participants.")
    return all_participant_csv_data

def run_group_analyses(group_level_data_df, config, logger, group_level_analyzer, fdr_corrector, xml_reporter, output_dir):
    for section_name in config.sections():
        if section_name.startswith('GroupAnalysis_'):
            analysis_name = section_name.replace('GroupAnalysis_', '')
            analysis_config = dict(config.items(section_name))
            logger.info(f"Running group analysis step: {analysis_name}")
            input_data_key = analysis_config.get('input_data_key')
            if not input_data_key or input_data_key not in group_level_data_df:
                logger.error(f"Input data key '{input_data_key}' for analysis '{analysis_name}' not found. Skipping.")
                continue
            analysis_input_df = group_level_data_df[input_data_key]
            logger.info(f"DataFrame for analysis '{analysis_name}': shape={analysis_input_df.shape}, columns={list(analysis_input_df.columns)}")
            method_params = {k.replace('method_param_', ''): v for k, v in analysis_config.items() if k.startswith('method_param_')}
            required_cols = [method_params.get('dv'), method_params.get('subject')]
            missing_cols = [col for col in required_cols if col and col not in analysis_input_df.columns]
            if missing_cols:
                logger.error(f"Required columns {missing_cols} missing for analysis '{analysis_name}'. Skipping.")
                continue
            for param in ['within', 'between', 'grouping_vars']:
                if param in method_params and isinstance(method_params[param], str):
                    method_params[param] = [v.strip() for v in method_params[param].split(',')]
            analysis_step_config = {
                'analyzer_key': analysis_config.get('analyzer_key'),
                'method_to_call': analysis_config.get('method_to_call'),
                'method_params': method_params
            }
            try:
                analysis_results_df = group_level_analyzer.analyze_data(
                    data_df=analysis_input_df,
                    analysis_config=analysis_step_config,
                    task_name=analysis_name
                )
            except Exception as e:
                logger.error(f"Exception during analysis '{analysis_name}': {e}")
                continue
            if analysis_results_df is not None and not analysis_results_df.empty:
                if config.getboolean(section_name, 'apply_fdr', fallback=False):
                    p_val_col = analysis_config.get('p_value_col_name', 'p-unc')
                    if p_val_col in analysis_results_df.columns:
                        p_vals = pd.to_numeric(analysis_results_df[p_val_col], errors='coerce')
                        if not isinstance(p_vals, pd.Series):
                            p_vals = pd.Series(p_vals)
                        p_vals = p_vals.dropna()
                        if not p_vals.empty:
                            rejected, p_corrected = fdr_corrector.apply_fdr_correction(p_vals.to_numpy())
                            analysis_results_df[f'{p_val_col}_fdr'] = pd.Series(p_corrected, index=p_vals.index)
                            logger.info(f"Applied FDR correction on '{p_val_col}' column.")
                output_filename = analysis_config.get('output_filename_xml')
                if output_filename:
                    xml_reporter.save_dataframe(data_df=analysis_results_df, output_dir=output_dir, filename=output_filename)
                    logger.info(f"Saved analysis results to {os.path.join(output_dir, output_filename)}")

def log_provenance(config, logger, all_participant_csv_data, output_dir):
    import platform, datetime, importlib, hashlib
    logger.info("=== GROUP ANALYSIS PROVENANCE SUMMARY ===")
    logger.info(f"Date: {datetime.datetime.now().isoformat()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"pandas version: {pd.__version__}")
    logger.info(f"numpy version: {importlib.import_module('numpy').__version__}")
    config_file = os.path.join(os.path.dirname(__file__), "EV_config.cfg")
    logger.info(f"Config file: {config_file}")
    try:
        with open(config_file, 'rb') as f:
            config_hash = hashlib.md5(f.read()).hexdigest()
        logger.info(f"Config file hash (md5): {config_hash}")
    except Exception:
        logger.info("Config file hash: <unavailable>")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Participants included: {[p['participant_id'] for p in all_participant_csv_data]}")
    logger.info("Input files and hashes:")
    for pdata in all_participant_csv_data:
        for key in ['plv_results_df', 'fai_results_df', 'questionnaire_results_df']:
            fname = f"{pdata['participant_id']}_{key.replace('_df','')}.xml"
            fpath = os.path.join(config['Data']['base_output_dir'], pdata['participant_id'], fname)
            try:
                with open(fpath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                logger.info(f"  {fname}: {file_hash}")
            except Exception:
                logger.info(f"  {fname}: <unavailable>")
    logger.info("=== END GROUP ANALYSIS PROVENANCE ===")

def run_group_analysis():
    # === Configuration Loading ===
    config_file = os.path.join(os.path.dirname(__file__), "EV_config.cfg")
    config = ConfigParser()
    config.optionxform = lambda optionstr: optionstr
    if not os.path.exists(config_file):
        print(f"FATAL: Configuration file not found at {config_file}")
        return
    config.read(config_file)

    # === Logging Setup ===
    logging.basicConfig(level=config['DEFAULT'].get('log_level', 'INFO').upper(),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("GroupAnalysis")
    logger.info("--- Starting Modular Group Analysis Orchestrator ---")

    # === Output Directory Setup ===
    base_output_dir = config['Data']['base_output_dir']
    group_results_dir_name = config['GroupLevelSettings'].get('results_dir_name', 'EV_GroupResults')
    group_output_dir = os.path.join(base_output_dir, group_results_dir_name)
    os.makedirs(group_output_dir, exist_ok=True)
    logger.info(f"Group level results will be saved to: {group_output_dir}")

    # === Toolbox Module Instantiation ===
    anova_analyzer = ANOVAAnalyzer(logger=logger)
    xml_reporter = XMLReporter(logger=logger)
    fdr_corrector = FDRCorrector(logger=logger)
    available_analyzers = {"anova": anova_analyzer}
    group_level_analyzer = GroupLevelAnalyzing(logger=logger, available_analyzers=available_analyzers)

    # === 1. Load Individual Participant Results ===
    all_participant_csv_data = load_participant_results(config, logger)

    # === 2. Aggregate Data ===
    aggregator = GroupAggregator(logger=logger)
    group_level_data_df = aggregator.aggregate(
        all_participant_csv_data,
        config,
        output_dir=group_output_dir
    )

    # === 3. Analysis Steps (ANOVA, FDR, etc.) ===
    run_group_analyses(
        group_level_data_df,
        config,
        logger,
        group_level_analyzer,
        fdr_corrector,
        xml_reporter,
        group_output_dir
    )

    # === 4. Provenance Logging ===
    log_provenance(
        config,
        logger,
        all_participant_csv_data,
        group_output_dir
    )
    logger.info("--- Modular Group Analysis Orchestrator Finished ---")


if __name__ == "__main__":
    run_group_analysis()
