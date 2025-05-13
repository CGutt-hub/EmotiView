import os
import pandas as pd
import numpy as np # Added for placeholder metrics
import json # For saving/loading sampling rates

# Import configurations and utilities
from .. import config
from .. import utils

# Import modular components
from .data_handling.data_loader import DataLoader
from .data_handling.questionnaire_parser import QuestionnaireParser
from .preprocessing.eeg_preprocessor import EEGPreprocessor
from .preprocessing.ecg_preprocessor import ECGPreprocessor
from .preprocessing.eda_preprocessor import EDAPreprocessor
from .preprocessing.fnirs_preprocessor import FNIRSPreprocessor
from .analysis.psd_analyzer import PSDAnalyzer
from .analysis.hrv_analyzer import HRVAnalyzer
from .analysis.connectivity_analyzer import ConnectivityAnalyzer
from .analysis.fnirs_glm_analyzer import FNIRSGLMAnalyzer
# from .analysis.statistics.correlation_analyzer import CorrelationAnalyzer # For group stats later
# from .analysis.statistics.anova_analyzer import ANOVAAnalyzer # For group stats later
from .reporting.plotting_service import PlottingService


def run_participant_preprocessing(participant_id, p_logger, data_loader):
    """Handles data loading and preprocessing for a single participant."""
    p_logger.info(f"Preprocessing - Stage: Start for {participant_id}")
    participant_raw_data_path = os.path.join(config.PARTICIPANT_DATA_BASE_DIR, participant_id)
    preprocessed_output_dir = os.path.join(config.RESULTS_BASE_DIR, participant_id, "preprocessed")
    os.makedirs(preprocessed_output_dir, exist_ok=True)

    loaded_streams = data_loader.load_participant_streams(participant_id, participant_raw_data_path)
    processed_data_artifacts = {} # To store paths or MNE objects of processed data

    # EEG Preprocessing
    if loaded_streams.get('eeg') and loaded_streams.get('eeg_sfreq'):
        eeg_prep = EEGPreprocessor(p_logger)
        processed_eeg = eeg_prep.process(loaded_streams['eeg'], loaded_streams['eeg_sfreq'])
        if processed_eeg:
            eeg_file_path = os.path.join(preprocessed_output_dir, f"{participant_id}_eeg_preprocessed.fif")
            processed_eeg.save(eeg_file_path, overwrite=True, verbose=False)
            p_logger.info(f"Preprocessing - EEG saved to {eeg_file_path}")
            processed_data_artifacts['eeg_processed_mne_obj'] = processed_eeg # Keep in memory for analysis

    # fNIRS Preprocessing
    if loaded_streams.get('fnirs_od'):
        fnirs_prep = FNIRSPreprocessor(p_logger)
        processed_fnirs_haemo = fnirs_prep.process(loaded_streams['fnirs_od'])
        if processed_fnirs_haemo:
            fnirs_file_path = fnirs_prep.save_preprocessed_data(processed_fnirs_haemo, participant_id, preprocessed_output_dir)
            if fnirs_file_path:
                processed_data_artifacts['fnirs_haemo_mne_obj'] = processed_fnirs_haemo # Keep for analysis

    # ECG Preprocessing
    if loaded_streams.get('ecg_signal') is not None and loaded_streams.get('ecg_sfreq') is not None:
        ecg_prep = ECGPreprocessor(p_logger)
        nn_path, rpeaks_path = ecg_prep.process_and_save(loaded_streams['ecg_signal'], loaded_streams['ecg_sfreq'], participant_id, preprocessed_output_dir)
        if nn_path: processed_data_artifacts['ecg_nn_intervals_path'] = nn_path
        if rpeaks_path: processed_data_artifacts['ecg_rpeaks_times_path'] = rpeaks_path

    # EDA Preprocessing
    if loaded_streams.get('eda_signal') is not None and loaded_streams.get('eda_sfreq') is not None:
        eda_prep = EDAPreprocessor(p_logger)
        phasic_eda_path = eda_prep.process_and_save(loaded_streams['eda_signal'], loaded_streams['eda_sfreq'], participant_id, preprocessed_output_dir)
        if phasic_eda_path: processed_data_artifacts['eda_phasic_path'] = phasic_eda_path

    # Save sampling rates
    sampling_rates_info = {
        "eeg_sampling_rate": loaded_streams.get('eeg_sfreq'),
        "ecg_sampling_rate": loaded_streams.get('ecg_sfreq'),
        "eda_sampling_rate": loaded_streams.get('eda_sfreq'), # This is original, before resampling for PLV
        "fnirs_sampling_rate": loaded_streams.get('fnirs_sfreq')
    }
    sampling_rates_file = os.path.join(preprocessed_output_dir, f"{participant_id}_sampling_rates.json")
    with open(sampling_rates_file, 'w') as f_sr: json.dump(sampling_rates_info, f_sr, indent=4)
    p_logger.info(f"Preprocessing - Sampling rates info saved to {sampling_rates_file}")
    processed_data_artifacts['sampling_rates'] = sampling_rates_info # Make available for analysis

    p_logger.info(f"Preprocessing - Stage: End for {participant_id}")
    return processed_data_artifacts


def run_participant_analysis(participant_id, p_logger, processed_artifacts, questionnaire_data):
    """Handles analysis for a single participant using preprocessed data artifacts."""
    p_logger.info(f"Analysis - Stage: Start for {participant_id}")
    analysis_results_dir = os.path.join(config.RESULTS_BASE_DIR, participant_id, "analysis")
    os.makedirs(analysis_results_dir, exist_ok=True)
    
    analysis_metrics = {'participant_id': participant_id}
    if questionnaire_data and isinstance(questionnaire_data, dict): # Ensure it's a dict
        analysis_metrics.update(questionnaire_data)

    # PSD and FAI Analysis
    raw_eeg_obj = processed_artifacts.get('eeg_processed_mne_obj')
    if raw_eeg_obj:
        psd_analyzer = PSDAnalyzer(p_logger)
        psd_analyzer.calculate_psd_and_fai(raw_eeg_obj, analysis_metrics)

    # HRV Analysis
    hrv_analyzer = HRVAnalyzer(p_logger)
    if processed_artifacts.get('ecg_nn_intervals_path'):
        hrv_analyzer.calculate_hrv_metrics(processed_artifacts['ecg_nn_intervals_path'], analysis_metrics)
    
    phase_hrv, target_time_hrv = None, None
    if processed_artifacts.get('ecg_rpeaks_times_path') and processed_artifacts.get('ecg_nn_intervals_path'):
        phase_hrv, target_time_hrv = hrv_analyzer.get_hrv_phase_signal(
            processed_artifacts['ecg_rpeaks_times_path'], 
            processed_artifacts['ecg_nn_intervals_path']
        )

    # Connectivity Analysis (PLV)
    if raw_eeg_obj: # Requires preprocessed EEG
        conn_analyzer = ConnectivityAnalyzer(p_logger)
        phasic_eda_signal_for_plv = None
        eda_original_sfreq = processed_artifacts.get('sampling_rates', {}).get('eda_sampling_rate')
        if processed_artifacts.get('eda_phasic_path') and eda_original_sfreq:
            try:
                phasic_eda_signal_for_plv = pd.read_csv(processed_artifacts['eda_phasic_path'])['EDA_Phasic'].values
            except Exception as e:
                p_logger.warning(f"Could not load phasic EDA for PLV: {e}")
        
        conn_analyzer.calculate_all_plv(raw_eeg_obj,
                                        phase_hrv, target_time_hrv,
                                        phasic_eda_signal_for_plv, eda_original_sfreq,
                                        analysis_metrics)
    # fNIRS GLM Analysis
    fnirs_haemo_obj = processed_artifacts.get('fnirs_haemo_mne_obj')
    if fnirs_haemo_obj:
        fnirs_glm_analyzer = FNIRSGLMAnalyzer(p_logger)
        fnirs_glm_analyzer.run_glm_and_extract_rois(fnirs_haemo_obj, analysis_metrics, participant_id, analysis_results_dir)

    # Save metrics
    metrics_df = pd.DataFrame([analysis_metrics])
    metrics_file = os.path.join(analysis_results_dir, f"{participant_id}_pilot_analysis_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    p_logger.info(f"Analysis - Metrics saved to {metrics_file}")

    # Plotting
    plotter = PlottingService(p_logger)
    if raw_eeg_obj:
        plotter.plot_eeg_psd(raw_eeg_obj, participant_id, analysis_results_dir)
    # plotter.plot_fnirs_glm_contrast(...) # Add if GLM analyzer saves contrast objects

    p_logger.info(f"Analysis - Stage: End for {participant_id}")
    return analysis_metrics


def run_pilot_pipeline():
    """Main orchestrator for the pilot data processing pipeline."""
    utils.main_logger.info("===== EmotiView Pilot Data Pipeline Started =====")
    
    data_loader = DataLoader(utils.main_logger) 
    q_parser = QuestionnaireParser(utils.main_logger)

    all_participants_metrics_list = []
    all_questionnaires_list = []
    
    participant_ids = [d for d in os.listdir(config.PARTICIPANT_DATA_BASE_DIR) if os.path.isdir(os.path.join(config.PARTICIPANT_DATA_BASE_DIR, d))]
    if not participant_ids:
        utils.main_logger.info("No participant subfolders found. Exiting.")
        return

    utils.main_logger.info(f"Found participants: {', '.join(sorted(participant_ids))}")

    for p_id in sorted(participant_ids):
        p_logger = utils.get_participant_logger(p_id)
        utils.main_logger.info(f"--- Processing Participant: {p_id} ---")
        
        participant_raw_path = os.path.join(config.PARTICIPANT_DATA_BASE_DIR, p_id)
        questionnaire = q_parser.parse(p_id, participant_raw_path)
        if questionnaire and questionnaire.get('participant_id'): # Ensure valid questionnaire data
             if not any(isinstance(d, dict) and d.get('participant_id') == p_id for d in all_questionnaires_list):
                all_questionnaires_list.append(questionnaire)
        
        processed_artifacts = run_participant_preprocessing(p_id, p_logger, data_loader)
        
        if not processed_artifacts: 
            p_logger.error(f"Preprocessing failed or yielded no usable data artifacts for {p_id}. Skipping analysis.")
            utils.close_participant_logger(p_id)
            continue
            
        metrics = run_participant_analysis(p_id, p_logger, processed_artifacts, questionnaire)
        all_participants_metrics_list.append(metrics)
        
        utils.close_participant_logger(p_id)
        utils.main_logger.info(f"--- Finished Participant: {p_id} ---")

    if all_participants_metrics_list:
        final_metrics_df = pd.DataFrame(all_participants_metrics_list)
        aggregated_metrics_file = os.path.join(config.RESULTS_BASE_DIR, "pilot_all_participants_metrics.xlsx")
        final_metrics_df.to_excel(aggregated_metrics_file, index=False)
        utils.main_logger.info(f"All participant metrics aggregated and saved to: {aggregated_metrics_file}")

    if all_questionnaires_list:
        q_df = pd.DataFrame(all_questionnaires_list)
        q_excel_path = os.path.join(config.RESULTS_BASE_DIR, config.AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME)
        q_df.to_excel(q_excel_path, index=False)
        utils.main_logger.info(f"Aggregated questionnaires saved to: {q_excel_path}")

    utils.main_logger.info("===== EmotiView Pilot Data Pipeline Finished =====")
    utils.main_logger.info(f"Main log file: {utils.main_log_file_path}")

if __name__ == "__main__":
    run_pilot_pipeline()