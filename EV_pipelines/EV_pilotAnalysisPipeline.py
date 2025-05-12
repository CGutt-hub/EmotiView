import os
import subprocess
import logging
import shutil
import datetime
import pandas as pd # For handling results
import numpy as np
import mne
import json # For saving/loading sampling rates
import neurokit2 as nk
from scipy.signal import hilbert, welch # For Hilbert transform to get phase, and PSD
from scipy.interpolate import interp1d # For resampling/interpolating signals
import matplotlib.pyplot as plt # For plotting
import mne_nirs # For fNIRS specific functions

# --- Configuration ---
# Ensure BASE_REPO_PATH points to the root of your Git repository.
BASE_REPO_PATH = r"D:\repoShaggy\EmotiView"

# Directory where raw data for each participant is stored.
# The script expects subdirectories here, one for each participant_id.
PARTICIPANT_DATA_BASE_DIR = os.path.join(BASE_REPO_PATH, "rawData", "pilotData") # For testing with pilotData

# Base directory to save processed data and analysis results.
# Results for each participant will be stored in a subdirectory named after their ID.
# Updated to EV_results:
RESULTS_BASE_DIR = os.path.join(BASE_REPO_PATH, "EV_results")

# Directory to store log files.
LOG_DIR = os.path.join(BASE_REPO_PATH, "logs", "pipeline_runs")

# --- Questionnaire Configuration ---
QUESTIONNAIRE_TXT_FILENAME = "questionnaire.txt" # Name of the questionnaire text file in each participant's raw data folder
AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME = "aggregated_questionnaires.xlsx"
# MONITOR_INTERVAL_SECONDS = 60 # Not used in non-automated version

# Ensure base directories exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Main Pipeline Logger Setup ---
main_log_file = os.path.join(LOG_DIR, f"pipeline_main_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file),
        logging.StreamHandler() # Also log to console
    ]
)
main_logger = logging.getLogger("MainPipeline")

# --- Participant-specific Logger ---
participant_loggers = {} # Stores active participant loggers

def setup_participant_logger(participant_id, participant_log_directory_path):
    """
    Sets up a specific logger for a participant, writing to a log file
    named EV_{participant_id}_processingLog.txt in their raw data directory.
    """
    if participant_id in participant_loggers:
        return participant_loggers[participant_id]

    # Use a logger name that reflects the desired output structure
    logger_name = f"EV_{participant_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # Capture all levels for participant log
    logger.propagate = False # Prevent logs from bubbling up to the main logger

    # Ensure the directory for the log file exists (should be created by run_pipeline before this)
    os.makedirs(participant_log_directory_path, exist_ok=True)

    # Log filename remains the same, but the logger name used in formatting will be different
    log_filename = f"EV_{participant_id}_processingLog.txt" 
    participant_log_file_path = os.path.join(participant_log_directory_path, log_filename)

    # Formatter for the participant's file log
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(participant_log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Optional: Add a stream handler for console output specific to this participant logger
    # This ensures its messages also go to console if desired, formatted with its name.
    # If the main_logger's StreamHandler is sufficient, this can be omitted.
    # console_formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    # ch_participant = logging.StreamHandler()
    # ch_participant.setLevel(logging.INFO) # Or DEBUG
    # ch_participant.setFormatter(console_formatter)
    # logger.addHandler(ch_participant)
    
    participant_loggers[participant_id] = logger
    main_logger.info(f"Participant processing log for {participant_id} set up at: {participant_log_file_path}")
    return logger

def close_participant_logger(participant_id):
    """Closes file handlers for a participant-specific logger to release resources."""
    if participant_id in participant_loggers:
        logger = participant_loggers[participant_id]
        for handler in logger.handlers[:]: # Iterate over a copy
            handler.close()
            logger.removeHandler(handler)
        del participant_loggers[participant_id]
        main_logger.info(f"Participant logger for {participant_id} closed.")

# --- Git Functions ---
def run_git_command(command_args, cwd, logger_to_use):
    """Runs a Git command using subprocess and logs its output."""
    full_command = ['git'] + command_args
    logger_to_use.info(f"Running Git command: {' '.join(full_command)} in {cwd}")
    try:
        process = subprocess.run(full_command, cwd=cwd, capture_output=True, text=True, check=True, encoding='utf-8')
        if process.stdout:
            logger_to_use.debug(f"Git command stdout:\n{process.stdout.strip()}")
        if process.stderr: # Git often uses stderr for info messages too
            logger_to_use.debug(f"Git command stderr:\n{process.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger_to_use.error(f"Git command failed: {' '.join(full_command)}")
        if e.stdout:
            logger_to_use.error(f"Stdout:\n{e.stdout.strip()}")
        if e.stderr:
            logger_to_use.error(f"Stderr:\n{e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logger_to_use.error("Git command not found. Is Git installed and in your system's PATH?")
        return False
    except Exception as e:
        logger_to_use.error(f"An unexpected error occurred while running git command: {e}", exc_info=True)
        return False

def git_pull(repo_path, logger_to_use):
    """Pulls the latest changes from the remote repository."""
    logger_to_use.info(f"Pulling latest changes for repository at {repo_path}...")
    return run_git_command(['pull'], cwd=repo_path, logger_to_use=logger_to_use)

def git_add_commit_push(repo_path, commit_message, files_to_add, logger_to_use):
    """Adds specified files, commits, and pushes to the remote repository."""
    logger_to_use.info(f"Preparing to commit and push changes for: {commit_message}")

    for file_path_or_pattern in files_to_add:
        if not run_git_command(['add', file_path_or_pattern], cwd=repo_path, logger_to_use=logger_to_use):
            logger_to_use.error(f"Failed to add '{file_path_or_pattern}' to Git. Aborting commit for this participant.")
            return False
    
    status_process = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_path, capture_output=True, text=True)
    if not status_process.stdout.strip():
        logger_to_use.info("No changes to commit.")
        return True 

    if not run_git_command(['commit', '-m', commit_message], cwd=repo_path, logger_to_use=logger_to_use):
        logger_to_use.error("Failed to commit changes. Aborting push.")
        return False
    
    if not run_git_command(['push'], cwd=repo_path, logger_to_use=logger_to_use):
        logger_to_use.error("Failed to push changes.")
        return False
    
    logger_to_use.info("Successfully committed and pushed changes.")
    return True

# --- Helper Function for PLV (Simplified Placeholder) ---
def calculate_plv(phase1, phase2, p_logger_local):
    """
    Calculates Phase Locking Value (PLV) between two phase time series.
    This is a simplified placeholder. Actual implementation needs careful
    handling of signal alignment, windowing, and epoching based on events.
    """
    # Ensure phase1 and phase2 are aligned and of same length for the window of interest
    # This requires careful time alignment of EEG epochs with continuous autonomic signals
    min_len = min(len(phase1), len(phase2))
    if min_len == 0:
        p_logger_local.warning(f"PLV - Status: Skipped (ZeroLengthSignal)") # Participant ID from logger name
        return np.nan
    
    # Truncate to the minimum length
    phase1_aligned = phase1[:min_len]
    phase2_aligned = phase2[:min_len]

    # Calculate phase difference
    phase_difference = phase1_aligned - phase2_aligned

    # Calculate PLV
    plv = np.abs(np.mean(np.exp(1j * phase_difference)))
    
    p_logger_local.debug(f"PLV - Value: {plv:.4f}") # Participant ID from logger name
    return plv

# --- Plotting Functions ---

def generate_participant_plots(participant_id, analysis_results_dir, p_logger):
    """
    Generates plots specific to a single participant (e.g., PSD, time series).
    Saves plots to the participant's analysis results directory.
    """
    p_logger.info(f"EV_{participant_id} - Plotting - Action: GenerateParticipantPlots")
    try:
        # Example: Plot preprocessed EEG PSD
        eeg_preprocessed_file = os.path.join(RESULTS_BASE_DIR, participant_id, "preprocessed", f"{participant_id}_eeg_preprocessed.fif")
        if os.path.exists(eeg_preprocessed_file):
            raw_eeg = mne.io.read_raw_fif(eeg_preprocessed_file, preload=True)
            p_logger.info(f"Plotting - Action: ComputePSDEEG")
            fig_psd = raw_eeg.compute_psd(picks='eeg', fmax=40, n_fft=2048).plot(show=False, average=True)
            psd_plot_path = os.path.join(analysis_results_dir, f"{participant_id}_eeg_psd.png")
            fig_psd.savefig(psd_plot_path)
            plt.close(fig_psd)
            p_logger.info(f"Plotting - Action: SavePlot - File: {psd_plot_path}")

        # Example: Plot fNIRS GLM contrast map (if GLM results are saved as an Evoked object or similar)
        # fnirs_glm_contrast_file = os.path.join(analysis_results_dir, f"{participant_id}_fnirs_glm_contrast_hbo.fif")
        # if os.path.exists(fnirs_glm_contrast_file):
        #     contrast_evoked = mne.read_evokeds(fnirs_glm_contrast_file)[0] # Assuming one contrast saved
        #     p_logger.info(f"Plotting - Action: PlotfNIRSContrast")
        #     # TODO: Define sensor positions for plotting on a head model
        #     # montage = mne.channels.make_standard_montage('standard_1020') # Or a specific fNIRS montage
        #     # contrast_evoked.set_montage(montage)
        #     fig_contrast = contrast_evoked.plot_topomap(times=[0], # Assuming contrast is at a single time point (e.g., peak)
        #                                                 ch_type='hbo', # or 'fnirs_od' if plotting OD
        #                                                 show=False,
        #                                                 title=f"{participant_id} - HbO Contrast")
        #     contrast_plot_path = os.path.join(analysis_results_dir, f"{participant_id}_fnirs_hbo_contrast_map.png")
        #     fig_contrast.savefig(contrast_plot_path)
        #     plt.close(fig_contrast)
        #     p_logger.info(f"Plotting - Action: SavePlot - File: {contrast_plot_path}")

        # TODO: Add other participant-specific plots (e.g., autonomic signals time series, ICA components)

        p_logger.info(f"EV_{participant_id} - Plotting - Status: ParticipantPlotsGenerated")
    except Exception as e:
        p_logger.error(f"Plotting - Error: {e}", exc_info=True)

def generate_group_plots(all_participants_metrics_df, results_base_dir, main_logger_to_use): # main_logger_to_use is the main_logger
    """
    Generates or updates group-level plots based on aggregated metrics.
    Saves plots to a central location (e.g., RESULTS_BASE_DIR).
    """
    main_logger_to_use.info("Plotting - Action: GenerateGroupPlots")
    if all_participants_metrics_df.empty:
        main_logger_to_use.info("Plotting - Status: Skipped (NoAggregatedMetrics)")
        return
    try:
        # Example: Plot group-averaged PLV for different conditions
        # Assuming metrics like 'plv_alpha_eeg_hrv_Positive', 'plv_alpha_eeg_hrv_Negative', etc. exist
        plv_columns = [col for col in all_participants_metrics_df.columns if 'plv_' in col and '_Positive' in col or '_Negative' in col or '_Neutral' in col]
        if plv_columns:
            conditions = ['Positive', 'Negative', 'Neutral']
            bands = ['alpha', 'beta'] # Example bands
            autonomics = ['eeg_hrv', 'eeg_eda'] # Example pairs
            
            for band in bands:
                for autonomic_pair in autonomics:
                    current_plv_means = []
                    valid_conditions_for_plot = []
                    for cond in conditions:
                        col_name = f"plv_{band}_{autonomic_pair}_{cond}"
                        if col_name in all_participants_metrics_df.columns and not all_participants_metrics_df[col_name].isnull().all():
                            current_plv_means.append(all_participants_metrics_df[col_name].mean())
                            valid_conditions_for_plot.append(cond)
                    
                    if current_plv_means:
                        fig, ax = plt.subplots()
                        ax.bar(valid_conditions_for_plot, current_plv_means)
                        ax.set_ylabel(f'Mean PLV ({band.capitalize()} {autonomic_pair.replace("_", "-")})')
                        ax.set_title(f'Group PLV: {band.capitalize()} {autonomic_pair.replace("_", "-")} by Condition')
                        group_plv_plot_path = os.path.join(results_base_dir, f"group_plv_{band}_{autonomic_pair}.png")
                        fig.savefig(group_plv_plot_path)
                        plt.close(fig)
                        main_logger_to_use.info(f"Plotting - Action: SavePlot - File: {group_plv_plot_path}")

        # TODO: Add other group-level plots (e.g., correlation matrices, ANOVA results visualizations)

        main_logger_to_use.info("Plotting - Status: GroupPlotsGenerationComplete")
    except Exception as e:
        main_logger_to_use.error(f"Plotting - GroupPlots - Error: {e}", exc_info=True)


# --- Questionnaire Parsing Function ---
def parse_questionnaire_data(participant_id, participant_raw_data_path, p_logger):
    """
    Parses a questionnaire.txt file for a participant.
    Assumes a 'Key: Value' format per line in the text file.
    Returns a dictionary with parsed data, including participant_id, or None on error/not found.
    """
    questionnaire_file_path = os.path.join(participant_raw_data_path, QUESTIONNAIRE_TXT_FILENAME)
    parsed_data = {'participant_id': participant_id}

    if not os.path.exists(questionnaire_file_path):
        p_logger.info(f"Questionnaire - Status: FileNotFound - File: {questionnaire_file_path}")
        return None

    try:
        p_logger.info(f"Questionnaire - Action: ParseFile - File: {questionnaire_file_path}")
        with open(questionnaire_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or ':' not in line:
                    if line: # Non-empty line without colon
                        p_logger.warning(f"Questionnaire - Warning: SkippingMalformedLine - LineNum: {line_num} - Content: '{line}'")
                    continue
                key, value = line.split(':', 1)
                parsed_data[key.strip()] = value.strip()
        # TODO: Convert specific questionnaire scores to numeric. Handle potential errors during conversion.
        # Example conversions (ensure these keys match your questionnaire.txt):
        # try:
        #     if 'SAM_arousal' in parsed_data: parsed_data['SAM_arousal'] = int(parsed_data['SAM_arousal'])
        #     if 'SAM_valence' in parsed_data: parsed_data['SAM_valence'] = int(parsed_data['SAM_valence'])
        #     if 'BIS_score' in parsed_data: parsed_data['BIS_score'] = int(parsed_data['BIS_score'])
        #     if 'BAS_drive' in parsed_data: parsed_data['BAS_drive'] = int(parsed_data['BAS_drive'])
        #     if 'PANAS_Positive' in parsed_data: parsed_data['PANAS_Positive'] = int(parsed_data['PANAS_Positive'])
        #     if 'PANAS_Negative' in parsed_data: parsed_data['PANAS_Negative'] = int(parsed_data['PANAS_Negative'])
        # except ValueError as ve:
        #     p_logger.warning(f"Questionnaire - Warning: Could not convert a score to numeric for {participant_id} - {ve}")
        p_logger.info(f"Questionnaire - Status: ParsedSuccessfully")
        return parsed_data
    except Exception as e:
        p_logger.error(f"Questionnaire - Error: ParsingFailed - File: {questionnaire_file_path} - Details: {e}", exc_info=True)
        return None

# --- Placeholder Processing Functions ---

def run_preprocessing(participant_id, raw_data_path, processed_data_output_dir, p_logger):
    """
    Participant data preprocessing based on EV_proposal.tex.
    Saves preprocessed files into `processed_data_output_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"==== Preprocessing ====") 
    p_logger.debug(f"Config - RawDataPath: {raw_data_path}")
    p_logger.debug(f"Config - ProcessedDataOutputPath: {processed_data_output_dir}")
    
    os.makedirs(processed_data_output_dir, exist_ok=True)

    try:
        if not os.path.exists(raw_data_path):
            p_logger.error(f"Error: RawDataPathNotFound - Path: {raw_data_path}")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error(f"Error: LibrariesNotLoaded - Details: NumPy/MNE missing")
            return False

        # --- Data Loading ---
        # For a non-automated test version, explicitly define how to load each modality.
        # These are placeholders; you'll need to adapt them to your pilotData file structure and types.
        raw_eeg, ecg_signal_raw, eda_signal_raw, fnirs_raw_od = None, None, None, None # Initialize
        eeg_sampling_rate, ecg_sampling_rate, eda_sampling_rate = None, None, None
        combined_data_loaded = False

        # --- Load Combined LSL Stream Data (EEG, ECG, EDA) ---
        # Adapt this to your actual file format (e.g., .vhdr, .xdf from LSL)
        # For this example, we'll assume a BrainVision file contains all three.
        combined_stream_file = os.path.join(raw_data_path, f"{participant_id}_combined_lsl.vhdr") # ADJUST FILENAME
        if os.path.exists(combined_stream_file):
            p_logger.info(f"CombinedStream - Action: LoadFile - File: {combined_stream_file}")
            raw_combined = mne.io.read_raw_brainvision(combined_stream_file, preload=True) # Or read_raw_xdf if it's an XDF
            combined_sampling_rate = raw_combined.info['sfreq']
            p_logger.info(f"CombinedStream - Status: Loaded - SFreq: {combined_sampling_rate} Hz")
            combined_data_loaded = True

            # Channel indices are 0-based. So 7th channel is index 6, 8th is index 7.
            eda_channel_index = 6 
            ecg_channel_index = 7
            # Assuming EEG channels start immediately after ECG
            eeg_start_channel_index = 8 

            # Extract EDA
            if len(raw_combined.ch_names) > eda_channel_index:
                eda_ch_name = raw_combined.ch_names[eda_channel_index]
                eda_signal_raw = raw_combined.get_data(picks=[eda_ch_name])[0]
                eda_sampling_rate = combined_sampling_rate
                p_logger.info(f"EDA - Action: ExtractFromCombined - ChannelName: {eda_ch_name} (Index: {eda_channel_index})")
            else:
                p_logger.warning(f"EDA - Status: ChannelIndexOutOfRange - ExpectedIndex: {eda_channel_index}")

            # Extract ECG
            if len(raw_combined.ch_names) > ecg_channel_index:
                ecg_ch_name = raw_combined.ch_names[ecg_channel_index]
                ecg_signal_raw = raw_combined.get_data(picks=[ecg_ch_name])[0]
                ecg_sampling_rate = combined_sampling_rate
                p_logger.info(f"ECG - Action: ExtractFromCombined - ChannelName: {ecg_ch_name} (Index: {ecg_channel_index})")
            else:
                p_logger.warning(f"ECG - Status: ChannelIndexOutOfRange - ExpectedIndex: {ecg_channel_index}")

            # Prepare EEG data (all channels from eeg_start_channel_index onwards)
            if len(raw_combined.ch_names) > eeg_start_channel_index:
                eeg_ch_names = raw_combined.ch_names[eeg_start_channel_index:]
                if eeg_ch_names: # Ensure there are EEG channels to pick
                    raw_eeg = raw_combined.copy().pick_channels(eeg_ch_names)
                    # Set channel types for EEG if not already correct (MNE might infer them)
                    # raw_eeg.set_channel_types({ch_name: 'eeg' for ch_name in eeg_ch_names})
                    eeg_sampling_rate = combined_sampling_rate
                    p_logger.info(f"EEG - Action: ExtractFromCombined - Channels: {len(eeg_ch_names)} (StartingIndex: {eeg_start_channel_index})")
                else:
                    p_logger.warning(f"EEG - Status: NoEEGChannelsFoundAfterIndex - StartIndex: {eeg_start_channel_index}")
            else:
                p_logger.warning(f"EEG - Status: ChannelIndexOutOfRangeForEEGStart - ExpectedStartIndex: {eeg_start_channel_index}")
        else:
            p_logger.warning(f"CombinedStream - Status: FileNotFound - File: {combined_stream_file}")

        # Example: Load fNIRS from a .nirs file (or similar MNE-NIRS compatible format)
        fnirs_data_file = os.path.join(raw_data_path, f"{participant_id}_fnirs.nirs") # Adjust filename as needed
        if os.path.exists(fnirs_data_file):
            p_logger.info(f"fNIRS - Action: LoadFile - File: {fnirs_data_file}")
            # Use the appropriate MNE-NIRS reader, e.g., read_raw_nirx, read_raw_snirf, etc.
            # This example assumes a generic reader or that it's already optical density.
            # If it's raw intensity, you'll need mne_nirs.optical_density() later.
            fnirs_raw_od = mne.io.read_raw_nirx(fnirs_data_file, preload=True) # Replace with actual reader
            p_logger.info(f"fNIRS - Status: Loaded - SFreq: {fnirs_raw_od.info['sfreq']} Hz")
            # If the loaded fNIRS data is not yet optical density, convert it:
            # fnirs_raw_od = mne_nirs.optical_density(fnirs_raw_od_or_intensity)
        else:
            p_logger.warning(f"fNIRS - Status: FileNotFound - File: {fnirs_data_file}")

        # --- EEG Preprocessing (MNE-Python) ---
        if raw_eeg is not None:
            p_logger.info(f"\nEEG - Stage: Preprocessing")

            p_logger.info(f"EEG - Action: LoadDataToMemory") # If not already preloaded
            raw_eeg.load_data() # Explicitly load data for the EEG stream

            p_logger.info(f"EEG - Action: Filter - Params: 0.5-40Hz")
            raw_eeg.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)

            p_logger.info(f"EEG - Action: SetReference - Type: Average")
            raw_eeg.set_eeg_reference('average', projection=True)

            p_logger.info(f"EEG - Action: FitICA - Components: 15")
            ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
            ica.fit(raw_eeg)

            p_logger.info(f"EEG - Action: AutoLabelICArtifacts")
            try:
                from mne_icalabel import label_components # Try import here
                label_components(raw_eeg, ica, method='iclabel')
                # Exclude components labeled as artifacts by ICLabel
                # Common artifact labels: 'eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise'
                # 'brain' and 'other' are typically kept.
                ica.exclude = [idx for idx, label in enumerate(ica.labels_) 
                               if label not in ['brain', 'other']]
                p_logger.info(f"EEG - ICLabel - ExcludedIndices: {ica.exclude} - Labels: {[ica.labels_[i] for i in ica.exclude]}")
            except ImportError:
                p_logger.warning(f"EEG - ICLabel - Status: PackageNotAvailable (mne_icalabel). Manual ICA component selection would be needed.")
            except Exception as e_icalabel:
                p_logger.error(f"EEG - ICLabel - Error: {e_icalabel}. Manual ICA component selection needed.", exc_info=True)

            ica.apply(raw_eeg, verbose=False)
            p_logger.info(f"EEG - Action: ApplyICA - ExcludedComponents: {ica.exclude}")

            preprocessed_eeg_file = os.path.join(processed_data_output_dir, f"{participant_id}_eeg_preprocessed.fif")
            raw_eeg.save(preprocessed_eeg_file, overwrite=True)
            p_logger.info(f"EEG - Action: SavePreprocessed - File: {preprocessed_eeg_file}")
        else:
            p_logger.info(f"EEG - Status: Skipped (NoData)")

        # --- ECG Preprocessing (NeuroKit2) ---
        if ecg_signal_raw is not None and ecg_sampling_rate is not None:
            p_logger.info(f"\nECG - Stage: Preprocessing")

            p_logger.info(f"ECG - Action: CleanSignal")
            ecg_cleaned = nk.ecg_clean(ecg_signal_raw, sampling_rate=ecg_sampling_rate)

            p_logger.info(f"ECG - Action: DetectRPeaks")
            peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate, correct_artifacts=True)
            rpeaks = peaks_info[0]['ECG_R_Peaks']
            p_logger.info(f"ECG - Status: RPeaksDetected - Count: {len(rpeaks)}")

            p_logger.info(f"ECG - Action: CalculateNNIntervals")
            nn_intervals = np.diff(rpeaks) / ecg_sampling_rate * 1000 # in ms
            nn_intervals_file = os.path.join(processed_data_output_dir, f"{participant_id}_nn_intervals.csv")
            pd.DataFrame(nn_intervals, columns=['NN_ms']).to_csv(nn_intervals_file, index=False)
            p_logger.info(f"ECG - Action: SaveNNIntervals - File: {nn_intervals_file}")

            rpeaks_times_file = os.path.join(processed_data_output_dir, f"{participant_id}_rpeaks_times_sec.csv")
            pd.DataFrame(rpeaks / ecg_sampling_rate, columns=['R_Peak_Time_s']).to_csv(rpeaks_times_file, index=False)
            p_logger.info(f"ECG - Action: SaveRPeakTimes - File: {rpeaks_times_file}")
        else:
            p_logger.info(f"ECG - Status: Skipped (NoData)")

        # --- EDA Preprocessing (NeuroKit2) ---
        if eda_signal_raw is not None and eda_sampling_rate is not None:
            p_logger.info(f"\nEDA - Stage: Preprocessing")

            p_logger.info(f"EDA - Action: ProcessSignal (Decomposition)")
            eda_signals, info = nk.eda_process(eda_signal_raw, sampling_rate=eda_sampling_rate)
            phasic_eda = eda_signals['EDA_Phasic']
            tonic_eda = eda_signals['EDA_Tonic']
            p_logger.info(f"EDA - Status: Decomposed")

            phasic_eda_file = os.path.join(processed_data_output_dir, f"{participant_id}_phasic_eda.csv")
            if isinstance(phasic_eda, (pd.Series, pd.DataFrame)):
                phasic_eda.to_csv(phasic_eda_file, index=False, header=True)
            else: # If it's a numpy array
                pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_file, index=False, header=True)
            p_logger.info(f"EDA - Action: SavePhasicEDA - File: {phasic_eda_file}")
        else:
            p_logger.info(f"EDA - Status: Skipped (NoData)")

        # --- fNIRS Preprocessing (MNE-NIRS) ---
        if fnirs_raw_od is not None: # fnirs_raw_od is the MNE Raw object for optical density from XDF
            p_logger.info(f"\nfNIRS - Stage: Preprocessing")

            p_logger.info(f"fNIRS - Action: LoadDataToMemory") # If not already preloaded
            fnirs_raw_od.load_data() # Explicitly load data for the fNIRS OD stream
            # Ensure it's treated as optical density if it's not already marked as such by read_raw_xdf
            # This might involve checking channel types or manually setting them if necessary.
            # For now, assume fnirs_raw_od is ready for Beer-Lambert or is raw voltages needing optical_density()
            raw_od_for_beer_lambert = fnirs_raw_od # Default assumption
            
            # If your XDF fNIRS stream provides raw voltages, you might need mne_nirs.optical_density() first
            # Example: if 'fnirs_raw' in fnirs_raw_od.info.get('ch_types', []): # Hypothetical check
            #    raw_od_for_beer_lambert = mne_nirs.optical_density(fnirs_raw_od)
            #    p_logger.info(f"fNIRS - Converted fNIRS raw voltages to optical density.")
            raw_od_for_beer_lambert.info['bads'] = [] # Mark bad channels if any
            # Convert to HbO/HbR
            raw_haemo = raw_od_for_beer_lambert # If already OD from XDF and correctly typed by MNE
            # If it's raw data that needs conversion to OD first, then Beer-Lambert:
            # raw_haemo = mne_nirs.optical_density(fnirs_raw_od) # if fnirs_raw_od is raw sensor data
            # p_logger.info(f"fNIRS - Converted fNIRS stream to optical density (if needed).")
            raw_haemo = mne_nirs.beer_lambert_law(raw_haemo, ppf=6.0)
            p_logger.info(f"fNIRS - Action: ApplyBeerLambert")

            # Motion correction (TDDR)
            corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            p_logger.info(f"fNIRS - Action: ApplyTDDR")

            # Filtering
            corrected_haemo.filter(0.01, 0.1, h_trans_bandwidth=0.02, l_trans_bandwidth=0.002, fir_design='firwin', verbose=False)
            p_logger.info(f"fNIRS - Action: Filter - Params: 0.01-0.1Hz")

            preprocessed_fnirs_file = os.path.join(processed_data_output_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
            corrected_haemo.save(preprocessed_fnirs_file, overwrite=True)
            p_logger.info(f"fNIRS - Action: SavePreprocessed - File: {preprocessed_fnirs_file}")
        else:
            p_logger.info(f"fNIRS - Status: Skipped (NoData)")

        # Dummy file to indicate completion
        manifest_file_path = os.path.join(processed_data_output_dir, f"{participant_id}_preprocessing_manifest.txt")
        with open(manifest_file_path, "w") as f:
            f.write(f"Preprocessing completed for {participant_id} at {datetime.datetime.now().isoformat()}\n")
            # Example: List successfully created files
            # if 'preprocessed_eeg_file' in locals() and os.path.exists(preprocessed_eeg_file): f.write(f"- EEG: {os.path.basename(preprocessed_eeg_file)}\n")
            # if 'nn_intervals_file' in locals() and os.path.exists(nn_intervals_file): f.write(f"- ECG NN-Intervals: {os.path.basename(nn_intervals_file)}\n")
            # if 'phasic_eda_file' in locals() and os.path.exists(phasic_eda_file): f.write(f"- EDA Phasic: {os.path.basename(phasic_eda_file)}\n")
            # if 'preprocessed_fnirs_file' in locals() and os.path.exists(preprocessed_fnirs_file): f.write(f"- fNIRS Hb: {os.path.basename(preprocessed_fnirs_file)}\n")
        p_logger.info(f"Preprocessing manifest saved to {manifest_file_path}")

        # Save sampling rates
        sampling_rates_info = {
            "eeg_sampling_rate": eeg_sampling_rate,
            "ecg_sampling_rate": ecg_sampling_rate,
            "eda_sampling_rate": eda_sampling_rate,
            "fnirs_sampling_rate": fnirs_raw_od.info['sfreq'] if fnirs_raw_od else None
        }
        sampling_rates_file = os.path.join(processed_data_output_dir, f"{participant_id}_sampling_rates.json")
        with open(sampling_rates_file, 'w') as f_sr:
            json.dump(sampling_rates_info, f_sr, indent=4)
        p_logger.info(f"Sampling rates info saved to {sampling_rates_file}")
        p_logger.info(f"==== END Preprocessing - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"Preprocessing - Error: {e}", exc_info=True)
        p_logger.info(f"==== END Preprocessing - Status: FAILURE ====")
        return False

def run_analysis(participant_id, preprocessed_data_dir, analysis_results_dir, questionnaire_data, p_logger):
    """
    Participant data analysis based on EV_proposal.tex.
    Saves analysis metrics and reports into `analysis_results_dir`.
    Returns True on success, False on failure.
    """
    # `questionnaire_data` is the dictionary returned by parse_questionnaire_data()
    p_logger.info(f"==== Analysis ====") 
    p_logger.debug(f"Config - PreprocessedDataPath: {preprocessed_data_dir}")
    p_logger.debug(f"Config - AnalysisResultsPath: {analysis_results_dir}")

    os.makedirs(analysis_results_dir, exist_ok=True)
    analysis_metrics = {'participant_id': participant_id} 

    try:
        if not os.path.exists(preprocessed_data_dir):
            p_logger.error(f"Error: PreprocessedDataPathNotFound - Path: {preprocessed_data_dir}")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error(f"Error: LibrariesNotLoaded - Details: NumPy/MNE missing")
            return False

        # --- Load Preprocessed Data ---
        p_logger.info(f"\nStage: LoadPreprocessedData")
        eeg_file = os.path.join(preprocessed_data_dir, f"{participant_id}_eeg_preprocessed.fif")
        nn_intervals_file = os.path.join(preprocessed_data_dir, f"{participant_id}_nn_intervals.csv")
        phasic_eda_file = os.path.join(preprocessed_data_dir, f"{participant_id}_phasic_eda.csv")
        fnirs_file = os.path.join(preprocessed_data_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
        rpeak_times_file = os.path.join(preprocessed_data_dir, f"{participant_id}_rpeaks_times_sec.csv")

        raw_eeg = mne.io.read_raw_fif(eeg_file, preload=True) if os.path.exists(eeg_file) else None
        nn_intervals_df = pd.read_csv(nn_intervals_file) if os.path.exists(nn_intervals_file) else None
        phasic_eda_df = pd.read_csv(phasic_eda_file) if os.path.exists(phasic_eda_file) else None
        fnirs_haemo = mne.io.read_raw_fif(fnirs_file, preload=True) if os.path.exists(fnirs_file) else None
        
        # Load sampling rates
        loaded_sampling_rates = {}
        sampling_rates_file = os.path.join(preprocessed_data_dir, f"{participant_id}_sampling_rates.json")
        original_eda_sampling_rate = 1000 # Default fallback
        if os.path.exists(sampling_rates_file):
            try:
                with open(sampling_rates_file, 'r') as f_sr:
                    loaded_sampling_rates = json.load(f_sr)
                original_eda_sampling_rate = loaded_sampling_rates.get('eda_sampling_rate', original_eda_sampling_rate)
                p_logger.info(f"Loaded sampling rates from {sampling_rates_file}. EDA sampling rate: {original_eda_sampling_rate} Hz.")
            except Exception as e_sr:
                p_logger.warning(f"Could not load sampling rates file {sampling_rates_file}: {e_sr}. Using default EDA rate: {original_eda_sampling_rate} Hz.")
        
        # Integrate questionnaire data into analysis_metrics
        if questionnaire_data:
            p_logger.info("Integrating questionnaire data into analysis metrics.")
            for key, value in questionnaire_data.items():
                if key != 'participant_id': # participant_id is already set
                    analysis_metrics[key] = value
                    # Ensure numeric scores are indeed numeric for correlations
                    # if key in ['SAM_arousal', 'SAM_valence', 'BIS_score', 'BAS_drive', 'PANAS_Positive', 'PANAS_Negative']:
                    #     try: analysis_metrics[key] = float(value)
                    #     except (ValueError, TypeError): analysis_metrics[key] = np.nan

        p_logger.info(f"Status: PreprocessedDataLoaded")
        # --- fNIRS ROI Analysis (GLM based on proposal) ---
        # Used to identify functionally relevant EEG channels for PLV
        p_logger.info(f"\nfNIRS - Stage: ROI_Analysis")
        selected_eeg_channels_for_plv = ['Fp1', 'Fp2', 'F3', 'F4'] # Default if GLM fails or not used
        if fnirs_haemo:
            try:
                events_fnirs, event_id_fnirs = mne.events_from_annotations(fnirs_haemo, verbose=False)
                p_logger.info(f"fNIRS - Events found: {len(events_fnirs)}, Event IDs: {event_id_fnirs}")

                # TODO: Define ROIs based on channel names or 10-20 system if applicable to your fNIRS montage
                # rois = {'DLPFC_L': ['S1_D1', 'S1_D2'], 'DLPFC_R': ['S5_D3', 'S5_D4']} # Example

                # TODO: Create design matrix using mne_nirs.statistics.make_first_level_design_matrix
                # design_matrix = mne_nirs.statistics.make_first_level_design_matrix(
                #     fnirs_haemo.times, events_fnirs,
                #     hrf_model='spm', # or 'glover' or 'fir'
                #     drift_model='polynomial', drift_order=1, # or 'cosine'
                #     stim_dur=5.0, # Example: duration of your stimulus in seconds
                #     event_id=event_id_fnirs # Ensure event_id_fnirs keys match your annotation descriptions
                # )
                # p_logger.info(f"fNIRS - DesignMatrixCreated - Shape: {design_matrix.shape if 'design_matrix' in locals() else 'N/A'}")

                # TODO: Run GLM using mne_nirs.statistics.run_glm
                # glm_estimates = mne_nirs.statistics.run_glm(fnirs_haemo, design_matrix, noise_model='ar1') # or 'ols'
                # p_logger.info(f"fNIRS - GLMRunComplete")

                # TODO: Define and compute contrasts (e.g., 'Positive_Condition - Neutral_Condition')
                # contrast_matrix = np.eye(design_matrix.shape[1]) # Placeholder
                # basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])
                # contrast_val = glm_estimates.compute_contrast(basic_contrasts['Positive_Emotion'] - basic_contrasts['Neutral']) # Example
                # analysis_metrics['fnirs_hbo_positive_vs_neutral_dlpfc_mean'] = contrast_val.get_data(picks=roi_dlpfc_hbo_channels).mean()

                # TODO: Based on significant GLM results, update selected_eeg_channels_for_plv
                # This is complex and might involve finding EEG channels closest to significant fNIRS channels.
                # For now, we keep the default or a predefined list.
                p_logger.info(f"fNIRS - Using EEG channels for PLV: {selected_eeg_channels_for_plv} (Update if GLM guides selection)")

            except Exception as e_fnirs_glm:
                p_logger.error(f"fNIRS - GLMAnalysisError: {e_fnirs_glm}", exc_info=True)
            # analysis_metrics['fnirs_dlpfc_hbo_contrast_mean'] = contrast_dlpfc_hbo['theta'].mean()
        p_logger.info(f"fNIRS - Status: ROI_Analysis_Complete")

        # --- EEG Analysis (Power, Phase for PLV, FAI) ---
        p_logger.info(f"\nEEG - Stage: Analysis")
        if raw_eeg:
            p_logger.info(f"EEG - Action: SegmentEpochs")
            events_eeg, event_id_eeg = mne.events_from_annotations(raw_eeg) # Assuming events are annotated in raw_eeg
            epochs = mne.Epochs(raw_eeg, events_eeg, event_id=event_id_eeg, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True, verbose=False) # TODO: Adjust tmax & baseline
            # TODO: Calculate PSD for Alpha (8-13 Hz) and Beta (13-30 Hz) for F3, F4, Fp1, Fp2
            # (and potentially channels selected by fNIRS, ensure selected_eeg_channels_for_plv is defined)
            # Use the selected_eeg_channels_for_plv which might be updated by fNIRS GLM, or defaults.
            psd_picks = [ch for ch in selected_eeg_channels_for_plv if ch in raw_eeg.ch_names]
            p_logger.info(f"EEG - PSDCalculation - Channels: {psd_picks}")
            
            psd_params_alpha = dict(fmin=8, fmax=13, method='welch', picks=psd_picks)
            p_logger.info(f"EEG - Action: CalculatePSD - Band: Alpha")
            psds_alpha, freqs_alpha = epochs.compute_psd(**psd_params_alpha).get_data(return_freqs=True)
            
            psd_params_beta = dict(fmin=13, fmax=30, method='welch', picks=psd_picks)
            p_logger.info(f"EEG - Action: CalculatePSD - Band: Beta")
            psds_beta, freqs_beta = epochs.compute_psd(**psd_params_beta).get_data(return_freqs=True)
            
            # Store average power for key channels if they were in psd_picks
            for ch_name in ['F3', 'F4', 'Fp1', 'Fp2']:
                if ch_name in psd_picks:
                    ch_idx_in_psd_picks = psd_picks.index(ch_name)
                    analysis_metrics[f'alpha_power_{ch_name}_mean'] = np.mean(psds_alpha[:, ch_idx_in_psd_picks, :])
                    analysis_metrics[f'beta_power_{ch_name}_mean'] = np.mean(psds_beta[:, ch_idx_in_psd_picks, :])
                else:
                    analysis_metrics[f'alpha_power_{ch_name}_mean'] = np.nan
                    analysis_metrics[f'beta_power_{ch_name}_mean'] = np.nan

            # FAI Calculation
            p_logger.info(f"EEG - Action: CalculateFAI")
            # FAI for F4/F3
            if analysis_metrics.get('alpha_power_F4_mean', np.nan) is not np.nan and \
               analysis_metrics.get('alpha_power_F3_mean', np.nan) is not np.nan:
                power_right_alpha_f4 = analysis_metrics['alpha_power_F4_mean']
                power_left_alpha_f3 = analysis_metrics['alpha_power_F3_mean']
                analysis_metrics['fai_alpha_F4_F3'] = np.log(power_right_alpha_f4 + 1e-9) - np.log(power_left_alpha_f3 + 1e-9)
            else:
                analysis_metrics['fai_alpha_F4_F3'] = np.nan
                p_logger.warning("EEG - Metrics - Status: SkippedFAI_F4F3 (F4 or F3 power not available)")
            
            # FAI for Fp2/Fp1 (as per proposal)
            if analysis_metrics.get('alpha_power_Fp2_mean', np.nan) is not np.nan and \
               analysis_metrics.get('alpha_power_Fp1_mean', np.nan) is not np.nan:
                power_right_alpha_fp2 = analysis_metrics['alpha_power_Fp2_mean']
                power_left_alpha_fp1 = analysis_metrics['alpha_power_Fp1_mean']
                analysis_metrics['fai_alpha_Fp2_Fp1'] = np.log(power_right_alpha_fp2 + 1e-9) - np.log(power_left_alpha_fp1 + 1e-9)
            else:
                analysis_metrics['fai_alpha_Fp2_Fp1'] = np.nan
                p_logger.warning("EEG - Metrics - Status: SkippedFAI_Fp2Fp1 (Fp2 or Fp1 power not available)")
            
            p_logger.info(f"EEG - Action: ExtractPhase - Bands: Alpha, Beta")
            # TODO: Filter EEG in Alpha/Beta for phase extraction (e.g., on continuous data or epochs)
            eeg_alpha_filtered = raw_eeg.copy().filter(8, 13, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            eeg_beta_filtered = raw_eeg.copy().filter(13, 30, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            phase_eeg_alpha = np.angle(hilbert(eeg_alpha_filtered.get_data())) # Shape (n_channels, n_times)
            phase_eeg_beta = np.angle(hilbert(eeg_beta_filtered.get_data())) # Shape (n_channels, n_times)
        p_logger.info(f"EEG - Status: AnalysisComplete")

        # --- HRV Analysis (RMSSD, Continuous Signal for PLV) ---
        p_logger.info(f"\nHRV - Stage: Analysis")
        if nn_intervals_df is not None:
            rmssd_results = nk.hrv_time(nn_intervals_df['NN_ms'], sampling_rate=1000) # Assuming NN_ms is already clean
            analysis_metrics['rmssd_overall'] = rmssd_results['HRV_RMSSD'].iloc[0]
            # Create continuous HRV signal (interpolate NN intervals at 4Hz)
            rpeak_times_sec_df = pd.read_csv(rpeak_times_file) if os.path.exists(rpeak_times_file) else None
            if rpeak_times_sec_df is not None:
                rpeak_times_sec = rpeak_times_sec_df['R_Peak_Time_s'].values
                nn_values_ms = nn_intervals_df['NN_ms'].values
                if len(rpeak_times_sec) > 1 and len(nn_values_ms) > 0 and len(rpeak_times_sec) == len(nn_values_ms) + 1: # R-peaks are one more than NN-intervals
                    nn_interp_times = rpeak_times_sec[:-1] + np.diff(rpeak_times_sec)/2 # Midpoints of NN intervals
                    if len(nn_interp_times) > 1: # Need at least 2 points for arange and interp
                        interp_func_hrv = interp1d(nn_interp_times, nn_values_ms, kind='cubic', fill_value="extrapolate")
                        # TODO: Determine target_time_hrv based on EEG/experiment timing for alignment
                        target_time_hrv = np.arange(nn_interp_times[0], nn_interp_times[-1], 1/4.0) # 4 Hz
                        if len(target_time_hrv) > 0:
                            continuous_hrv_signal = interp_func_hrv(target_time_hrv)
                            phase_hrv = np.angle(hilbert(continuous_hrv_signal - np.mean(continuous_hrv_signal)))
                        else: p_logger.warning("HRV - Status: SkippedPhaseExtraction (TargetTimeHRV empty after arange)")
                    else: p_logger.warning("HRV - Status: SkippedPhaseExtraction (NotEnoughPointsForInterpolation)")
                else: p_logger.warning("HRV - Status: SkippedPhaseExtraction (DataLengthMismatch or InsufficientData)")
        p_logger.info(f"HRV - Status: AnalysisComplete")

        # --- EDA Analysis (Resample, Phase for PLV) ---
        p_logger.info(f"\nEDA - Stage: AnalysisForPLV")
        if phasic_eda_df is not None and not phasic_eda_df.empty:
            # original_eda_sampling_rate is now loaded from the JSON file or defaults to 1000
            if original_eda_sampling_rate is None: # Should not happen if saved correctly or default is used
                p_logger.error("EDA - Error: OriginalSamplingRateUnknown. Cannot resample for PLV.")
            phasic_signal = phasic_eda_df['EDA_Phasic'].values
            original_eda_time = np.arange(len(phasic_signal)) / original_eda_sampling_rate
            if len(original_eda_time) > 1: # Need at least 2 points for arange and interp
                interp_func_eda = interp1d(original_eda_time, phasic_signal, kind='linear', fill_value="extrapolate")
                # TODO: Determine target_time_eda based on EEG/experiment timing for alignment
                target_time_eda = np.arange(original_eda_time[0], original_eda_time[-1], 1/4.0) # Resample to 4 Hz
                if len(target_time_eda) > 0:
                    resampled_phasic_eda = interp_func_eda(target_time_eda)
                    phase_eda = np.angle(hilbert(resampled_phasic_eda - np.mean(resampled_phasic_eda)))
                else: p_logger.warning("EDA - Status: SkippedPhaseExtraction (TargetTimeEDA empty after arange)")
            else: p_logger.warning("EDA - Status: SkippedPhaseExtraction (SignalTooShortForResampling)")
        p_logger.info(f"EDA - Status: AnalysisForPLV_Complete")

        # --- PLV Calculation ---
        p_logger.info(f"\nConnectivity - Stage: PLV_Calculation")
        
        conditions = ['Positive', 'Negative', 'Neutral'] # Example conditions from your proposal
        # Ensure event_id_eeg keys match your annotation descriptions for these conditions
        # e.g., event_id_eeg = {'Positive': 1, 'Negative': 2, 'Neutral': 3}

        if raw_eeg and 'phase_eeg_alpha' in locals() and 'phase_eeg_beta' in locals() and \
           ('phase_hrv' in locals() or 'phase_eda' in locals()):
            
            eeg_sfreq = raw_eeg.info['sfreq']
            # Time vectors for continuous autonomic signals (assuming 4Hz resampling)
            # These need to be aligned with EEG time.
            # target_time_hrv and target_time_eda from earlier interpolation steps are key.

            for condition in conditions:
                p_logger.info(f"Connectivity - ProcessingCondition: {condition}")
                # TODO: Get event onsets and durations for the current 'condition' from 'events_eeg'
                # condition_event_indices = events_eeg[events_eeg[:, 2] == event_id_eeg[condition], 0] # Sample indices of event onsets
                # event_duration_samples = int(stimulus_duration_seconds * eeg_sfreq) # Example

                # for event_onset_sample in condition_event_indices:
                #     eeg_epoch_start_sample = event_onset_sample # Or relative to onset, e.g., event_onset_sample + int(tmin_plv * eeg_sfreq)
                #     eeg_epoch_end_sample = eeg_epoch_start_sample + event_duration_samples
                    
                #     # TODO: Extract corresponding segments from phase_hrv and phase_eda
                #     # This requires converting EEG sample times to autonomic signal sample times.
                #     # eeg_time_onset = eeg_epoch_start_sample / eeg_sfreq
                #     # eeg_time_offset = eeg_epoch_end_sample / eeg_sfreq
                #     # hrv_start_idx = np.argmin(np.abs(target_time_hrv - eeg_time_onset))
                #     # hrv_end_idx = np.argmin(np.abs(target_time_hrv - eeg_time_offset))
                #     # eda_start_idx = ...
                #     # eda_end_idx = ...

                #     # For each EEG channel in selected_eeg_channels_for_plv:
                #     for eeg_ch_idx, eeg_ch_name in enumerate(eeg_alpha_filtered.ch_names): # Assuming eeg_alpha_filtered has the selected channels
                #         eeg_phase_alpha_epoch = phase_eeg_alpha[eeg_ch_idx, eeg_epoch_start_sample:eeg_epoch_end_sample]
                #         eeg_phase_beta_epoch = phase_eeg_beta[eeg_ch_idx, eeg_epoch_start_sample:eeg_epoch_end_sample]
                        
                #         if 'phase_hrv' in locals():
                #             # phase_hrv_epoch = phase_hrv[hrv_start_idx:hrv_end_idx]
                #             # analysis_metrics[f'plv_alpha_{eeg_ch_name}_hrv_{condition}'] = calculate_plv(eeg_phase_alpha_epoch, phase_hrv_epoch, p_logger)
                #             # analysis_metrics[f'plv_beta_{eeg_ch_name}_hrv_{condition}'] = calculate_plv(eeg_phase_beta_epoch, phase_hrv_epoch, p_logger)
                #             pass # Placeholder
                #         if 'phase_eda' in locals():
                #             # phase_eda_epoch = phase_eda[eda_start_idx:eda_end_idx]
                #             # analysis_metrics[f'plv_alpha_{eeg_ch_name}_eda_{condition}'] = calculate_plv(eeg_phase_alpha_epoch, phase_eda_epoch, p_logger)
                #             # analysis_metrics[f'plv_beta_{eeg_ch_name}_eda_{condition}'] = calculate_plv(eeg_phase_beta_epoch, phase_eda_epoch, p_logger)
                #             pass # Placeholder
                # TODO: Average PLV values across epochs for each condition/channel/band/autonomic_signal if multiple epochs per condition
                pass # Placeholder for the loop over events within a condition
        else:
            p_logger.info("Connectivity - Status: SkippedPLV (Missing necessary phase signals or EEG data)")
        p_logger.info(f"Connectivity - Status: PLV_Calculation_Complete")

        # --- FAI Calculation (WP4) ---
        p_logger.info(f"\nEEG - Stage: FAI_Calculation")
        # FAI calculations (F4/F3 and Fp2/Fp1) are now done in the EEG Analysis section.
        # This section can be removed or used for other FAI variants if needed.
        p_logger.info(f"EEG - Status: FAI_Calculation_Complete")

        # Save all calculated metrics
        p_logger.info(f"\nStage: SaveResults")
        p_logger.info(f"Action: SaveAnalysisMetrics")
        metrics_df = pd.DataFrame([analysis_metrics])
        metrics_output_file = os.path.join(analysis_results_dir, f"{participant_id}_analysis_metrics.csv")
        metrics_df.to_csv(metrics_output_file, index=False)
        p_logger.info(f"Status: AnalysisMetricsSaved - File: {metrics_output_file}")

        # --- Generate Participant-Specific Plots ---
        p_logger.info(f"\nStage: GeneratePlots")
        generate_participant_plots(participant_id, analysis_results_dir, p_logger)

        # --- Statistical Testing (WP1, WP2, WP3, WP4) ---
        p_logger.info(f"\nStage: StatisticalTesting")
        # These are placeholders for where group-level statistical tests would be performed
        # *after* all participants are processed and metrics are aggregated.
        # The per-participant metrics saved above are the input for the group-level stats.

        # TODO WP1: Repeated measures ANOVA on PLV values (Emotion x Autonomic Branch x Frequency Band).
        # Example: pingouin.rm_anova(data=all_participants_aggregated_df, dv='PLV_Value', 
        #                            within=['Emotion', 'AutonomicBranch', 'FrequencyBand'], subject='participant_id')

        # TODO WP2: Correlate SAM arousal ratings with PLV values.
        # Example: pingouin.corr(all_participants_aggregated_df['SAM_arousal'], all_participants_aggregated_df['plv_alpha_Fp1_hrv_Positive'])
        # Ensure 'SAM_arousal' (and other questionnaire scores like BIS, PANAS) are in analysis_metrics.
        # This is now handled by merging questionnaire_data into analysis_metrics earlier.

        # TODO WP3: Correlate resting-state RMSSD with PLV values during negative emotional stimuli.
        # Example: pingouin.corr(all_participants_aggregated_df['rmssd_overall'], all_participants_aggregated_df['plv_alpha_Fp1_hrv_Negative'])

        # TODO WP4: Correlate FAI (Alpha band, Fp2/Fp1 and F4/F3) with branch-specific PLV.
        # Example for F4/F3 FAI and EEG-HRV PLV:
        # pingouin.corr(all_participants_aggregated_df['fai_alpha_F4_F3'], all_participants_aggregated_df['plv_alpha_F4_hrv_Positive']) # Assuming F4 was used for PLV
        # Example for F4/F3 FAI and EEG-EDA PLV:
        # pingouin.corr(all_participants_aggregated_df['fai_alpha_F4_F3'], all_participants_aggregated_df['plv_alpha_F4_eda_Positive'])
        # Repeat for Fp2/Fp1 FAI and relevant PLV metrics.

        p_logger.info(f"Status: StatisticalTestingPlaceholder (Group-level stats after all participants)")

        # TODO: Generate analysis reports or visualizations if needed (e.g., using matplotlib, seaborn)

        p_logger.info(f"==== END Analysis - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"Analysis - Error: {e}", exc_info=True)
        p_logger.info(f"==== END Analysis - Status: FAILURE ====")
        return False

# --- Main Orchestration ---
def process_participant(participant_id):
    """Processes a single participant: setup log, preprocess, analyze, and commit results."""
    participant_raw_data_path = os.path.join(PARTICIPANT_DATA_BASE_DIR, participant_id)
    # Setup participant-specific logger to write into their rawData subfolder
    p_logger = setup_participant_logger(participant_id, participant_raw_data_path)

    success = False
    questionnaire_data = None
    try:
        p_logger.info(f"==== START Participant Processing: {participant_id} ====") # Explicitly state ID here for clarity
        participant_preprocessed_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "preprocessed")
        participant_analysis_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "analysis")

        # Step 0: Parse Questionnaire Data (if exists)
        p_logger.info(f"\nStage: QuestionnaireParsing")
        questionnaire_data = parse_questionnaire_data(participant_id, participant_raw_data_path, p_logger)
        
        # Step 1: Preprocessing
        p_logger.info(f"\nStage: Preprocessing")
        if not os.path.isdir(participant_raw_data_path):
            p_logger.error(f"Error: RawDataPathNotFound - Path: {participant_raw_data_path}")
            return False, questionnaire_data # Return questionnaire data even if other steps fail
        if not run_preprocessing(participant_id, participant_raw_data_path, participant_preprocessed_dir, p_logger):
            p_logger.error(f"Error: PreprocessingFailed")
            return False, questionnaire_data

        # Step 2: Analysis
        p_logger.info(f"\nStage: Analysis")
        if not run_analysis(participant_id, participant_preprocessed_dir, participant_analysis_dir, questionnaire_data, p_logger):
            p_logger.error(f"Error: AnalysisFailed")
            return False, questionnaire_data

        # Step 3: Commit results to Git
        p_logger.info(f"\nStage: GitCommit")
        commit_message = f"Pipeline: Processed data for participant {participant_id}"
        participant_log_file_actual_path = p_logger.handlers[0].baseFilename
        
        # Add the entire results directory for the participant (preprocessed + analysis)
        # and the participant's log file.
        files_to_add_to_git = [
            os.path.relpath(os.path.join(RESULTS_BASE_DIR, participant_id), BASE_REPO_PATH).replace("\\", "/"),
            os.path.relpath(participant_log_file_actual_path, BASE_REPO_PATH).replace("\\", "/")
        ]
        
        if not git_add_commit_push(BASE_REPO_PATH, commit_message, files_to_add_to_git, p_logger):
            p_logger.error(f"Error: GitCommitPushFailed")
            return False, questionnaire_data

        p_logger.info(f"Status: GitCommitPushSuccessful")
        success = True
    except Exception as e:
        p_logger.error(f"Error: UnhandledExceptionInProcessing - Details: {e}", exc_info=True)
        success = False
    finally:
        # --- Processing Summary ---
        p_logger.info(f"\n==== END Participant Processing: {participant_id} ====")
        if success:
            p_logger.info(f"Status: SUCCESS")
        else:
            p_logger.error(f"Status: FAILURE")
            # Read the log file to find errors/warnings for summary
            error_summary = []
            try:
                with open(p_logger.handlers[0].baseFilename, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for lines with ERROR or WARNING level
                        if ' - ERROR - ' in line or ' - WARNING - ' in line:
                            error_summary.append(line.strip())
            except Exception as log_read_error:
                p_logger.error(f"Failed to read log file for error summary: {log_read_error}", exc_info=True)
                error_summary = ["Could not read log file for detailed errors."]

            if error_summary: # Only print summary if issues were found
                p_logger.error("Summary of issues found:")
                for issue in error_summary:
                    p_logger.error(f"- {issue}")
            else:
                 p_logger.info("No specific ERROR/WARNING messages found in log despite overall failure.")

        close_participant_logger(participant_id)
    return success, questionnaire_data

def run_pipeline():
    """Main pipeline execution function."""
    main_logger.info(f"===== EmotiView Analysis Pipeline Started (Processing data from: {PARTICIPANT_DATA_BASE_DIR}) =====")

    main_logger.info("Attempting to pull latest changes from the repository...")
    if not git_pull(BASE_REPO_PATH, main_logger):
        main_logger.warning("Failed to pull latest changes. Continuing with local version, but repo might be outdated.")

    processed_participant_ids_in_session = set() # Keep track of processed participants in this run
    all_participant_metrics_aggregated = [] # To store metrics from all participants for group plots

    # Initial scan of RESULTS_BASE_DIR to identify already processed participants
    # This helps avoid reprocessing if the script restarts.
    main_logger.info(f"Scanning {RESULTS_BASE_DIR} for previously processed participants...")
    try:
        for item_name in os.listdir(RESULTS_BASE_DIR):
            item_path = os.path.join(RESULTS_BASE_DIR, item_name)
            if os.path.isdir(item_path):
                expected_metrics_file = os.path.join(item_path, "analysis", f"{item_name}_analysis_metrics.csv")
                if os.path.exists(expected_metrics_file):
                    main_logger.info(f"Participant {item_name} appears to be processed (results found). Adding to skip list for this session.")
                    processed_participant_ids_in_session.add(item_name)
                    # Optionally load these metrics for group plotting if script is re-run
                    # try:
                    #     metrics_df = pd.read_csv(expected_metrics_file)
                    #     all_participant_metrics_aggregated.append(metrics_df.iloc[0].to_dict()) # Assuming one row per file
                    # except Exception as e_load:
                    #     main_logger.warning(f"Could not load metrics for already processed participant {item_name}: {e_load}")

    except Exception as e:
        main_logger.error(f"Error during initial scan of results directory: {e}", exc_info=True)

    all_questionnaire_data_collected = []
    successful_participants = 0
    failed_participants = 0

    if not os.path.isdir(PARTICIPANT_DATA_BASE_DIR):
        main_logger.error(f"CRITICAL: Participant data base directory not found: {PARTICIPANT_DATA_BASE_DIR}. Please check configuration. Exiting.")
        return

    try:
        # Get participant IDs from subdirectories in PARTICIPANT_DATA_BASE_DIR
        participant_ids = [
            d for d in os.listdir(PARTICIPANT_DATA_BASE_DIR)
            if os.path.isdir(os.path.join(PARTICIPANT_DATA_BASE_DIR, d))
        ]
    except FileNotFoundError:
        main_logger.error(f"Error listing participant data directory: {PARTICIPANT_DATA_BASE_DIR}. Exiting.")
        return

    if not participant_ids:
        main_logger.info(f"No participant subfolders found in {PARTICIPANT_DATA_BASE_DIR}.")
    else:
        main_logger.info(f"Found {len(participant_ids)} participant(s) to process: {', '.join(sorted(participant_ids))}")

    for participant_id in sorted(participant_ids):
        if participant_id in processed_participant_ids_in_session:
            main_logger.info(f"Participant {participant_id} already processed in this session or results found. Skipping.")
            # If you want to include their metrics for group plots even when skipping reprocessing:
            metrics_file_path = os.path.join(RESULTS_BASE_DIR, participant_id, "analysis", f"{participant_id}_analysis_metrics.csv")
            if os.path.exists(metrics_file_path):
                try:
                    metrics_df = pd.read_csv(metrics_file_path)
                    if not metrics_df.empty:
                         # Check if this participant's metrics are already loaded
                        if not any(d.get('participant_id') == participant_id for d in all_participant_metrics_aggregated):
                            all_participant_metrics_aggregated.append(metrics_df.iloc[0].to_dict())
                except Exception as e_load_skip:
                    main_logger.warning(f"Could not load metrics for skipped participant {participant_id}: {e_load_skip}")
            questionnaire_file_path = os.path.join(PARTICIPANT_DATA_BASE_DIR, participant_id, QUESTIONNAIRE_TXT_FILENAME)
            if os.path.exists(questionnaire_file_path):
                 if not any(d.get('participant_id') == participant_id for d in all_questionnaire_data_collected):
                    # Dummy parse to get participant_id if needed for consistency, or load from aggregated if available
                    # This part might need more robust handling if questionnaire data is also needed for group plots from skipped participants
                    pass
            continue

        main_logger.info(f"--- Starting pipeline for participant: {participant_id} ---")
        overall_success, q_data = process_participant(participant_id)
        if overall_success:
            successful_participants += 1
            # Load this participant's metrics for group plotting
            metrics_file_path = os.path.join(RESULTS_BASE_DIR, participant_id, "analysis", f"{participant_id}_analysis_metrics.csv")
            if os.path.exists(metrics_file_path):
                try:
                    metrics_df = pd.read_csv(metrics_file_path)
                    if not metrics_df.empty:
                        all_participant_metrics_aggregated.append(metrics_df.iloc[0].to_dict())
                except Exception as e_load_processed:
                    main_logger.warning(f"Could not load metrics for successfully processed participant {participant_id}: {e_load_processed}")
        else:
            failed_participants += 1
        
        if q_data:
            all_questionnaire_data_collected.append(q_data)
        main_logger.info(f"--- Finished pipeline for participant: {participant_id} ---")

    # After processing all participants, generate final group-level plots
    # Combine questionnaire data with physiological metrics if needed for plotting
    # For simplicity, passing all_participant_metrics_aggregated which should contain physiological metrics.
    # You might want to merge this with all_questionnaire_data_collected based on participant_id before plotting.
    if all_participant_metrics_aggregated:
        group_metrics_df = pd.DataFrame(all_participant_metrics_aggregated)
        # If you want to merge with questionnaire data:
        if all_questionnaire_data_collected:
            q_df = pd.DataFrame(all_questionnaire_data_collected)
            if not q_df.empty and 'participant_id' in q_df.columns and 'participant_id' in group_metrics_df.columns:
                group_metrics_df = pd.merge(group_metrics_df, q_df, on="participant_id", how="left") # or "outer"
        
        generate_group_plots(group_metrics_df, RESULTS_BASE_DIR, main_logger)
    else:
        main_logger.info("No participant metrics available to generate group plots.")


    # After processing all participants, aggregate and save questionnaire data
    if all_questionnaire_data_collected:
        main_logger.info("Aggregating questionnaire data from all processed participants...")
        questionnaire_df = pd.DataFrame(all_questionnaire_data_collected) # This was already done above, ensure it's the final one
        excel_output_path = os.path.join(RESULTS_BASE_DIR, AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME)
        try:
            questionnaire_df.to_excel(excel_output_path, index=False)
            main_logger.info(f"Aggregated questionnaire data for this session saved to: {excel_output_path}")
            q_commit_message = "Pipeline: Update aggregated questionnaire data"
            q_files_to_add = [os.path.relpath(excel_output_path, BASE_REPO_PATH).replace("\\", "/")]
            git_add_commit_push(BASE_REPO_PATH, q_commit_message, q_files_to_add, main_logger)
        except Exception as e:
            main_logger.error(f"Failed to save or commit aggregated questionnaire Excel at exit: {e}", exc_info=True)
    else:
        main_logger.info("No new questionnaire data collected in this session to aggregate.")

    main_logger.info("===== EmotiView Analysis Pipeline Finished =====")
    main_logger.info(f"Summary: {successful_participants} participant(s) processed successfully.")
    if failed_participants > 0:
        main_logger.warning(f"Summary: {failed_participants} participant(s) failed processing.")
    main_logger.info(f"Main log file: {main_log_file}")
    main_logger.info(f"Participant logs are in their respective rawData subfolders. Results are in: {RESULTS_BASE_DIR}.")

if __name__ == "__main__":
    try:
        git_version_process = subprocess.run(['git', '--version'], capture_output=True, text=True, check=True)
        main_logger.info(f"Git installation found: {git_version_process.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        main_logger.critical("CRITICAL: Git command not found. Ensure Git is installed and in PATH. Exiting.")
        exit(1)

    if not os.path.isdir(os.path.join(BASE_REPO_PATH, ".git")):
        main_logger.critical(f"CRITICAL: BASE_REPO_PATH '{BASE_REPO_PATH}' is not a Git repository. Exiting.")
        exit(1)

    run_pipeline()