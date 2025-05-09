import os
import subprocess
import logging
import shutil
import datetime
import pandas as pd # For handling results
import numpy as np
import mne
import neurokit2 as nk
from scipy.signal import hilbert, welch # For Hilbert transform to get phase, and PSD
from scipy.interpolate import interp1d # For resampling/interpolating signals
import matplotlib.pyplot as plt # For plotting
import mne_nirs # For fNIRS specific functions
from mne_icalabel import label_components # For automatic ICA component labeling

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
        # TODO: Load necessary preprocessed/analysis data for plotting
        # Example: Load preprocessed EEG to plot PSD
        # eeg_preprocessed_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "preprocessed")
        # eeg_file = os.path.join(eeg_preprocessed_dir, f"{participant_id}_eeg_preprocessed.fif")
        # if os.path.exists(eeg_file):
        #     raw_eeg = mne.io.read_raw_fif(eeg_file, preload=True)
        #     p_logger.info(f"EV_{participant_id} - Plotting - Action: ComputePSDEEG")
        #     fig = raw_eeg.compute_psd(picks='eeg', fmax=40).plot(show=False, average=True) # Plot average PSD
        #     psd_plot_path = os.path.join(analysis_results_dir, f"{participant_id}_eeg_psd.png")
        #     fig.savefig(psd_plot_path)
        #     plt.close(fig) # Close the figure to free memory
        #     p_logger.info(f"EV_{participant_id} - Plotting - Action: SavePlot - File: {psd_plot_path}")

        # TODO: Add other participant-specific plots (e.g., autonomic signals time series)

        p_logger.info(f"EV_{participant_id} - Plotting - Status: ParticipantPlotsGenerated")
    except Exception as e:
        p_logger.error(f"EV_{participant_id} - Plotting - Error: {e}", exc_info=True)

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
        # TODO: Implement group-level plotting logic here
        # This DataFrame contains metrics from all participants processed so far.
        # Example: Plot average FAI across participants
        # if 'fai_f4_f3_alpha' in all_participants_metrics_df.columns and not all_participants_metrics_df['fai_f4_f3_alpha'].isnull().all():
        #     fig, ax = plt.subplots()
        #     # Ensure participant_id is suitable for x-axis labels
        #     plot_df = all_participants_metrics_df.dropna(subset=['fai_f4_f3_alpha'])
        #     if not plot_df.empty:
        #         plot_df.plot(kind='bar', x='participant_id', y='fai_f4_f3_alpha', ax=ax, legend=False)
        #         ax.set_ylabel('Average FAI (F4-F3 Alpha)')
        #         ax.set_title('FAI per Participant')
        #         plt.xticks(rotation=45, ha="right")
        #         plt.tight_layout()
        #         group_fai_plot_path = os.path.join(results_base_dir, "group_fai_plot.png")
        #         fig.savefig(group_fai_plot_path)
        #         plt.close(fig)
        #         main_logger_to_use.info(f"Plotting - Action: SavePlot - File: {group_fai_plot_path}")
        #     else:
        #         main_logger_to_use.info("Plotting - FAI - Status: Skipped (NoValidFAIData)")
        # else:
        #     main_logger_to_use.info("Plotting - FAI - Status: Skipped (NoFAIDataColumn or AllNaN)")


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
        p_logger.info(f"EV_{participant_id} - Questionnaire - Status: FileNotFound - File: {questionnaire_file_path}")
        return None

    try:
        p_logger.info(f"EV_{participant_id} - Questionnaire - Action: ParseFile - File: {questionnaire_file_path}")
        with open(questionnaire_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or ':' not in line:
                    if line: # Non-empty line without colon
                        p_logger.warning(f"EV_{participant_id} - Questionnaire - Warning: SkippingMalformedLine - LineNum: {line_num} - Content: '{line}'")
                    continue
                key, value = line.split(':', 1)
                parsed_data[key.strip()] = value.strip()
        p_logger.info(f"EV_{participant_id} - Questionnaire - Status: ParsedSuccessfully")
        return parsed_data
    except Exception as e:
        p_logger.error(f"EV_{participant_id} - Questionnaire - Error: ParsingFailed - File: {questionnaire_file_path} - Details: {e}", exc_info=True)
        return None

# --- Placeholder Processing Functions ---

def run_preprocessing(participant_id, raw_data_path, processed_data_output_dir, p_logger):
    """
    Participant data preprocessing based on EV_proposal.tex.
    Saves preprocessed files into `processed_data_output_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"==== Preprocessing ====") 
    p_logger.debug(f"EV_{participant_id} - Config - RawDataPath: {raw_data_path}") # EV_xxx prefix kept for debug clarity
    p_logger.debug(f"EV_{participant_id} - Config - ProcessedDataOutputPath: {processed_data_output_dir}") # EV_xxx prefix kept for debug clarity
    
    os.makedirs(processed_data_output_dir, exist_ok=True)

    try:
        if not os.path.exists(raw_data_path):
            p_logger.error(f"EV_{participant_id} - Error: RawDataPathNotFound - Path: {raw_data_path}")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error(f"EV_{participant_id} - Error: LibrariesNotLoaded - Details: NumPy/MNE missing")
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
            p_logger.info(f"EV_{participant_id} - CombinedStream - Action: LoadFile - File: {combined_stream_file}")
            raw_combined = mne.io.read_raw_brainvision(combined_stream_file, preload=True) # Or read_raw_xdf if it's an XDF
            combined_sampling_rate = raw_combined.info['sfreq']
            p_logger.info(f"EV_{participant_id} - CombinedStream - Status: Loaded - SFreq: {combined_sampling_rate} Hz")
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
                p_logger.info(f"EV_{participant_id} - EDA - Action: ExtractFromCombined - ChannelName: {eda_ch_name} (Index: {eda_channel_index})")
            else:
                p_logger.warning(f"EV_{participant_id} - EDA - Status: ChannelIndexOutOfRange - ExpectedIndex: {eda_channel_index}")

            # Extract ECG
            if len(raw_combined.ch_names) > ecg_channel_index:
                ecg_ch_name = raw_combined.ch_names[ecg_channel_index]
                ecg_signal_raw = raw_combined.get_data(picks=[ecg_ch_name])[0]
                ecg_sampling_rate = combined_sampling_rate
                p_logger.info(f"EV_{participant_id} - ECG - Action: ExtractFromCombined - ChannelName: {ecg_ch_name} (Index: {ecg_channel_index})")
            else:
                p_logger.warning(f"EV_{participant_id} - ECG - Status: ChannelIndexOutOfRange - ExpectedIndex: {ecg_channel_index}")

            # Prepare EEG data (all channels from eeg_start_channel_index onwards)
            if len(raw_combined.ch_names) > eeg_start_channel_index:
                eeg_ch_names = raw_combined.ch_names[eeg_start_channel_index:]
                if eeg_ch_names: # Ensure there are EEG channels to pick
                    raw_eeg = raw_combined.copy().pick_channels(eeg_ch_names)
                    # Set channel types for EEG if not already correct (MNE might infer them)
                    # raw_eeg.set_channel_types({ch_name: 'eeg' for ch_name in eeg_ch_names})
                    eeg_sampling_rate = combined_sampling_rate
                    p_logger.info(f"EV_{participant_id} - EEG - Action: ExtractFromCombined - Channels: {len(eeg_ch_names)} (StartingIndex: {eeg_start_channel_index})")
                else:
                    p_logger.warning(f"EV_{participant_id} - EEG - Status: NoEEGChannelsFoundAfterIndex - StartIndex: {eeg_start_channel_index}")
            else:
                p_logger.warning(f"EV_{participant_id} - EEG - Status: ChannelIndexOutOfRangeForEEGStart - ExpectedStartIndex: {eeg_start_channel_index}")
        else:
            p_logger.warning(f"EV_{participant_id} - CombinedStream - Status: FileNotFound - File: {combined_stream_file}")

        # Example: Load fNIRS from a .nirs file (or similar MNE-NIRS compatible format)
        fnirs_data_file = os.path.join(raw_data_path, f"{participant_id}_fnirs.nirs") # Adjust filename as needed
        if os.path.exists(fnirs_data_file):
            p_logger.info(f"EV_{participant_id} - fNIRS - Action: LoadFile - File: {fnirs_data_file}")
            # Use the appropriate MNE-NIRS reader, e.g., read_raw_nirx, read_raw_snirf, etc.
            # This example assumes a generic reader or that it's already optical density.
            # If it's raw intensity, you'll need mne_nirs.optical_density() later.
            fnirs_raw_od = mne.io.read_raw_nirx(fnirs_data_file, preload=True) # Replace with actual reader
            p_logger.info(f"EV_{participant_id} - fNIRS - Status: Loaded - SFreq: {fnirs_raw_od.info['sfreq']} Hz")
            # If the loaded fNIRS data is not yet optical density, convert it:
            # fnirs_raw_od = mne_nirs.optical_density(fnirs_raw_od_or_intensity)
        else:
            p_logger.warning(f"EV_{participant_id} - fNIRS - Status: FileNotFound - File: {fnirs_data_file}")

        # --- EEG Preprocessing (MNE-Python) ---
        if raw_eeg is not None:
            p_logger.info(f"\nEV_{participant_id} - EEG - Stage: Preprocessing") # Added newline and EV_xxx
            p_logger.info(f"") # Empty line for separation

            p_logger.info(f"EV_{participant_id} - EEG - Action: LoadDataToMemory") # If not already preloaded
            raw_eeg.load_data() # Explicitly load data for the EEG stream
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - EEG - Action: Filter - Params: 0.5-40Hz")
            raw_eeg.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - EEG - Action: SetReference - Type: Average")
            raw_eeg.set_eeg_reference('average', projection=True)
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - EEG - Action: FitICA - Components: 15")
            ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
            ica.fit(raw_eeg)
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - EEG - Action: AutoLabelICArtifacts")
            if mne_icalabel_available:
                try:
                    label_components(raw_eeg, ica, method='iclabel')
                    # Exclude components labeled as artifacts by ICLabel
                    # Common artifact labels: 'eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise'
                    # 'brain' and 'other' are typically kept.
                    ica.exclude = [idx for idx, label in enumerate(ica.labels_) 
                                   if label not in ['brain', 'other']]
                    p_logger.info(f"EV_{participant_id} - EEG - ICLabel - ExcludedIndices: {ica.exclude} - Labels: {[ica.labels_[i] for i in ica.exclude]}")
                except Exception as e_icalabel:
                    p_logger.error(f"EV_{participant_id} - EEG - ICLabel - Error: {e_icalabel}. Manual ICA component selection needed.", exc_info=True)
                    p_logger.warning(f"EV_{participant_id} - EEG - Action: ApplyICA - Excluded: None (ICLabel error)") # Log that no components were auto-excluded
            else:
                p_logger.warning(f"EV_{participant_id} - EEG - ICLabel - Status: PackageNotAvailable. Manual ICA component selection needed.")
                # TODO: Implement fallback or manual step here if mne_icalabel is not used.
                # For now, ica.exclude will remain empty or as manually set if you add that logic.
            ica.apply(raw_eeg, verbose=False)
            p_logger.info(f"EV_{participant_id} - EEG - Action: ApplyICA - FinalExcluded: {ica.exclude if hasattr(ica, 'exclude') and ica.exclude else 'None'}")
            p_logger.info(f"")

            preprocessed_eeg_file = os.path.join(processed_data_output_dir, f"{participant_id}_eeg_preprocessed.fif")
            raw_eeg.save(preprocessed_eeg_file, overwrite=True)
            p_logger.info(f"EV_{participant_id} - EEG - Action: SavePreprocessed - File: {preprocessed_eeg_file}")
        else:
            p_logger.info(f"EV_{participant_id} - EEG - Status: Skipped (NoData)")
        p_logger.info(f"") # Empty line after section

        # --- ECG Preprocessing (NeuroKit2) ---
        if ecg_signal_raw is not None and ecg_sampling_rate is not None:
            p_logger.info(f"\nEV_{participant_id} - ECG - Stage: Preprocessing")
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - ECG - Action: CleanSignal")
            ecg_cleaned = nk.ecg_clean(ecg_signal_raw, sampling_rate=sampling_rate_ecg)
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - ECG - Action: DetectRPeaks")
            peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate_ecg, correct_artifacts=True)
            rpeaks = peaks_info[0]['ECG_R_Peaks']
            p_logger.info(f"EV_{participant_id} - ECG - Status: RPeaksDetected - Count: {len(rpeaks)}")
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - ECG - Action: CalculateNNIntervals")
            nn_intervals = np.diff(rpeaks) / ecg_sampling_rate * 1000 # in ms
            nn_intervals_file = os.path.join(processed_data_output_dir, f"{participant_id}_nn_intervals.csv")
            pd.DataFrame(nn_intervals, columns=['NN_ms']).to_csv(nn_intervals_file, index=False)
            p_logger.info(f"EV_{participant_id} - ECG - Action: SaveNNIntervals - File: {nn_intervals_file}")
            p_logger.info(f"")

            rpeaks_times_file = os.path.join(processed_data_output_dir, f"{participant_id}_rpeaks_times_sec.csv")
            pd.DataFrame(rpeaks / ecg_sampling_rate, columns=['R_Peak_Time_s']).to_csv(rpeaks_times_file, index=False)
            p_logger.info(f"EV_{participant_id} - ECG - Action: SaveRPeakTimes - File: {rpeaks_times_file}")
        else:
            p_logger.info(f"EV_{participant_id} - ECG - Status: Skipped (NoData)")
        p_logger.info(f"")

        # --- EDA Preprocessing (NeuroKit2) ---
        if eda_signal_raw is not None and eda_sampling_rate is not None:
            p_logger.info(f"\nEV_{participant_id} - EDA - Stage: Preprocessing")
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - EDA - Action: ProcessSignal (Decomposition)")
            eda_signals, info = nk.eda_process(eda_signal_raw, sampling_rate=sampling_rate_eda)
            phasic_eda = eda_signals['EDA_Phasic']
            tonic_eda = eda_signals['EDA_Tonic']
            p_logger.info(f"EV_{participant_id} - EDA - Status: Decomposed")
            p_logger.info(f"")

            phasic_eda_file = os.path.join(processed_data_output_dir, f"{participant_id}_phasic_eda.csv")
            if isinstance(phasic_eda, (pd.Series, pd.DataFrame)):
                phasic_eda.to_csv(phasic_eda_file, index=False, header=True)
            else: # If it's a numpy array
                pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_file, index=False, header=True)
            p_logger.info(f"EV_{participant_id} - EDA - Action: SavePhasicEDA - File: {phasic_eda_file}")
        else:
            p_logger.info(f"EV_{participant_id} - EDA - Status: Skipped (NoData)")
        p_logger.info(f"")

        # --- fNIRS Preprocessing (MNE-NIRS) ---
        if fnirs_raw_od is not None: # fnirs_raw_od is the MNE Raw object for optical density from XDF
            p_logger.info(f"\nEV_{participant_id} - fNIRS - Stage: Preprocessing")
            p_logger.info(f"")

            p_logger.info(f"EV_{participant_id} - fNIRS - Action: LoadDataToMemory") # If not already preloaded
            fnirs_raw_od.load_data() # Explicitly load data for the fNIRS OD stream
            # Ensure it's treated as optical density if it's not already marked as such by read_raw_xdf
            # This might involve checking channel types or manually setting them if necessary.
            # For now, assume fnirs_raw_od is ready for Beer-Lambert or is raw voltages needing optical_density()
            raw_od_for_beer_lambert = fnirs_raw_od # Default assumption
            
            # If your XDF fNIRS stream provides raw voltages, you might need mne_nirs.optical_density() first
            # Example: if 'fnirs_raw' in fnirs_raw_od.info.get('ch_types', []): # Hypothetical check
            #    raw_od_for_beer_lambert = mne_nirs.optical_density(fnirs_raw_od)
            #    p_logger.info(f"EV_{participant_id} - Converted fNIRS raw voltages to optical density.")
            raw_od_for_beer_lambert.info['bads'] = [] # Mark bad channels if any
            # Convert to HbO/HbR
            raw_haemo = raw_od_for_beer_lambert # If already OD from XDF and correctly typed by MNE
            # If it's raw data that needs conversion to OD first, then Beer-Lambert:
            # raw_haemo = mne_nirs.optical_density(fnirs_raw_od) # if fnirs_raw_od is raw sensor data
            # p_logger.info(f"EV_{participant_id} - Converted fNIRS stream to optical density (if needed).")
            raw_haemo = mne_nirs.beer_lambert_law(raw_haemo, ppf=6.0)
            p_logger.info(f"EV_{participant_id} - fNIRS - Action: ApplyBeerLambert")
            p_logger.info(f"")

            # Motion correction (TDDR)
            corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            p_logger.info(f"EV_{participant_id} - fNIRS - Action: ApplyTDDR")
            p_logger.info(f"")

            # Filtering
            corrected_haemo.filter(0.01, 0.1, h_trans_bandwidth=0.02, l_trans_bandwidth=0.002, fir_design='firwin', verbose=False)
            p_logger.info(f"EV_{participant_id} - fNIRS - Action: Filter - Params: 0.01-0.1Hz")
            p_logger.info(f"")

            preprocessed_fnirs_file = os.path.join(processed_data_output_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
            corrected_haemo.save(preprocessed_fnirs_file, overwrite=True)
            p_logger.info(f"EV_{participant_id} - fNIRS - Action: SavePreprocessed - File: {preprocessed_fnirs_file}")
        else:
            p_logger.info(f"EV_{participant_id} - fNIRS - Status: Skipped (NoData)")
        p_logger.info(f"")

        # Dummy file to indicate completion
        with open(os.path.join(processed_data_output_dir, f"{participant_id}_preprocessing_manifest.txt"), "w") as f:
            f.write(f"Preprocessing completed for {participant_id} at {datetime.datetime.now().isoformat()}\n")
            # TODO: List successfully created files here

        p_logger.info(f"==== END Preprocessing - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"EV_{participant_id} - Preprocessing - Error: {e}", exc_info=True)
        p_logger.info(f"==== END Preprocessing - Status: FAILURE ====")
        return False

def run_analysis(participant_id, preprocessed_data_dir, analysis_results_dir, p_logger):
    """
    Participant data analysis based on EV_proposal.tex.
    Saves analysis metrics and reports into `analysis_results_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"==== Analysis ====") 
    p_logger.debug(f"EV_{participant_id} - Config - PreprocessedDataPath: {preprocessed_data_dir}") # EV_xxx prefix kept for debug clarity
    p_logger.debug(f"EV_{participant_id} - Config - AnalysisResultsPath: {analysis_results_dir}") # EV_xxx prefix kept for debug clarity

    os.makedirs(analysis_results_dir, exist_ok=True)
    analysis_metrics = {'participant_id': participant_id} 

    try:
        if not os.path.exists(preprocessed_data_dir):
            p_logger.error(f"EV_{participant_id} - Error: PreprocessedDataPathNotFound - Path: {preprocessed_data_dir}")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error(f"EV_{participant_id} - Error: LibrariesNotLoaded - Details: NumPy/MNE missing")
            return False

        # --- Load Preprocessed Data ---
        p_logger.info(f"\nEV_{participant_id} - Stage: LoadPreprocessedData") # Added newline and EV_xxx
        p_logger.info(f"")
        eeg_file = os.path.join(preprocessed_data_dir, f"{participant_id}_eeg_preprocessed.fif")
        nn_intervals_file = os.path.join(preprocessed_data_dir, f"{participant_id}_nn_intervals.csv")
        phasic_eda_file = os.path.join(preprocessed_data_dir, f"{participant_id}_phasic_eda.csv")
        fnirs_file = os.path.join(preprocessed_data_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
        rpeak_times_file = os.path.join(preprocessed_data_dir, f"{participant_id}_rpeaks_times_sec.csv")

        raw_eeg = mne.io.read_raw_fif(eeg_file, preload=True) if os.path.exists(eeg_file) else None
        nn_intervals_df = pd.read_csv(nn_intervals_file) if os.path.exists(nn_intervals_file) else None
        phasic_eda_df = pd.read_csv(phasic_eda_file) if os.path.exists(phasic_eda_file) else None
        fnirs_haemo = mne.io.read_raw_fif(fnirs_file, preload=True) if os.path.exists(fnirs_file) else None
        p_logger.info(f"EV_{participant_id} - Status: PreprocessedDataLoaded")
        p_logger.info(f"")

        # --- fNIRS ROI Analysis (GLM based on proposal) ---
        # Used to identify functionally relevant EEG channels for PLV
        p_logger.info(f"\nEV_{participant_id} - fNIRS - Stage: ROI_Analysis")
        p_logger.info(f"")
        if fnirs_haemo:
            events, event_id = mne.events_from_annotations(fnirs_haemo) # Assuming events are annotated
            # TODO: Define design matrix (stimulus onsets, durations, conditions)
            # design_matrix = make_first_level_design_matrix(fnirs_haemo.times, events, event_id=event_id,
            #                                                drift_model='polynomial', drift_order=3)
            # TODO: Run GLM
            # levels_results = run_glm(fnirs_haemo, design_matrix)
            # TODO: Extract contrasts for DLPFC, VMPFC for HbO2 (e.g., Emotion vs Neutral)
            # contrast_dlpfc_hbo = levels_results['Emotion_vs_Neutral'].to_dataframe(picks_types='hbo', ch_types='fnirs_cw_amplitude', regions=['DLPFC'])
            # TODO: Identify fNIRS channels/sources/detectors within defined ROIs (e.g., using channel locations)
            # fnirs_roi_channels = ['S1_D1', 'S2_D2'] # Example channels in an ROI
            # TODO: Map fNIRS ROI location to corresponding EEG channels for PLV analysis
            selected_eeg_channels_for_plv = ['Fp1', 'Fp2'] # Example EEG channels near fNIRS ROI
            # analysis_metrics['fnirs_dlpfc_hbo_contrast_mean'] = contrast_dlpfc_hbo['theta'].mean()
        p_logger.info(f"EV_{participant_id} - fNIRS - Status: ROI_Analysis_Complete")
        p_logger.info(f"")

        # --- EEG Analysis (Power, Phase for PLV, FAI) ---
        p_logger.info(f"\nEV_{participant_id} - EEG - Stage: Analysis")
        p_logger.info(f"")
        if raw_eeg:
            p_logger.info(f"EV_{participant_id} - EEG - Action: SegmentEpochs")
            events_eeg, event_id_eeg = mne.events_from_annotations(raw_eeg) # Assuming events are annotated in raw_eeg
            epochs = mne.Epochs(raw_eeg, events_eeg, event_id=event_id_eeg, tmin=-0.2, tmax=0.8, preload=True) # TODO: Adjust tmax
            # TODO: Calculate PSD for Alpha (8-13 Hz) and Beta (13-30 Hz) for F3, F4, Fp1, Fp2
            # (and potentially channels selected by fNIRS, ensure selected_eeg_channels_for_plv is defined)
            psd_picks = ['F3', 'F4', 'Fp1', 'Fp2']
            if 'selected_eeg_channels_for_plv' in locals() and selected_eeg_channels_for_plv:
                psd_picks = list(set(psd_picks + selected_eeg_channels_for_plv))
            
            p_logger.info(f"")
            psd_params_alpha = dict(fmin=8, fmax=13, method='welch', picks=psd_picks)
            p_logger.info(f"EV_{participant_id} - EEG - Action: CalculatePSD - Band: Alpha")
            psds_alpha, freqs_alpha = epochs.compute_psd(**psd_params_alpha).get_data(return_freqs=True)
            
            p_logger.info(f"")
            psd_params_beta = dict(fmin=13, fmax=30, method='welch', picks=psd_picks)
            p_logger.info(f"EV_{participant_id} - EEG - Action: CalculatePSD - Band: Beta")
            psds_beta, freqs_beta = epochs.compute_psd(**psd_params_beta).get_data(return_freqs=True)
            
            p_logger.info(f"")
            if 'F3' in raw_eeg.ch_names: analysis_metrics['alpha_power_f3_mean'] = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
            if 'F3' in raw_eeg.ch_names: analysis_metrics['beta_power_f3_mean'] = np.mean(psds_beta[:, raw_eeg.ch_names.index('F3'), :])
            # FAI Calculation
            p_logger.info(f"EV_{participant_id} - EEG - Action: CalculateFAI")
            if 'F4' in raw_eeg.ch_names and 'F3' in raw_eeg.ch_names:
                power_right_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :])
                power_left_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
                analysis_metrics['fai_f4_f3_alpha'] = np.log(power_right_alpha_f4 + 1e-9) - np.log(power_left_alpha_f3 + 1e-9)
            
            p_logger.info(f"")
            p_logger.info(f"EV_{participant_id} - EEG - Action: ExtractPhase - Bands: Alpha, Beta")
            # TODO: Filter EEG in Alpha/Beta for phase extraction (e.g., on continuous data or epochs)
            eeg_alpha_filtered = raw_eeg.copy().filter(8, 13, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            eeg_beta_filtered = raw_eeg.copy().filter(13, 30, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            phase_eeg_alpha = np.angle(hilbert(eeg_alpha_filtered.get_data())) # Shape (n_channels, n_times)
            phase_eeg_beta = np.angle(hilbert(eeg_beta_filtered.get_data())) # Shape (n_channels, n_times)
        p_logger.info(f"EV_{participant_id} - EEG - Status: AnalysisComplete")
        p_logger.info(f"")

        # --- HRV Analysis (RMSSD, Continuous Signal for PLV) ---
        p_logger.info(f"\nEV_{participant_id} - HRV - Stage: Analysis")
        p_logger.info(f"")
        if nn_intervals_df is not None:
            rmssd_results = nk.hrv_time(nn_intervals_df['NN_ms'], sampling_rate=1000) # Assuming NN_ms is already clean
            analysis_metrics['rmssd_overall'] = rmssd_results['HRV_RMSSD'].iloc[0]
            # Create continuous HRV signal (interpolate NN intervals at 4Hz)
            rpeak_times_sec_df = pd.read_csv(rpeak_times_file) if os.path.exists(rpeak_times_file) else None
            if rpeak_times_sec_df is not None:
                rpeak_times_sec = rpeak_times_sec_df['R_Peak_Time_s'].values
                nn_values_ms = nn_intervals_df['NN_ms'].values
                if len(rpeak_times_sec) > 1 and len(nn_values_ms) > 0 and len(rpeak_times_sec) == len(nn_values_ms) + 1 : # R-peaks are one more than NN-intervals
                    nn_interp_times = rpeak_times_sec[:-1] + np.diff(rpeak_times_sec)/2 # Midpoints of NN intervals
                    interp_func_hrv = interp1d(nn_interp_times, nn_values_ms, kind='cubic', fill_value="extrapolate")
                    # TODO: Determine target_time_hrv based on EEG/experiment timing for alignment
                    target_time_hrv = np.arange(nn_interp_times[0], nn_interp_times[-1], 1/4.0) # 4 Hz
                    continuous_hrv_signal = interp_func_hrv(target_time_hrv)
                    phase_hrv = np.angle(hilbert(continuous_hrv_signal - np.mean(continuous_hrv_signal)))
        p_logger.info(f"EV_{participant_id} - HRV - Status: AnalysisComplete")
        p_logger.info(f"")

        # --- EDA Analysis (Resample, Phase for PLV) ---
        p_logger.info(f"\nEV_{participant_id} - EDA - Stage: AnalysisForPLV")
        p_logger.info(f"")
        if phasic_eda_df is not None and not phasic_eda_df.empty:
            # Assuming original EDA sampling rate was used in NeuroKit2 (e.g., 1000 Hz)
            original_eda_sampling_rate = 1000 # TODO: Get actual original EDA sampling rate from data or config
            phasic_signal = phasic_eda_df['EDA_Phasic'].values
            original_eda_time = np.arange(len(phasic_signal)) / original_eda_sampling_rate
            interp_func_eda = interp1d(original_eda_time, phasic_signal, kind='linear', fill_value="extrapolate")
            # TODO: Determine target_time_eda based on EEG/experiment timing for alignment
            target_time_eda = np.arange(original_eda_time[0], original_eda_time[-1], 1/4.0) # Resample to 4 Hz
            resampled_phasic_eda = interp_func_eda(target_time_eda)
            phase_eda = np.angle(hilbert(resampled_phasic_eda - np.mean(resampled_phasic_eda)))
        p_logger.info(f"EV_{participant_id} - EDA - Status: AnalysisForPLV_Complete")
        p_logger.info(f"")

        # --- PLV Calculation ---
        p_logger.info(f"\nEV_{participant_id} - Connectivity - Stage: PLV_Calculation")
        p_logger.info(f"")
        # TODO: This requires careful epoching and alignment of EEG phase with resampled autonomic phases for specific time windows (e.g., stimulus duration).
        # Example for one EEG channel and one autonomic signal for a specific epoch:
        if 'phase_eeg_alpha' in locals() and 'phase_hrv' in locals():
            # TODO: Extract aligned epochs for phase_eeg_alpha (e.g., phase_eeg_alpha_roi_positive_epoch)
            # and phase_hrv (e.g., phase_hrv_positive_epoch)
            # This is highly dependent on your event structure and how you segment your data.
            # For demonstration, let's assume you have these aligned segments:
            # phase_eeg_alpha_roi_positive_epoch = phase_eeg_alpha[0, 1000:2000] # Example: channel 0, time 1000-2000
            # phase_hrv_positive_epoch = phase_hrv[50:150] # Example: corresponding segment in HRV phase
            # analysis_metrics['plv_eeg_alpha_hrv_roi_positive'] = calculate_plv(phase_eeg_alpha_roi_positive_epoch, phase_hrv_positive_epoch, p_logger)
            pass # Placeholder for actual PLV calculation with aligned epochs
        if 'phase_eeg_alpha' in locals() and 'phase_eda' in locals():
            # analysis_metrics['plv_eeg_alpha_eda_roi_positive'] = calculate_plv(phase_eeg_alpha_roi_positive_epoch, phase_eda_positive_epoch, p_logger)
            pass # Placeholder
        # TODO: Calculate PLV between EEG (Alpha/Beta from frontal channels, potentially selected by fNIRS) and continuous HRV/EDA
        # For WP1: PLV for each condition (Positive, Negative, Neutral)
        # ... repeat for Negative, Neutral conditions and Beta band
        p_logger.info(f"EV_{participant_id} - Connectivity - Status: PLV_Calculation_Complete")
        p_logger.info(f"")

        # --- FAI Calculation (WP4) ---
        p_logger.info(f"\nEV_{participant_id} - EEG - Stage: FAI_Calculation")
        p_logger.info(f"")
        if 'psds_alpha' in locals() and raw_eeg:
            # Calculate FAI for relevant pairs (e.g., F4/F3, Fp2/Fp1)
            # Ensure channels exist in raw_eeg.ch_names
            if 'F4' in raw_eeg.ch_names and 'F3' in raw_eeg.ch_names:
                power_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :]) # Average over epochs and freqs
                power_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
                analysis_metrics['fai_f4_f3_alpha'] = np.log(power_alpha_f4 + 1e-9) - np.log(power_alpha_f3 + 1e-9)
        p_logger.info(f"EV_{participant_id} - EEG - Status: FAI_Calculation_Complete")
        p_logger.info(f"")

        # Save all calculated metrics
        p_logger.info(f"\nEV_{participant_id} - Stage: SaveResults")
        p_logger.info(f"")
        p_logger.info(f"EV_{participant_id} - Action: SaveAnalysisMetrics")
        metrics_df = pd.DataFrame([analysis_metrics])
        metrics_output_file = os.path.join(analysis_results_dir, f"{participant_id}_analysis_metrics.csv")
        metrics_df.to_csv(metrics_output_file, index=False)
        p_logger.info(f"EV_{participant_id} - Status: AnalysisMetricsSaved - File: {metrics_output_file}")
        p_logger.info(f"")

        # --- Generate Participant-Specific Plots ---
        p_logger.info(f"\nEV_{participant_id} - Stage: GeneratePlots")
        p_logger.info(f"")
        generate_participant_plots(participant_id, analysis_results_dir, p_logger)

        # --- Statistical Testing (WP1, WP2, WP3, WP4) ---
        p_logger.info(f"--- Stage: StatisticalTesting ---")
        # TODO: Implement statistical tests using statsmodels or pingouin
        # This typically involves aggregating metrics across participants and running tests
        # on the combined data, not per participant within this loop.
        # The per-participant metrics saved above are the input for the group-level stats.
        # Example placeholder for a correlation (WP2, WP3, WP4):
        if 'subjective_arousal' in analysis_metrics and 'plv_eeg_alpha_hrv_roi_positive' in analysis_metrics:
            # This correlation would be done *after* collecting data from all participants
            # using a DataFrame of all participants' metrics.
            pass # Placeholder
        p_logger.info(f"EV_{participant_id} - Status: StatisticalTestingPlaceholder (Group-level stats after all participants)")
        p_logger.info(f"")

        # TODO: Generate analysis reports or visualizations if needed (e.g., using matplotlib, seaborn)

        p_logger.info(f"==== END Analysis - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"EV_{participant_id} - Analysis - Error: {e}", exc_info=True)
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
        p_logger.info(f"\nEV_{participant_id} - Stage: QuestionnaireParsing")
        questionnaire_data = parse_questionnaire_data(participant_id, participant_raw_data_path, p_logger)
        
        # Step 1: Preprocessing
        p_logger.info(f"\nEV_{participant_id} - Stage: Preprocessing")
        if not os.path.isdir(participant_raw_data_path):
            p_logger.error(f"EV_{participant_id} - Error: RawDataPathNotFound - Path: {participant_raw_data_path}")
            return False, questionnaire_data # Return questionnaire data even if other steps fail
        if not run_preprocessing(participant_id, participant_raw_data_path, participant_preprocessed_dir, p_logger):
            p_logger.error(f"EV_{participant_id} - Error: PreprocessingFailed")
            return False, questionnaire_data

        # Step 2: Analysis
        p_logger.info(f"\nEV_{participant_id} - Stage: Analysis")
        if not run_analysis(participant_id, participant_preprocessed_dir, participant_analysis_dir, p_logger):
            p_logger.error(f"EV_{participant_id} - Error: AnalysisFailed")
            return False, questionnaire_data

        # Step 3: Commit results to Git
        p_logger.info(f"\nEV_{participant_id} - Stage: GitCommit")
        commit_message = f"Pipeline: Processed data for participant {participant_id}"
        participant_log_file_actual_path = p_logger.handlers[0].baseFilename
        
        # Add the entire results directory for the participant (preprocessed + analysis)
        # and the participant's log file.
        files_to_add_to_git = [
            os.path.relpath(os.path.join(RESULTS_BASE_DIR, participant_id), BASE_REPO_PATH).replace("\\", "/"),
            os.path.relpath(participant_log_file_actual_path, BASE_REPO_PATH).replace("\\", "/")
        ]
        
        if not git_add_commit_push(BASE_REPO_PATH, commit_message, files_to_add_to_git, p_logger):
            p_logger.error(f"EV_{participant_id} - Error: GitCommitPushFailed")
            return False, questionnaire_data

        p_logger.info(f"EV_{participant_id} - Status: GitCommitPushSuccessful")
        success = True
    except Exception as e:
        p_logger.error(f"EV_{participant_id} - Error: UnhandledExceptionInProcessing - Details: {e}", exc_info=True)
        success = False
    finally:
        # --- Processing Summary ---
        p_logger.info(f"\n==== END Participant Processing: {participant_id} ====")
        if success:
            p_logger.info(f"EV_{participant_id} - Status: SUCCESS")
        else:
            p_logger.error(f"EV_{participant_id} - Status: FAILURE")
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
                else:
                    p_logger.warning(f"Participant: {participant_id} - EEG - Status: NoEEGChannelsFoundAfterIndex - StartIndex: {eeg_start_channel_index}")
            else:
                p_logger.warning(f"Participant: {participant_id} - EEG - Status: ChannelIndexOutOfRangeForEEGStart - ExpectedStartIndex: {eeg_start_channel_index}")
        else:
            p_logger.warning(f"Participant: {participant_id} - CombinedStream - Status: FileNotFound - File: {combined_stream_file}")

        # Example: Load fNIRS from a .nirs file (or similar MNE-NIRS compatible format)
        fnirs_data_file = os.path.join(raw_data_path, f"{participant_id}_fnirs.nirs") # Adjust filename as needed
        if os.path.exists(fnirs_data_file):
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: LoadFile - File: {fnirs_data_file}")
            # Use the appropriate MNE-NIRS reader, e.g., read_raw_nirx, read_raw_snirf, etc.
            # This example assumes a generic reader or that it's already optical density.
            # If it's raw intensity, you'll need mne_nirs.optical_density() later.
            fnirs_raw_od = mne.io.read_raw_nirx(fnirs_data_file, preload=True) # Replace with actual reader
            p_logger.info(f"Participant: {participant_id} - fNIRS - Status: Loaded - SFreq: {fnirs_raw_od.info['sfreq']} Hz")
            # If the loaded fNIRS data is not yet optical density, convert it:
            # fnirs_raw_od = mne_nirs.optical_density(fnirs_raw_od_or_intensity)
        else:
            p_logger.warning(f"Participant: {participant_id} - fNIRS - Status: FileNotFound - File: {fnirs_data_file}")

        # --- EEG Preprocessing (MNE-Python) ---
        if raw_eeg is not None:
            p_logger.info(f"--- Participant: {participant_id} - EEG - Stage: Preprocessing ---")
            p_logger.info(f"Participant: {participant_id} - EEG - Action: LoadDataToMemory") # If not already preloaded
            raw_eeg.load_data() # Explicitly load data for the EEG stream
            p_logger.info(f"Participant: {participant_id} - EEG - Action: Filter - Params: 0.5-40Hz")
            raw_eeg.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: SetReference - Type: Average")
            raw_eeg.set_eeg_reference('average', projection=True)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: FitICA - Components: 15")
            ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
            ica.fit(raw_eeg)
            # TODO: Manually or automatically identify and exclude artifact components
            # ica.exclude = [...] 
            ica.apply(raw_eeg, verbose=False)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: ApplyICA - Excluded: {ica.exclude if hasattr(ica, 'exclude') else 'None'}")
            preprocessed_eeg_file = os.path.join(processed_data_output_dir, f"{participant_id}_eeg_preprocessed.fif")
            raw_eeg.save(preprocessed_eeg_file, overwrite=True)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: SavePreprocessed - File: {preprocessed_eeg_file}")
        else:
            p_logger.info(f"Participant: {participant_id} - EEG - Status: Skipped (NoData)")

        # --- ECG Preprocessing (NeuroKit2) ---
        if ecg_signal_raw is not None and ecg_sampling_rate is not None:
            p_logger.info(f"--- Participant: {participant_id} - ECG - Stage: Preprocessing ---")
            p_logger.info(f"Participant: {participant_id} - ECG - Action: CleanSignal")
            ecg_cleaned = nk.ecg_clean(ecg_signal_raw, sampling_rate=sampling_rate_ecg)
            p_logger.info(f"Participant: {participant_id} - ECG - Action: DetectRPeaks")
            peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate_ecg, correct_artifacts=True)
            rpeaks = peaks_info[0]['ECG_R_Peaks']
            p_logger.info(f"Participant: {participant_id} - ECG - Status: RPeaksDetected - Count: {len(rpeaks)}")
            p_logger.info(f"Participant: {participant_id} - ECG - Action: CalculateNNIntervals")
            nn_intervals = np.diff(rpeaks) / ecg_sampling_rate * 1000 # in ms
            nn_intervals_file = os.path.join(processed_data_output_dir, f"{participant_id}_nn_intervals.csv")
            pd.DataFrame(nn_intervals, columns=['NN_ms']).to_csv(nn_intervals_file, index=False)
            p_logger.info(f"Participant: {participant_id} - ECG - Action: SaveNNIntervals - File: {nn_intervals_file}")
            rpeaks_times_file = os.path.join(processed_data_output_dir, f"{participant_id}_rpeaks_times_sec.csv")
            pd.DataFrame(rpeaks / ecg_sampling_rate, columns=['R_Peak_Time_s']).to_csv(rpeaks_times_file, index=False)
            p_logger.info(f"Participant: {participant_id} - ECG - Action: SaveRPeakTimes - File: {rpeaks_times_file}")
        else:
            p_logger.info(f"Participant: {participant_id} - ECG - Status: Skipped (NoData)")

        # --- EDA Preprocessing (NeuroKit2) ---
        if eda_signal_raw is not None and eda_sampling_rate is not None:
            p_logger.info(f"--- Participant: {participant_id} - EDA - Stage: Preprocessing ---")
            p_logger.info(f"Participant: {participant_id} - EDA - Action: ProcessSignal (Decomposition)")
            eda_signals, info = nk.eda_process(eda_signal_raw, sampling_rate=sampling_rate_eda)
            phasic_eda = eda_signals['EDA_Phasic']
            tonic_eda = eda_signals['EDA_Tonic']
            p_logger.info(f"Participant: {participant_id} - EDA - Status: Decomposed")
            phasic_eda_file = os.path.join(processed_data_output_dir, f"{participant_id}_phasic_eda.csv")
            if isinstance(phasic_eda, (pd.Series, pd.DataFrame)):
                phasic_eda.to_csv(phasic_eda_file, index=False, header=True)
            else: # If it's a numpy array
                pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_file, index=False, header=True)
            p_logger.info(f"Participant: {participant_id} - EDA - Action: SavePhasicEDA - File: {phasic_eda_file}")
        else:
            p_logger.info(f"Participant: {participant_id} - EDA - Status: Skipped (NoData)")

        # --- fNIRS Preprocessing (MNE-NIRS) ---
        if fnirs_raw_od is not None: # fnirs_raw_od is the MNE Raw object for optical density from XDF
            p_logger.info(f"--- Participant: {participant_id} - fNIRS - Stage: Preprocessing ---")
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: LoadDataToMemory") # If not already preloaded
            fnirs_raw_od.load_data() # Explicitly load data for the fNIRS OD stream
            # Ensure it's treated as optical density if it's not already marked as such by read_raw_xdf
            # This might involve checking channel types or manually setting them if necessary.
            # For now, assume fnirs_raw_od is ready for Beer-Lambert or is raw voltages needing optical_density()
            raw_od_for_beer_lambert = fnirs_raw_od # Default assumption
            p_logger.info(f"EV_{participant_id} - fNIRS - Info: Assuming loaded data is optical density or will be converted if necessary before Beer-Lambert.")
            
            # If your XDF fNIRS stream provides raw voltages, you might need mne_nirs.optical_density() first
            # Example: if 'fnirs_raw' in fnirs_raw_od.info.get('ch_types', []): # Hypothetical check
            #    raw_od_for_beer_lambert = mne_nirs.optical_density(fnirs_raw_od)
            #    p_logger.info("Converted fNIRS raw voltages to optical density.")
            
            raw_od_for_beer_lambert.info['bads'] = [] # Mark bad channels if any
            # Convert to HbO/HbR
            raw_haemo = raw_od_for_beer_lambert # If already OD from XDF and correctly typed by MNE
            # If it's raw data that needs conversion to OD first, then Beer-Lambert:
            # raw_haemo = mne_nirs.optical_density(fnirs_raw_od) # if fnirs_raw_od is raw sensor data
            # p_logger.info("Converted fNIRS stream to optical density (if needed).")
            raw_haemo = mne_nirs.beer_lambert_law(raw_haemo, ppf=6.0)
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: ApplyBeerLambert")
            # Motion correction (TDDR)
            corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: ApplyTDDR")
            # Filtering
            corrected_haemo.filter(0.01, 0.1, h_trans_bandwidth=0.02, l_trans_bandwidth=0.002, fir_design='firwin', verbose=False)
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: Filter - Params: 0.01-0.1Hz")
            preprocessed_fnirs_file = os.path.join(processed_data_output_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
            corrected_haemo.save(preprocessed_fnirs_file, overwrite=True)
            p_logger.info(f"Participant: {participant_id} - fNIRS - Action: SavePreprocessed - File: {preprocessed_fnirs_file}")
        else:
            p_logger.info(f"Participant: {participant_id} - fNIRS - Status: Skipped (NoData)")

        # Dummy file to indicate completion
        with open(os.path.join(processed_data_output_dir, f"{participant_id}_preprocessing_manifest.txt"), "w") as f:
            f.write(f"Preprocessing completed for {participant_id} at {datetime.datetime.now().isoformat()}\n")
            # TODO: List successfully created files here

        p_logger.info(f"==== END Preprocessing - Participant: {participant_id} - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"Participant: {participant_id} - Preprocessing - Error: {e}", exc_info=True)
        p_logger.info(f"==== END Preprocessing - Participant: {participant_id} - Status: FAILURE ====")
        return False

def run_analysis(participant_id, preprocessed_data_dir, analysis_results_dir, p_logger):
    """
    Participant data analysis based on EV_proposal.tex.
    Saves analysis metrics and reports into `analysis_results_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"==== START Analysis - Participant: {participant_id} ====")
    p_logger.debug(f"Participant: {participant_id} - Config - PreprocessedDataPath: {preprocessed_data_dir}")
    p_logger.debug(f"Participant: {participant_id} - Config - AnalysisResultsPath: {analysis_results_dir}")

    os.makedirs(analysis_results_dir, exist_ok=True)
    analysis_metrics = {'participant_id': participant_id} 

    try:
        if not os.path.exists(preprocessed_data_dir):
            p_logger.error(f"Participant: {participant_id} - Error: PreprocessedDataPathNotFound - Path: {preprocessed_data_dir}")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error(f"Participant: {participant_id} - Error: LibrariesNotLoaded - Details: NumPy/MNE missing")
            return False

        # --- Load Preprocessed Data ---
        p_logger.info(f"--- Participant: {participant_id} - Stage: LoadPreprocessedData ---")
        eeg_file = os.path.join(preprocessed_data_dir, f"{participant_id}_eeg_preprocessed.fif")
        nn_intervals_file = os.path.join(preprocessed_data_dir, f"{participant_id}_nn_intervals.csv")
        phasic_eda_file = os.path.join(preprocessed_data_dir, f"{participant_id}_phasic_eda.csv")
        fnirs_file = os.path.join(preprocessed_data_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
        rpeak_times_file = os.path.join(preprocessed_data_dir, f"{participant_id}_rpeaks_times_sec.csv")

        raw_eeg = mne.io.read_raw_fif(eeg_file, preload=True) if os.path.exists(eeg_file) else None
        nn_intervals_df = pd.read_csv(nn_intervals_file) if os.path.exists(nn_intervals_file) else None
        phasic_eda_df = pd.read_csv(phasic_eda_file) if os.path.exists(phasic_eda_file) else None
        fnirs_haemo = mne.io.read_raw_fif(fnirs_file, preload=True) if os.path.exists(fnirs_file) else None
        p_logger.info(f"Participant: {participant_id} - Status: PreprocessedDataLoaded")

        # --- fNIRS ROI Analysis (GLM based on proposal) ---
        # Used to identify functionally relevant EEG channels for PLV
        p_logger.info(f"--- Participant: {participant_id} - fNIRS - Stage: ROI_Analysis ---")
        if fnirs_haemo:
            events, event_id = mne.events_from_annotations(fnirs_haemo) # Assuming events are annotated
            # TODO: Define design matrix (stimulus onsets, durations, conditions)
            # design_matrix = make_first_level_design_matrix(fnirs_haemo.times, events, event_id=event_id,
            #                                                drift_model='polynomial', drift_order=3)
            # TODO: Run GLM
            # levels_results = run_glm(fnirs_haemo, design_matrix)
            # TODO: Extract contrasts for DLPFC, VMPFC for HbO2 (e.g., Emotion vs Neutral)
            # contrast_dlpfc_hbo = levels_results['Emotion_vs_Neutral'].to_dataframe(picks_types='hbo', ch_types='fnirs_cw_amplitude', regions=['DLPFC'])
            # TODO: Identify fNIRS channels/sources/detectors within defined ROIs (e.g., using channel locations)
            # fnirs_roi_channels = ['S1_D1', 'S2_D2'] # Example channels in an ROI
            # TODO: Map fNIRS ROI location to corresponding EEG channels for PLV analysis
            selected_eeg_channels_for_plv = ['Fp1', 'Fp2'] # Example EEG channels near fNIRS ROI
            # analysis_metrics['fnirs_dlpfc_hbo_contrast_mean'] = contrast_dlpfc_hbo['theta'].mean()
        p_logger.info(f"Participant: {participant_id} - fNIRS - Status: ROI_Analysis_Complete")

        # --- EEG Analysis (Power, Phase for PLV, FAI) ---
        p_logger.info(f"--- Participant: {participant_id} - EEG - Stage: Analysis ---")
        if raw_eeg:
            p_logger.info(f"Participant: {participant_id} - EEG - Action: SegmentEpochs")
            events_eeg, event_id_eeg = mne.events_from_annotations(raw_eeg) # Assuming events are annotated in raw_eeg
            epochs = mne.Epochs(raw_eeg, events_eeg, event_id=event_id_eeg, tmin=-0.2, tmax=0.8, preload=True) # TODO: Adjust tmax
            # TODO: Calculate PSD for Alpha (8-13 Hz) and Beta (13-30 Hz) for F3, F4, Fp1, Fp2
            # (and potentially channels selected by fNIRS, ensure selected_eeg_channels_for_plv is defined)
            psd_picks = ['F3', 'F4', 'Fp1', 'Fp2']
            if 'selected_eeg_channels_for_plv' in locals() and selected_eeg_channels_for_plv:
                psd_picks = list(set(psd_picks + selected_eeg_channels_for_plv))
            
            psd_params_alpha = dict(fmin=8, fmax=13, method='welch', picks=psd_picks)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: CalculatePSD - Band: Alpha")
            psds_alpha, freqs_alpha = epochs.compute_psd(**psd_params_alpha).get_data(return_freqs=True)
            
            psd_params_beta = dict(fmin=13, fmax=30, method='welch', picks=psd_picks)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: CalculatePSD - Band: Beta")
            psds_beta, freqs_beta = epochs.compute_psd(**psd_params_beta).get_data(return_freqs=True)
            
            if 'F3' in raw_eeg.ch_names: analysis_metrics['alpha_power_f3_mean'] = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
            if 'F3' in raw_eeg.ch_names: analysis_metrics['beta_power_f3_mean'] = np.mean(psds_beta[:, raw_eeg.ch_names.index('F3'), :])
            # FAI Calculation
            p_logger.info(f"Participant: {participant_id} - EEG - Action: CalculateFAI")
            if 'F4' in raw_eeg.ch_names and 'F3' in raw_eeg.ch_names:
                power_right_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :])
                power_left_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
                analysis_metrics['fai_f4_f3_alpha'] = np.log(power_right_alpha_f4 + 1e-9) - np.log(power_left_alpha_f3 + 1e-9)
            p_logger.info(f"Participant: {participant_id} - EEG - Action: ExtractPhase - Bands: Alpha, Beta")
            # TODO: Filter EEG in Alpha/Beta for phase extraction (e.g., on continuous data or epochs)
            eeg_alpha_filtered = raw_eeg.copy().filter(8, 13, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            eeg_beta_filtered = raw_eeg.copy().filter(13, 30, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            phase_eeg_alpha = np.angle(hilbert(eeg_alpha_filtered.get_data())) # Shape (n_channels, n_times)
            phase_eeg_beta = np.angle(hilbert(eeg_beta_filtered.get_data())) # Shape (n_channels, n_times)
        p_logger.info(f"Participant: {participant_id} - EEG - Status: AnalysisComplete")

        # --- HRV Analysis (RMSSD, Continuous Signal for PLV) ---
        p_logger.info(f"--- Participant: {participant_id} - HRV - Stage: Analysis ---")
        if nn_intervals_df is not None:
            rmssd_results = nk.hrv_time(nn_intervals_df['NN_ms'], sampling_rate=1000) # Assuming NN_ms is already clean
            analysis_metrics['rmssd_overall'] = rmssd_results['HRV_RMSSD'].iloc[0]
            # Create continuous HRV signal (interpolate NN intervals at 4Hz)
            rpeak_times_sec_df = pd.read_csv(rpeak_times_file) if os.path.exists(rpeak_times_file) else None
            if rpeak_times_sec_df is not None:
                rpeak_times_sec = rpeak_times_sec_df['R_Peak_Time_s'].values
                nn_values_ms = nn_intervals_df['NN_ms'].values
                if len(rpeak_times_sec) > 1 and len(nn_values_ms) > 0 and len(rpeak_times_sec) == len(nn_values_ms) + 1 : # R-peaks are one more than NN-intervals
                    nn_interp_times = rpeak_times_sec[:-1] + np.diff(rpeak_times_sec)/2 # Midpoints of NN intervals
                    interp_func_hrv = interp1d(nn_interp_times, nn_values_ms, kind='cubic', fill_value="extrapolate")
                    # TODO: Determine target_time_hrv based on EEG/experiment timing for alignment
                    target_time_hrv = np.arange(nn_interp_times[0], nn_interp_times[-1], 1/4.0) # 4 Hz
                    continuous_hrv_signal = interp_func_hrv(target_time_hrv)
                    phase_hrv = np.angle(hilbert(continuous_hrv_signal - np.mean(continuous_hrv_signal)))
        p_logger.info(f"Participant: {participant_id} - HRV - Status: AnalysisComplete")

        # --- EDA Analysis (Resample, Phase for PLV) ---
        p_logger.info(f"--- Participant: {participant_id} - EDA - Stage: AnalysisForPLV ---")
        if phasic_eda_df is not None and not phasic_eda_df.empty:
            # Assuming original EDA sampling rate was used in NeuroKit2 (e.g., 1000 Hz)
            original_eda_sampling_rate = 1000 # TODO: Get actual original EDA sampling rate from data or config
            phasic_signal = phasic_eda_df['EDA_Phasic'].values
            original_eda_time = np.arange(len(phasic_signal)) / original_eda_sampling_rate
            interp_func_eda = interp1d(original_eda_time, phasic_signal, kind='linear', fill_value="extrapolate")
            # TODO: Determine target_time_eda based on EEG/experiment timing for alignment
            target_time_eda = np.arange(original_eda_time[0], original_eda_time[-1], 1/4.0) # Resample to 4 Hz
            resampled_phasic_eda = interp_func_eda(target_time_eda)
            phase_eda = np.angle(hilbert(resampled_phasic_eda - np.mean(resampled_phasic_eda)))
        p_logger.info(f"Participant: {participant_id} - EDA - Status: AnalysisForPLV_Complete")

        # --- PLV Calculation ---
        p_logger.info(f"--- Participant: {participant_id} - Connectivity - Stage: PLV_Calculation ---")
        # TODO: This requires careful epoching and alignment of EEG phase with resampled autonomic phases for specific time windows (e.g., stimulus duration).
        # Example for one EEG channel and one autonomic signal for a specific epoch:
        if 'phase_eeg_alpha' in locals() and 'phase_hrv' in locals():
            # TODO: Extract aligned epochs for phase_eeg_alpha (e.g., phase_eeg_alpha_roi_positive_epoch)
            # and phase_hrv (e.g., phase_hrv_positive_epoch)
            # This is highly dependent on your event structure and how you segment your data.
            # For demonstration, let's assume you have these aligned segments:
            # phase_eeg_alpha_roi_positive_epoch = phase_eeg_alpha[0, 1000:2000] # Example: channel 0, time 1000-2000
            # phase_hrv_positive_epoch = phase_hrv[50:150] # Example: corresponding segment in HRV phase
            # analysis_metrics['plv_eeg_alpha_hrv_roi_positive'] = calculate_plv(phase_eeg_alpha_roi_positive_epoch, phase_hrv_positive_epoch, p_logger)
            pass # Placeholder for actual PLV calculation with aligned epochs
        if 'phase_eeg_alpha' in locals() and 'phase_eda' in locals():
            # analysis_metrics['plv_eeg_alpha_eda_roi_positive'] = calculate_plv(phase_eeg_alpha_roi_positive_epoch, phase_eda_positive_epoch, p_logger)
            pass # Placeholder
        # TODO: Calculate PLV between EEG (Alpha/Beta from frontal channels, potentially selected by fNIRS) and continuous HRV/EDA
        # For WP1: PLV for each condition (Positive, Negative, Neutral)
        # ... repeat for Negative, Neutral conditions and Beta band
        p_logger.info(f"Participant: {participant_id} - Connectivity - Status: PLV_Calculation_Complete")

        # --- FAI Calculation (WP4) ---
        p_logger.info(f"--- Participant: {participant_id} - EEG - Stage: FAI_Calculation ---")
        if 'psds_alpha' in locals() and raw_eeg:
            # Calculate FAI for relevant pairs (e.g., F4/F3, Fp2/Fp1)
            # Ensure channels exist in raw_eeg.ch_names
            if 'F4' in raw_eeg.ch_names and 'F3' in raw_eeg.ch_names:
                power_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :]) # Average over epochs and freqs
                power_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
                analysis_metrics['fai_f4_f3_alpha'] = np.log(power_alpha_f4 + 1e-9) - np.log(power_alpha_f3 + 1e-9)
        p_logger.info(f"Participant: {participant_id} - EEG - Status: FAI_Calculation_Complete")

        # Save all calculated metrics
        p_logger.info(f"--- Participant: {participant_id} - Stage: SaveResults ---")
        p_logger.info(f"Participant: {participant_id} - Action: SaveAnalysisMetrics")
        metrics_df = pd.DataFrame([analysis_metrics])
        metrics_output_file = os.path.join(analysis_results_dir, f"{participant_id}_analysis_metrics.csv")
        metrics_df.to_csv(metrics_output_file, index=False)
        p_logger.info(f"Participant: {participant_id} - Status: AnalysisMetricsSaved - File: {metrics_output_file}")

        # --- Generate Participant-Specific Plots ---
        p_logger.info(f"--- Participant: {participant_id} - Stage: GeneratePlots ---")
        generate_participant_plots(participant_id, analysis_results_dir, p_logger)

        # --- Statistical Testing (WP1, WP2, WP3, WP4) ---
        p_logger.info(f"--- Participant: {participant_id} - Stage: StatisticalTesting ---")
        # TODO: Implement statistical tests using statsmodels or pingouin
        # This typically involves aggregating metrics across participants and running tests
        # on the combined data, not per participant within this loop.
        # The per-participant metrics saved above are the input for the group-level stats.
        # Example placeholder for a correlation (WP2, WP3, WP4):
        if 'subjective_arousal' in analysis_metrics and 'plv_eeg_alpha_hrv_roi_positive' in analysis_metrics:
            # This correlation would be done *after* collecting data from all participants
            # using a DataFrame of all participants' metrics.
            pass # Placeholder
        p_logger.info(f"Participant: {participant_id} - Status: StatisticalTestingPlaceholder (Group-level stats after all participants)")

        # TODO: Generate analysis reports or visualizations if needed (e.g., using matplotlib, seaborn)

        p_logger.info(f"==== END Analysis - Participant: {participant_id} - Status: SUCCESS ====")
        return True
    except Exception as e:
        p_logger.error(f"Participant: {participant_id} - Analysis - Error: {e}", exc_info=True)
        p_logger.info(f"==== END Analysis - Participant: {participant_id} - Status: FAILURE ====")
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
        p_logger.info(f"==== START Participant Processing - ID: {participant_id} ====")
        participant_preprocessed_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "preprocessed")
        participant_analysis_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "analysis")

        # Step 0: Parse Questionnaire Data (if exists)
        p_logger.info(f"--- Participant: {participant_id} - Stage: QuestionnaireParsing ---")
        questionnaire_data = parse_questionnaire_data(participant_id, participant_raw_data_path, p_logger)
        
        # Step 1: Preprocessing
        p_logger.info(f"--- Participant: {participant_id} - Stage: Preprocessing ---")
        if not os.path.isdir(participant_raw_data_path):
            p_logger.error(f"Participant: {participant_id} - Error: RawDataPathNotFound - Path: {participant_raw_data_path}")
            return False, questionnaire_data # Return questionnaire data even if other steps fail
        if not run_preprocessing(participant_id, participant_raw_data_path, participant_preprocessed_dir, p_logger):
            p_logger.error(f"Participant: {participant_id} - Error: PreprocessingFailed")
            return False, questionnaire_data

        # Step 2: Analysis
        p_logger.info(f"--- Participant: {participant_id} - Stage: Analysis ---")
        if not run_analysis(participant_id, participant_preprocessed_dir, participant_analysis_dir, p_logger):
            p_logger.error(f"Participant: {participant_id} - Error: AnalysisFailed")
            return False, questionnaire_data

        # Step 3: Commit results to Git
        p_logger.info(f"--- Participant: {participant_id} - Stage: GitCommit ---")
        commit_message = f"Pipeline: Processed data for participant {participant_id}"
        participant_log_file_actual_path = p_logger.handlers[0].baseFilename
        
        # Add the entire results directory for the participant (preprocessed + analysis)
        # and the participant's log file.
        files_to_add_to_git = [
            os.path.relpath(os.path.join(RESULTS_BASE_DIR, participant_id), BASE_REPO_PATH).replace("\\", "/"),
            os.path.relpath(participant_log_file_actual_path, BASE_REPO_PATH).replace("\\", "/")
        ]
        
        if not git_add_commit_push(BASE_REPO_PATH, commit_message, files_to_add_to_git, p_logger):
            p_logger.error(f"Participant: {participant_id} - Error: GitCommitPushFailed")
            return False, questionnaire_data

        p_logger.info(f"Participant: {participant_id} - Status: GitCommitPushSuccessful")
        success = True
    except Exception as e:
        p_logger.error(f"Participant: {participant_id} - Error: UnhandledExceptionInProcessing - Details: {e}", exc_info=True)
        success = False
    finally:
        if success:
            p_logger.info(f"==== END Participant Processing - ID: {participant_id} - Status: SUCCESS ====")
        else:
            p_logger.error(f"==== END Participant Processing - ID: {participant_id} - Status: FAILURE ====")
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
