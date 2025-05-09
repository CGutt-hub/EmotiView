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
import mne_nirs # For fNIRS specific functions

# --- Configuration ---
# TODO: User should adjust these paths as needed
# Ensure BASE_REPO_PATH points to the root of your Git repository.
BASE_REPO_PATH = r"D:\repoShaggy\EmotiView"

# Directory where raw data for each participant is stored.
# The script expects subdirectories here, one for each participant_id.
# For testing with pilotData:
# PARTICIPANT_DATA_BASE_DIR = os.path.join(BASE_REPO_PATH, "rawData", "pilotData") # Original for pilot
PARTICIPANT_DATA_BASE_DIR = os.path.join(BASE_REPO_PATH, "rawData") # For monitoring XDF files in rawData root
# Original: PARTICIPANT_DATA_BASE_DIR = os.path.join(BASE_REPO_PATH, "data", "raw")

# Base directory to save processed data and analysis results.
# Results for each participant will be stored in a subdirectory named after their ID.
# Updated to EV_results:
RESULTS_BASE_DIR = os.path.join(BASE_REPO_PATH, "EV_results")

# Directory to store log files.
LOG_DIR = os.path.join(BASE_REPO_PATH, "logs", "pipeline_runs")

# --- Questionnaire Configuration ---
QUESTIONNAIRE_TXT_FILENAME = "questionnaire.txt" # Name of the questionnaire text file in each participant's raw data folder
AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME = "aggregated_questionnaires.xlsx"
MONITOR_INTERVAL_SECONDS = 60 # How often to scan for new XDF files

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

    logger = logging.getLogger(f"Participant_{participant_id}")
    logger.setLevel(logging.DEBUG) # Capture all levels for participant log
    logger.propagate = False # Prevent logs from bubbling up to the main logger

    # Ensure the directory for the log file exists (should be created by run_pipeline before this)
    os.makedirs(participant_log_directory_path, exist_ok=True)

    log_filename = f"EV_{participant_id}_processingLog.txt"
    participant_log_file_path = os.path.join(participant_log_directory_path, log_filename)

    fh = logging.FileHandler(participant_log_file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
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
        p_logger_local.warning("Cannot calculate PLV with zero-length phase signals.")
        return np.nan
    
    # Truncate to the minimum length
    phase1_aligned = phase1[:min_len]
    phase2_aligned = phase2[:min_len]

    # Calculate phase difference
    phase_difference = phase1_aligned - phase2_aligned

    # Calculate PLV
    plv = np.abs(np.mean(np.exp(1j * phase_difference)))
    
    # p_logger_local.debug(f"Calculated PLV: {plv}")
    return plv


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
        p_logger.info(f"Questionnaire file not found for participant {participant_id} at {questionnaire_file_path}")
        return None

    try:
        with open(questionnaire_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or ':' not in line:
                    if line: # Non-empty line without colon
                        p_logger.warning(f"Skipping malformed line {line_num} in {questionnaire_file_path} for {participant_id}: '{line}' (no colon found)")
                    continue
                key, value = line.split(':', 1)
                parsed_data[key.strip()] = value.strip()
        p_logger.info(f"Successfully parsed questionnaire for {participant_id} from {questionnaire_file_path}")
        return parsed_data
    except Exception as e:
        p_logger.error(f"Error parsing questionnaire file {questionnaire_file_path} for {participant_id}: {e}", exc_info=True)
        return None

# --- Placeholder Processing Functions ---

def run_preprocessing(participant_id, raw_data_path, processed_data_output_dir, p_logger):
    """
    Participant data preprocessing based on EV_proposal.tex.
    Saves preprocessed files into `processed_data_output_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"Starting preprocessing for participant {participant_id}...")
    p_logger.info(f"Raw data location: {raw_data_path}")
    p_logger.info(f"Processed data output directory: {processed_data_output_dir}")

    os.makedirs(processed_data_output_dir, exist_ok=True)

    try:
        if not os.path.exists(raw_data_path):
            p_logger.error(f"Raw data path {raw_data_path} does not exist.")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error("Scientific libraries (NumPy, MNE, etc.) not loaded. Cannot perform preprocessing.")
            return False

        # --- Load XDF and Extract Streams ---
        p_logger.info("Looking for XDF file...")
        # The XDF file was moved into participant_specific_raw_dir, which is `raw_data_path` here.
        # The filename is participant_id + ".xdf" (or original XDF filename if different)
        # Let's find the first .xdf file in the raw_data_path directory
        xdf_file_name_in_dir = None
        for f_name in os.listdir(raw_data_path):
            if f_name.lower().endswith(".xdf"):
                xdf_file_name_in_dir = f_name
                break
        
        if not xdf_file_name_in_dir:
            p_logger.error(f"No XDF file found in {raw_data_path} for participant {participant_id}")
            return False
            
        xdf_file_path = os.path.join(raw_data_path, xdf_file_name_in_dir)
        
        raw_eeg, ecg_signal_raw, eda_signal_raw, fnirs_raw_od = None, None, None, None # Initialize
        eeg_sampling_rate, ecg_sampling_rate, eda_sampling_rate = None, None, None

        try:
            p_logger.info(f"Loading XDF file: {xdf_file_path}")
            # Optimization: Load XDF metadata first (preload=False).
            # Data for each stream will be loaded into memory only when needed.
            streams_metadata = mne.io.read_raw_xdf(xdf_file_path, preload=False, stream_id=None)
            streams = streams_metadata if isinstance(streams_metadata, list) else [streams_metadata]
            
            # Ensure streams_data is a list, even if only one stream is found
            streams = streams_data if isinstance(streams_data, list) else [streams_data]

            p_logger.info(f"Found {len(streams)} stream(s) in {xdf_file_path}.")
            for stream_idx, stream in enumerate(streams):
                stream_name = stream.info.get('description') or stream.info.get('name') or f"Stream_{stream_idx}"
                stream_type = stream.info.get('type', 'UnknownType').lower()
                num_channels = len(stream.ch_names)
                sfreq = stream.info['sfreq']
                p_logger.info(f"Stream {stream_idx}: Name='{stream_name}', Type='{stream_type}', #Chans={num_channels}, SFreq={sfreq}")

                # --- !!! USER CUSTOMIZATION REQUIRED FOR STREAM IDENTIFICATION !!! ---
                # TODO: Adapt these conditions to reliably identify your streams based on their names, types, channel counts, etc.
                if raw_eeg is None and ('eeg' in stream_type or num_channels >= 8): # Example: EEG stream
                    raw_eeg = stream # This is still a metadata-only Raw object
                    eeg_sampling_rate = sfreq
                    p_logger.info(f"Identified EEG stream: '{stream_name}' (SFreq: {eeg_sampling_rate} Hz).")
                elif ecg_signal_raw is None and ('ecg' in stream_type or 'EKG' in stream_name.upper()):
                    # Load data for this specific stream now
                    ecg_signal_raw = stream.copy().load_data().get_data()[0] # Assuming single channel ECG
                    ecg_sampling_rate = sfreq
                    p_logger.info(f"Identified ECG stream: '{stream_name}' (SFreq: {ecg_sampling_rate} Hz).")
                elif eda_signal_raw is None and ('eda' in stream_type or 'gsr' in stream_type or 'GSR' in stream_name.upper()):
                    # Load data for this specific stream now
                    eda_signal_raw = stream.copy().load_data().get_data()[0] # Assuming single channel EDA
                    eda_sampling_rate = sfreq
                    p_logger.info(f"Identified EDA stream: '{stream_name}' (SFreq: {eda_sampling_rate} Hz).")
                elif fnirs_raw_od is None and ('nirs' in stream_type or 'nirx' in stream_name.lower()): # Check for fNIRS optical density
                    fnirs_raw_od = stream # This is still a metadata-only Raw object
                    p_logger.info(f"Identified fNIRS OD stream: '{stream_name}' (SFreq: {sfreq} Hz).")
                # --- END OF USER CUSTOMIZATION SECTION ---

            if raw_eeg is None:
                p_logger.warning(f"Could not identify dedicated EEG stream in {xdf_file_path}. EEG processing might be affected or skipped.")

        except Exception as e:
            p_logger.error(f"Error loading or parsing XDF file {xdf_file_path}: {e}", exc_info=True)
            return False

        # --- EEG Preprocessing (MNE-Python) ---
        if raw_eeg is not None:
            p_logger.info("Starting EEG preprocessing...")
            p_logger.info("Loading EEG data into memory...")
            raw_eeg.load_data() # Explicitly load data for the EEG stream
            p_logger.info(f"Using loaded EEG data from XDF (SFreq: {eeg_sampling_rate} Hz).")
            raw_eeg.filter(l_freq=0.5, h_freq=40., fir_design='firwin')
            p_logger.info("Applied band-pass filter (0.5-40 Hz).")
            raw_eeg.set_eeg_reference('average', projection=True)
            p_logger.info("Applied average reference.")
            ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
            ica.fit(raw_eeg)
            # TODO: Manually or automatically identify and exclude artifact components
            # ica.exclude = [...] 
            ica.apply(raw_eeg)
            p_logger.info(f"Applied ICA artifact correction (excluded components: {ica.exclude}).")
            preprocessed_eeg_file = os.path.join(processed_data_output_dir, f"{participant_id}_eeg_preprocessed.fif")
            raw_eeg.save(preprocessed_eeg_file, overwrite=True) # Save in MNE's .fif format
            p_logger.info(f"EEG data preprocessed and saved to {preprocessed_eeg_file}")
        else:
            p_logger.info("Skipping EEG preprocessing as no EEG stream was loaded/identified.")

        # --- ECG Preprocessing (NeuroKit2) ---
        if ecg_signal_raw is not None and ecg_sampling_rate is not None:
            p_logger.info("Starting ECG processing...")
            ecg_cleaned = nk.ecg_clean(ecg_signal_raw, sampling_rate=sampling_rate_ecg)
            p_logger.info("Cleaned ECG signal.")
            peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate_ecg, correct_artifacts=True)
            rpeaks = peaks_info[0]['ECG_R_Peaks']
            p_logger.info(f"Detected {len(rpeaks)} R-peaks.")
            nn_intervals = np.diff(rpeaks) / ecg_sampling_rate * 1000 # in ms
            nn_intervals_file = os.path.join(processed_data_output_dir, f"{participant_id}_nn_intervals.csv")
            pd.DataFrame(nn_intervals, columns=['NN_ms']).to_csv(nn_intervals_file, index=False)
            p_logger.info(f"NN intervals extracted and saved to {nn_intervals_file}")
            rpeaks_times_file = os.path.join(processed_data_output_dir, f"{participant_id}_rpeaks_times_sec.csv")
            pd.DataFrame(rpeaks / ecg_sampling_rate, columns=['R_Peak_Time_s']).to_csv(rpeaks_times_file, index=False)
            p_logger.info(f"R-peak times saved to {rpeaks_times_file}")
        else:
            p_logger.info("Skipping ECG preprocessing as no ECG stream was loaded/identified.")

        # --- EDA Preprocessing (NeuroKit2) ---
        if eda_signal_raw is not None and eda_sampling_rate is not None:
            p_logger.info("Starting EDA processing...")
            eda_signals, info = nk.eda_process(eda_signal_raw, sampling_rate=sampling_rate_eda)
            phasic_eda = eda_signals['EDA_Phasic']
            tonic_eda = eda_signals['EDA_Tonic']
            p_logger.info("Decomposed EDA into tonic and phasic components.")
            phasic_eda_file = os.path.join(processed_data_output_dir, f"{participant_id}_phasic_eda.csv")
            # Ensure phasic_eda is a DataFrame or Series before calling to_csv
            if isinstance(phasic_eda, (pd.Series, pd.DataFrame)):
                phasic_eda.to_csv(phasic_eda_file, index=False, header=True)
            else: # If it's a numpy array
                pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_file, index=False, header=True)
            p_logger.info(f"Phasic EDA extracted and saved to {phasic_eda_file}")
        else:
            p_logger.info("Skipping EDA preprocessing as no EDA stream was loaded/identified.")

        # --- fNIRS Preprocessing (MNE-NIRS) ---
        if fnirs_raw_od is not None: # fnirs_raw_od is the MNE Raw object for optical density from XDF
            p_logger.info("Starting fNIRS preprocessing...")
            p_logger.info("Loading fNIRS OD data into memory...")
            fnirs_raw_od.load_data() # Explicitly load data for the fNIRS OD stream
            p_logger.info(f"Using loaded fNIRS OD data from XDF (SFreq: {fnirs_raw_od.info['sfreq']} Hz).")
            # Ensure it's treated as optical density if it's not already marked as such by read_raw_xdf
            # This might involve checking channel types or manually setting them if necessary.
            # For now, assume fnirs_raw_od is ready for Beer-Lambert or is raw voltages needing optical_density()
            raw_od_for_beer_lambert = fnirs_raw_od # Default assumption
            
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
            raw_haemo = mne_nirs.beer_lambert_law(raw_haemo, ppf=6.0) # Adjust ppf as needed
            p_logger.info("Applied modified Beer-Lambert Law (OD to HbO/HbR).")
            # Motion correction (TDDR)
            corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            p_logger.info("Applied TDDR motion artifact correction.")
            # Filtering
            corrected_haemo.filter(0.01, 0.1, h_trans_bandwidth=0.02, l_trans_bandwidth=0.002, fir_design='firwin')
            p_logger.info("Applied band-pass filter (0.01-0.1 Hz).")
            preprocessed_fnirs_file = os.path.join(processed_data_output_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
            corrected_haemo.save(preprocessed_fnirs_file, overwrite=True)
            p_logger.info(f"fNIRS data preprocessed and saved to {preprocessed_fnirs_file}")
        else:
            p_logger.info("Skipping fNIRS preprocessing as no fNIRS OD stream was loaded/identified.")

        # Dummy file to indicate completion
        with open(os.path.join(processed_data_output_dir, f"{participant_id}_preprocessing_manifest.txt"), "w") as f:
            f.write(f"Preprocessing completed for {participant_id} at {datetime.datetime.now().isoformat()}\n")
            # TODO: List successfully created files here

        p_logger.info(f"Preprocessing for participant {participant_id} completed successfully.")
        return True
    except Exception as e:
        p_logger.error(f"Error during preprocessing for participant {participant_id}: {e}", exc_info=True)
        return False

def run_analysis(participant_id, preprocessed_data_dir, analysis_results_dir, p_logger):
    """
    Participant data analysis based on EV_proposal.tex.
    Saves analysis metrics and reports into `analysis_results_dir`.
    Returns True on success, False on failure.
    """
    p_logger.info(f"Starting analysis for participant {participant_id}...")
    p_logger.info(f"Preprocessed data location: {preprocessed_data_dir}")
    p_logger.info(f"Analysis output directory: {analysis_results_dir}")

    os.makedirs(analysis_results_dir, exist_ok=True)
    analysis_metrics = {'participant_id': participant_id} 

    try:
        if not os.path.exists(preprocessed_data_dir):
            p_logger.error(f"Preprocessed data directory {preprocessed_data_dir} does not exist.")
            return False
        if np is None: # Check if scientific libraries loaded
            p_logger.error("Scientific libraries (NumPy, MNE, etc.) not loaded. Cannot perform analysis.")
            return False

        # --- Load Preprocessed Data ---
        p_logger.info("Loading preprocessed data...")
        eeg_file = os.path.join(preprocessed_data_dir, f"{participant_id}_eeg_preprocessed.fif")
        nn_intervals_file = os.path.join(preprocessed_data_dir, f"{participant_id}_nn_intervals.csv")
        phasic_eda_file = os.path.join(preprocessed_data_dir, f"{participant_id}_phasic_eda.csv")
        fnirs_file = os.path.join(preprocessed_data_dir, f"{participant_id}_fnirs_haemo_preprocessed.fif")
        rpeak_times_file = os.path.join(preprocessed_data_dir, f"{participant_id}_rpeaks_times_sec.csv")

        raw_eeg = mne.io.read_raw_fif(eeg_file, preload=True) if os.path.exists(eeg_file) else None
        nn_intervals_df = pd.read_csv(nn_intervals_file) if os.path.exists(nn_intervals_file) else None
        phasic_eda_df = pd.read_csv(phasic_eda_file) if os.path.exists(phasic_eda_file) else None
        fnirs_haemo = mne.io.read_raw_fif(fnirs_file, preload=True) if os.path.exists(fnirs_file) else None
        p_logger.info("Preprocessed data loading placeholder completed.")

        # --- fNIRS ROI Analysis (GLM based on proposal) ---
        # Used to identify functionally relevant EEG channels for PLV
        p_logger.info("Performing fNIRS ROI analysis...")
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
        p_logger.info("fNIRS ROI analysis placeholder completed.")

        # --- EEG Analysis (Power, Phase for PLV, FAI) ---
        p_logger.info("Performing EEG analysis...")
        if raw_eeg:
            # TODO: Segment EEG into epochs based on events (stimulus onset/offset)
            events_eeg, event_id_eeg = mne.events_from_annotations(raw_eeg) # Assuming events are annotated in raw_eeg
            epochs = mne.Epochs(raw_eeg, events_eeg, event_id=event_id_eeg, tmin=-0.2, tmax=0.8, preload=True) # TODO: Adjust tmax
            # TODO: Calculate PSD for Alpha (8-13 Hz) and Beta (13-30 Hz) for F3, F4, Fp1, Fp2
            # (and potentially channels selected by fNIRS, ensure selected_eeg_channels_for_plv is defined)
            psd_picks = ['F3', 'F4', 'Fp1', 'Fp2']
            if 'selected_eeg_channels_for_plv' in locals() and selected_eeg_channels_for_plv:
                psd_picks = list(set(psd_picks + selected_eeg_channels_for_plv))
            
            psd_params_alpha = dict(fmin=8, fmax=13, method='welch', picks=psd_picks)
            psds_alpha, freqs_alpha = epochs.compute_psd(**psd_params_alpha).get_data(return_freqs=True)
            
            psd_params_beta = dict(fmin=13, fmax=30, method='welch', picks=psd_picks)
            psds_beta, freqs_beta = epochs.compute_psd(**psd_params_beta).get_data(return_freqs=True)
            
            analysis_metrics['alpha_power_f3_mean'] = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :]) # Average over epochs and freqs
            analysis_metrics['beta_power_f3_mean'] = np.mean(psds_beta[:, raw_eeg.ch_names.index('F3'), :])
            # FAI Calculation
            power_right_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :])
            power_left_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
            analysis_metrics['fai_f4_f3_alpha'] = np.log(power_right_alpha_f4 + 1e-9) - np.log(power_left_alpha_f3 + 1e-9) # Add epsilon
            # TODO: Filter EEG in Alpha/Beta for phase extraction (e.g., on continuous data or epochs)
            eeg_alpha_filtered = raw_eeg.copy().filter(8, 13, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            eeg_beta_filtered = raw_eeg.copy().filter(13, 30, picks=selected_eeg_channels_for_plv if 'selected_eeg_channels_for_plv' in locals() else ['Fp1', 'Fp2'])
            phase_eeg_alpha = np.angle(hilbert(eeg_alpha_filtered.get_data())) # Shape (n_channels, n_times)
            phase_eeg_beta = np.angle(hilbert(eeg_beta_filtered.get_data())) # Shape (n_channels, n_times)
        p_logger.info("EEG analysis placeholder completed.")

        # --- HRV Analysis (RMSSD, Continuous Signal for PLV) ---
        p_logger.info("Performing HRV analysis...")
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
                    phase_hrv = np.angle(hilbert(continuous_hrv_signal - np.mean(continuous_hrv_signal))) # Demeaned
        p_logger.info("HRV analysis placeholder completed.")

        # --- EDA Analysis (Resample, Phase for PLV) ---
        p_logger.info("Performing EDA analysis for PLV...")
        if phasic_eda_df is not None and not phasic_eda_df.empty:
            # Assuming original EDA sampling rate was used in NeuroKit2 (e.g., 1000 Hz)
            original_eda_sampling_rate = 1000 # TODO: Get actual original EDA sampling rate from data or config
            phasic_signal = phasic_eda_df['EDA_Phasic'].values
            original_eda_time = np.arange(len(phasic_signal)) / original_eda_sampling_rate
            interp_func_eda = interp1d(original_eda_time, phasic_signal, kind='linear', fill_value="extrapolate")
            # TODO: Determine target_time_eda based on EEG/experiment timing for alignment
            target_time_eda = np.arange(original_eda_time[0], original_eda_time[-1], 1/4.0) # Resample to 4 Hz
            resampled_phasic_eda = interp_func_eda(target_time_eda)
            phase_eda = np.angle(hilbert(resampled_phasic_eda - np.mean(resampled_phasic_eda))) # Demeaned
        p_logger.info("EDA analysis for PLV placeholder completed.")

        # --- PLV Calculation ---
        p_logger.info("Calculating PLV...")
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
        p_logger.info("PLV calculation placeholder completed.")

        # --- FAI Calculation (WP4) ---
        p_logger.info("Calculating FAI...")
        if 'psds_alpha' in locals() and raw_eeg:
            # Calculate FAI for relevant pairs (e.g., F4/F3, Fp2/Fp1)
            # Ensure channels exist in raw_eeg.ch_names
            if 'F4' in raw_eeg.ch_names and 'F3' in raw_eeg.ch_names:
                power_alpha_f4 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F4'), :]) # Average over epochs and freqs
                power_alpha_f3 = np.mean(psds_alpha[:, raw_eeg.ch_names.index('F3'), :])
                # Add a small epsilon to avoid log(0) if power is exactly zero
                analysis_metrics['fai_f4_f3_alpha'] = np.log(power_alpha_f4 + 1e-9) - np.log(power_alpha_f3 + 1e-9)
        p_logger.info("FAI calculation placeholder completed.")

        # Save all calculated metrics
        metrics_df = pd.DataFrame([analysis_metrics])
        metrics_output_file = os.path.join(analysis_results_dir, f"{participant_id}_analysis_metrics.csv")
        metrics_df.to_csv(metrics_output_file, index=False)
        p_logger.info(f"Analysis metrics saved to {metrics_output_file}")

        # --- Statistical Testing (WP1, WP2, WP3, WP4) ---
        p_logger.info("Performing statistical tests...")
        # TODO: Implement statistical tests using statsmodels or pingouin
        # This typically involves aggregating metrics across participants and running tests
        # on the combined data, not per participant within this loop.
        # The per-participant metrics saved above are the input for the group-level stats.
        # Example placeholder for a correlation (WP2, WP3, WP4):
        if 'subjective_arousal' in analysis_metrics and 'plv_eeg_alpha_hrv_roi_positive' in analysis_metrics:
            # This correlation would be done *after* collecting data from all participants
            # using a DataFrame of all participants' metrics.
            pass # Placeholder
        p_logger.info("Statistical testing placeholder completed. Note: Group-level stats run after all participants.")

        # TODO: Generate analysis reports or visualizations if needed (e.g., using matplotlib, seaborn)

        p_logger.info(f"Analysis for participant {participant_id} completed successfully.")
        return True
    except Exception as e:
        p_logger.error(f"Error during analysis for participant {participant_id}: {e}", exc_info=True)
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
        p_logger.info(f"--- Starting processing for participant: {participant_id} ---")
        participant_preprocessed_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "preprocessed")
        participant_analysis_dir = os.path.join(RESULTS_BASE_DIR, participant_id, "analysis")

        # Step 0: Parse Questionnaire Data (if exists)
        p_logger.info("Step 0: Parsing Questionnaire Data...")
        questionnaire_data = parse_questionnaire_data(participant_id, participant_raw_data_path, p_logger)
        # This data will be aggregated later in run_pipeline()

        # Step 1: Preprocessing
        p_logger.info("Step 1: Running Preprocessing...")
        if not os.path.isdir(participant_raw_data_path):
            p_logger.error(f"Raw data directory not found for participant {participant_id}: {participant_raw_data_path}")
            return False, questionnaire_data # Return questionnaire data even if other steps fail
        if not run_preprocessing(participant_id, participant_raw_data_path, participant_preprocessed_dir, p_logger):
            p_logger.error(f"Preprocessing failed for participant {participant_id}.")
            return False, questionnaire_data

        # Step 2: Analysis
        p_logger.info("Step 2: Running Analysis...")
        if not run_analysis(participant_id, participant_preprocessed_dir, participant_analysis_dir, p_logger):
            p_logger.error(f"Analysis failed for participant {participant_id}.")
            return False, questionnaire_data

        # Step 3: Commit results to Git
        p_logger.info("Step 3: Committing results...")
        commit_message = f"Pipeline: Processed data for participant {participant_id}"
        participant_log_file_actual_path = p_logger.handlers[0].baseFilename
        
        # Add the entire results directory for the participant (preprocessed + analysis)
        # and the participant's log file.
        files_to_add_to_git = [
            os.path.relpath(os.path.join(RESULTS_BASE_DIR, participant_id), BASE_REPO_PATH).replace("\\", "/"),
            os.path.relpath(participant_log_file_actual_path, BASE_REPO_PATH).replace("\\", "/")
        ]
        
        if not git_add_commit_push(BASE_REPO_PATH, commit_message, files_to_add_to_git, p_logger):
            p_logger.error(f"Failed to commit and push results for participant {participant_id}.")
            return False, questionnaire_data

        p_logger.info(f"Successfully processed and committed results for participant {participant_id}.")
        success = True
    except Exception as e:
        p_logger.error(f"An unhandled error occurred during processing of participant {participant_id}: {e}", exc_info=True)
        success = False
    finally:
        if success:
            p_logger.info(f"--- Finished processing participant: {participant_id} SUCCESSFULLY ---")
        else:
            p_logger.error(f"--- Finished processing participant: {participant_id} WITH ERRORS ---")
        close_participant_logger(participant_id)
    return success, questionnaire_data

def run_pipeline():
    """Main pipeline execution function."""
    main_logger.info(f"===== Automated EmotiView Analysis Pipeline Started (Continuous Monitoring Mode on {PARTICIPANT_DATA_BASE_DIR}) =====")

    main_logger.info("Attempting to pull latest changes from the repository...")
    if not git_pull(BASE_REPO_PATH, main_logger):
        main_logger.warning("Failed to pull latest changes. Continuing with local version, but repo might be outdated.")

    processed_participant_ids_in_session = set() # Keep track of processed participants in this run

    # Initial scan of RESULTS_BASE_DIR to identify already processed participants
    # This helps avoid reprocessing if the script restarts.
    main_logger.info(f"Scanning {RESULTS_BASE_DIR} for previously processed participants...")
    try:
        for item_name in os.listdir(RESULTS_BASE_DIR):
            item_path = os.path.join(RESULTS_BASE_DIR, item_name)
            if os.path.isdir(item_path):
                # Check for a completion marker, e.g., the analysis metrics file or participant log
                # For simplicity, if the directory exists, we assume it was processed or attempted.
                # A more robust check would be for a specific output file.
                expected_metrics_file = os.path.join(item_path, "analysis", f"{item_name}_analysis_metrics.csv")
                if os.path.exists(expected_metrics_file):
                    main_logger.info(f"Participant {item_name} appears to be processed (results found). Adding to skip list for this session.")
                    processed_participant_ids_in_session.add(item_name)
    except Exception as e:
        main_logger.error(f"Error during initial scan of results directory: {e}", exc_info=True)

    all_questionnaire_data_collected = []

    try:
        while True:
            main_logger.info(f"Scanning {PARTICIPANT_DATA_BASE_DIR} for new XDF files...")
            found_new_xdf_to_process = False
            
            if not os.path.isdir(PARTICIPANT_DATA_BASE_DIR):
                main_logger.error(f"CRITICAL: Monitored directory {PARTICIPANT_DATA_BASE_DIR} does not exist. Waiting...")
                time.sleep(MONITOR_INTERVAL_SECONDS)
                continue

            for item_name in os.listdir(PARTICIPANT_DATA_BASE_DIR):
                item_path = os.path.join(PARTICIPANT_DATA_BASE_DIR, item_name)
                if item_name.lower().endswith(".xdf") and os.path.isfile(item_path):
                    participant_id = os.path.splitext(item_name)[0] # Use filename (no ext) as ID

                    if participant_id in processed_participant_ids_in_session:
                        main_logger.debug(f"XDF for participant {participant_id} already processed or marked for processing in this session. Skipping {item_name}.")
                        continue

                    main_logger.info(f"New XDF file found: {item_name} (Participant ID: {participant_id})")
                    found_new_xdf_to_process = True
                    processed_participant_ids_in_session.add(participant_id) # Mark for this session

                    # Create participant-specific subdirectory and move the XDF file
                    participant_specific_raw_dir = os.path.join(PARTICIPANT_DATA_BASE_DIR, participant_id)
                    destination_xdf_path = os.path.join(participant_specific_raw_dir, item_name) # Keep original filename

                    try:
                        os.makedirs(participant_specific_raw_dir, exist_ok=True)
                        shutil.move(item_path, destination_xdf_path)
                        main_logger.info(f"Moved {item_path} to {destination_xdf_path}")

                        # TODO: Handle associated questionnaire.txt file.
                        # If questionnaire.txt is expected to be named e.g. {participant_id}_questionnaire.txt
                        # in PARTICIPANT_DATA_BASE_DIR, move it too.
                        # For now, assumes questionnaire.txt will be manually placed in participant_specific_raw_dir if needed.

                        main_logger.info(f"--- Starting pipeline for newly discovered participant: {participant_id} ---")
                        overall_success, q_data = process_participant(participant_id)

                        if q_data:
                            all_questionnaire_data_collected.append(q_data)
                            # Consider saving/committing aggregated questionnaire data more frequently in continuous mode

                        main_logger.info(f"--- Finished pipeline for participant: {participant_id} ---")

                    except Exception as e:
                        main_logger.error(f"Error organizing or processing data for {participant_id} from {item_name}: {e}", exc_info=True)
                        # If moving or initial setup failed, remove from session processed list to allow retry
                        if participant_id in processed_participant_ids_in_session:
                            processed_participant_ids_in_session.remove(participant_id)
                        # Optionally, move the XDF file back to the root if it was partially moved or handle error state
                        if os.path.exists(destination_xdf_path) and not os.path.exists(item_path): # If move happened but processing failed
                            main_logger.info(f"Attempting to move {destination_xdf_path} back to {item_path} due to error.")
                            try: shutil.move(destination_xdf_path, item_path)
                            except Exception as move_back_err: main_logger.error(f"Could not move file back: {move_back_err}")
            
            if not found_new_xdf_to_process:
                main_logger.info(f"No new XDF files found in the root of {PARTICIPANT_DATA_BASE_DIR}. Waiting for {MONITOR_INTERVAL_SECONDS}s...")
            
            time.sleep(MONITOR_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        main_logger.info("KeyboardInterrupt received. Shutting down continuous monitoring.")
    finally:
        main_logger.info("Finalizing questionnaire data aggregation before exit...")
        if all_questionnaire_data_collected:
            questionnaire_df = pd.DataFrame(all_questionnaire_data_collected)
            excel_output_path = os.path.join(RESULTS_BASE_DIR, AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME)
            try:
                # Consider appending to existing Excel or merging, rather than overwriting, in a long-running scenario.
                questionnaire_df.to_excel(excel_output_path, index=False)
                main_logger.info(f"Aggregated questionnaire data for this session saved to: {excel_output_path}")
                q_commit_message = "Pipeline: Update aggregated questionnaire data (session end)"
                q_files_to_add = [os.path.relpath(excel_output_path, BASE_REPO_PATH).replace("\\", "/")]
                git_add_commit_push(BASE_REPO_PATH, q_commit_message, q_files_to_add, main_logger)
            except Exception as e:
                main_logger.error(f"Failed to save or commit aggregated questionnaire Excel at exit: {e}", exc_info=True)
        else:
            main_logger.info("No new questionnaire data collected in this session to aggregate.")
        main_logger.info("===== Automated EmotiView Analysis Pipeline Shut Down =====")

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