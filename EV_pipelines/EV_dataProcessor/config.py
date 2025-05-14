import os

# --- Path Configurations ---
BASE_REPO_PATH = r"D:\repoShaggy\EmotiView" # Adjust if your repo is elsewhere
PARTICIPANT_DATA_BASE_DIR = os.path.join(BASE_REPO_PATH, "rawData", "pilotData")
RESULTS_BASE_DIR = os.path.join(BASE_REPO_PATH, "EV_results")
LOG_DIR = os.path.join(BASE_REPO_PATH, "logs", "pilot_runs") # Specific logs for pilot

# --- Filename Configurations ---
QUESTIONNAIRE_TXT_FILENAME = "questionnaire.txt"
AGGREGATED_QUESTIONNAIRE_EXCEL_FILENAME = "pilot_aggregated_questionnaires.xlsx"
COMBINED_LSL_STREAM_FILENAME_SUFFIX = "_combined_lsl.vhdr" # Assumes BrainVision format from LSL
FNIRS_RAW_FILENAME_SUFFIX = "_fnirs.nirs" # Assumes NIRx format

# --- Data Loading & Channel Configurations ---
# These indices are for the combined LSL stream (e.g., from a BrainVision file)
COMBINED_EDA_CHANNEL_INDEX = 6
COMBINED_ECG_CHANNEL_INDEX = 7
COMBINED_EEG_START_CHANNEL_INDEX = 8

# --- Preprocessing Configurations ---
EEG_LPASS_HZ = 0.5
EEG_HPASS_HZ = 40.0
ICA_N_COMPONENTS = 15
FNIRS_FILTER_LPASS_HZ = 0.01
FNIRS_FILTER_HPASS_HZ = 0.1
FNIRS_BEER_LAMBERT_PPF = 6.0
FNIRS_USE_SHORT_CHANNEL_REGRESSION = True # Set to False if not using or no short channels
FNIRS_SHORT_CHANNEL_MAX_DISTANCE_MM = 15.0 # Max distance for a channel to be considered short by MNE-NIRS default

# --- Analysis Configurations ---
STIMULUS_DURATION_SECONDS = 5.0
PLV_EPOCH_TMIN_RELATIVE_TO_ONSET = 0.0
PLV_EPOCH_TMAX_RELATIVE_TO_ONSET = 5.0
AUTONOMIC_RESAMPLE_SFREQ = 4.0
DEFAULT_EDA_SAMPLING_RATE_HZ = 1000.0 # Fallback if not found in saved metadata

DEFAULT_EEG_CHANNELS_FOR_PLV = ['Fp1', 'Fp2', 'F3', 'F4']
DEFAULT_EEG_CHANNELS_FOR_FAI_PSD = ['Fp1', 'Fp2', 'F3', 'F4', 'AF3', 'AF4'] # Added AF for more options

FNIRS_ROIS = {
    'DLPFC_L': ['S1_D1', 'S1_D2', 'S2_D1', 'S2_D2'],
    'DLPFC_R': ['S5_D7', 'S5_D8', 'S6_D7', 'S6_D8'],
    'VMPFC': ['S3_D1', 'S4_D2', 'S3_D5', 'S4_D6'] # Example, adjust to your actual layout
}

# Mapping fNIRS ROIs to EEG channels for guided PLV
FNIRS_TO_EEG_ROI_MAP = {
    'DLPFC_L': ['Fp1', 'F3', 'AF3'],
    'DLPFC_R': ['Fp2', 'F4', 'AF4'],
    'VMPFC': ['Fpz', 'AFz', 'Fz'] # Example, adjust to your EEG montage
}
FNIRS_ROI_ACTIVATION_THRESHOLD_THETA = 0.05 # Example threshold for mean theta (beta coefficient) to consider an ROI active

# --- Correlation Configurations (WP2 & WP3) ---
# Define pairs of metrics to correlate. Format: (metric1_name, metric2_name, description)
# Metric names should match columns in the aggregated metrics Excel file.
# Example PLV names: 'plv_avg_alpha_DLPFC_L_hrv_Positive', 'plv_avg_beta_VMPFC_eda_Negative'
# Example Questionnaire names: 'BIS_score', 'BAS_drive', 'SAM_valence_Positive'
CORRELATION_PAIRS = [
    # Add your specific correlation pairs here based on your hypotheses
    # Example: ('BIS_score', 'plv_avg_alpha_DLPFC_L_hrv_Positive', 'BIS vs Alpha-HRV PLV (DLPFC_L Pos)'),
    # Example: ('SAM_valence_Positive', 'plv_avg_alpha_DLPFC_L_hrv_Positive', 'SAM Valence (Pos) vs Alpha-HRV PLV (DLPFC_L Pos)'),
    # ...
]

# --- Logging Configuration ---
MAIN_LOG_LEVEL = "INFO"
PARTICIPANT_LOG_LEVEL = "DEBUG"

# --- Plotting Configuration ---
EEG_PSD_FMAX = EEG_HPASS_HZ

# Ensure base directories exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)