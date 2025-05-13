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

# --- Analysis Configurations ---
STIMULUS_DURATION_SECONDS = 5.0
PLV_EPOCH_TMIN_RELATIVE_TO_ONSET = 0.0
PLV_EPOCH_TMAX_RELATIVE_TO_ONSET = 5.0
AUTONOMIC_RESAMPLE_SFREQ = 4.0
DEFAULT_EDA_SAMPLING_RATE_HZ = 1000.0 # Fallback if not found in saved metadata

DEFAULT_EEG_CHANNELS_FOR_PLV = ['Fp1', 'Fp2', 'F3', 'F4']
DEFAULT_EEG_CHANNELS_FOR_FAI_PSD = ['Fp1', 'Fp2', 'F3', 'F4']

FNIRS_ROIS = {
    'DLPFC_L': ['S1_D1', 'S1_D2', 'S2_D1', 'S2_D2'],
    'DLPFC_R': ['S5_D7', 'S5_D8', 'S6_D7', 'S6_D8'],
    'VMPFC': ['S3_D1', 'S4_D2']
}

# --- Logging Configuration ---
MAIN_LOG_LEVEL = "INFO"
PARTICIPANT_LOG_LEVEL = "DEBUG"

# --- Plotting Configuration ---
EEG_PSD_FMAX = EEG_HPASS_HZ

# Ensure base directories exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)