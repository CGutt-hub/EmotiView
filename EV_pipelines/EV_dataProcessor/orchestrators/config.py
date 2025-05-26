# --- General Configuration ---
BASE_OUTPUT_DIR = "EV_Processed_Data"
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Data Loading Configuration ---
# Define expected file extensions or naming patterns if needed
# e.g., EEG_FILE_EXTENSION = ".vhdr"
# This can be used by the DataLoader to be more specific.
# For now, assuming generic names like 'eeg', 'fnirs', etc. in filenames.

# --- Preprocessing Configuration ---
# EEG
EEG_FILTER_BAND = (1., 40.) # Bandpass filter for EEG data (Hz)
ICA_N_COMPONENTS = 15       # Number of ICA components
ICA_RANDOM_STATE = 42       # Random state for ICA for reproducibility
ICA_ACCEPT_LABELS = ["brain", "other"] # Labels from mne_icalabel to keep
ICA_REJECT_THRESHOLD = 0.7 # Probability threshold to reject a component if not in accept_labels

# ECG/HRV
ECG_SAMPLING_RATE_DEFAULT = 1000 # Default if not found in data, Hz
ECG_CLEAN_METHOD = 'neurokit'    # Method for cleaning ECG (e.g., 'neurokit', 'biosppy')
ECG_PEAK_DETECTION_METHOD = 'neurokit' # Method for R-peak detection
ECG_FILTER_BAND = (5., 35.) # Bandpass filter for ECG data (Hz) - used if cleaning involves filtering

# EDA
EDA_SAMPLING_RATE_DEFAULT = 1000 # Default if not found in data, Hz
# NeuroKit's eda_process handles decomposition, so specific filter params might be internal to it.
# If custom filtering is needed before nk.eda_process, add params here.

# fNIRS
FNIRS_OD_TO_CONC_METHOD = 'beer_lambert_law' # Method for OD to concentration conversion
FNIRS_FILTER_BAND = (0.01, 0.1) # Bandpass filter for fNIRS data (Hz)
FNIRS_MOTION_CORRECTION_METHOD = 'tddr' # Motion correction method (e.g., 'tddr', 'savgol', None)
FNIRS_MOTION_CORRECTION_PARAMS = {'window': 20, 'polyorder': 3, 'deriv': 2} # Parameters for savgol/tddr
FNIRS_BEER_LAMBERT_PPF = [6.0, 6.0] # Partial pathlength factors for HbO and HbR
FNIRS_SHORT_CHANNEL_REGRESSION = True # Whether to perform short channel regression
# fNIRS GLM related
FNIRS_HRF_MODEL = 'spm' # Hemodynamic Response Function model for GLM
FNIRS_CONTRASTS = { # Define contrasts for fNIRS GLM
    'Emotion_vs_Neutral': {'Positive': 0.5, 'Negative': 0.5, 'Neutral': -1.0},
    'Positive_vs_Neutral': {'Positive': 1.0, 'Neutral': -1.0},
    'Negative_vs_Neutral': {'Negative': 1.0, 'Neutral': -1.0},
    # Add more contrasts as needed
}
FNIRS_ACTIVATION_P_THRESHOLD = 0.05 # p-value threshold for considering an fNIRS channel/ROI active from GLM contrast
# Define your fNIRS ROIs by listing the fNIRS channel names (including chromophore) that belong to each ROI.
# IMPORTANT: Replace these example channel names with your actual fNIRS channel names.
FNIRS_ROIS = {
    'dlPFC_L': ['S1_D1 hbo', 'S1_D2 hbo', 'S2_D1 hbo', 'S2_D3 hbo'], # Dorsolateral Prefrontal Cortex Left
    'dlPFC_R': ['S3_D4 hbo', 'S3_D5 hbo', 'S4_D4 hbo', 'S4_D6 hbo'], # Dorsolateral Prefrontal Cortex Right
    'mPFC': ['S7_D7 hbo', 'S7_D8 hbo', 'S8_D7 hbo', 'S8_D9 hbo'],    # Medial Prefrontal / Frontopolar
    'Parietal_L': ['S5_D10 hbo', 'S5_D11 hbo', 'S6_D10 hbo'],      # Left Parietal
    'Parietal_R': ['S9_D12 hbo', 'S9_D13 hbo', 'S10_D12 hbo'],     # Right Parietal
    # Add HbR channels to these lists if you analyze them separately or combined for ROI definition
    # Example HbR channels: 'S1_D1 hbr', etc.
}
FNIRS_ROI_TO_EEG_CHANNELS_MAP = { # Example mapping, adjust to your setup
    'dlPFC_L': ['F3', 'F7', 'AF3', 'F5'], 'dlPFC_R': ['F4', 'F8', 'AF4', 'F6'],
    'mPFC': ['Fp1', 'Fpz', 'Fp2', 'AFz', 'Fz'], # Medial PFC often maps to midline frontal EEG
    'Parietal_L': ['P3', 'P7', 'CP5', 'TP7'], 'Parietal_R': ['P4', 'P8', 'CP6', 'TP8'],
}


# --- Analysis Configuration ---
# General
ANALYSIS_EPOCH_TIMES = (-0.5, 4.0) # Epoch start and end times relative to event onset (seconds)
ANALYSIS_BASELINE_TIMES = (-0.5, 0.0) # Baseline period for correction relative to event onset (seconds)

# PLV Analysis
PLV_RESAMPLE_SFREQ_AUTONOMIC = 4 # Hz, resampling frequency for continuous autonomic signals for PLV
DEFAULT_EEG_CHANNELS_FOR_PLV = ['F3', 'F4', 'Fp1', 'Fp2', 'C3', 'C4'] # Fallback if fNIRS guidance fails
EEG_CHANNEL_SELECTION_STRATEGY_FOR_PLV = 'mapping' # 'mapping', 'nearest', 'predefined'
# Specific bands/modalities for WP correlations (adjust as needed based on hypotheses)
PLV_PRIMARY_EEG_BAND_FOR_WP1 = 'Alpha' # Band for ANOVA
PLV_PRIMARY_EEG_BAND_FOR_WP2 = 'Alpha' # Band for Arousal correlation
PLV_PRIMARY_EEG_BAND_FOR_WP3 = 'Alpha' # Band for RMSSD correlation (EEG-HRV)
PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV = 'Alpha' # Band for FAI vs EEG-HRV PLV
PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA = 'Alpha' # Band for FAI vs EEG-EDA PLV

PLV_EEG_BANDS = {'Alpha': (8, 13), 'Beta': (13, 30)} # EEG bands for PLV

# FAI Analysis
FAI_ALPHA_BAND = (8, 13) # Hz, alpha band for FAI calculation

# --- Reporting Configuration ---
REPORTING_FIGURE_FORMAT = 'png'
# Ensure this directory exists or is created by PlottingService
REPORTING_BASE_PLOT_DIR = "plots" 
REPORTING_DPI = 300

# --- Event Processing Configuration ---
EVENT_TYPE_MAPPING = {
    # Example: map raw event markers to meaningful condition names
    # 'stim_positive_start': 'Positive',
    # 'stim_negative_start': 'Negative',
    # 'stim_neutral_start': 'Neutral',
    # 'button_press': 'Response'
}
EVENT_DURATION_DEFAULT = 4.0 # Default duration for events if not specified (seconds)
BASELINE_MARKER_START = "Baseline_Start" # Marker name for baseline start
BASELINE_MARKER_END = "Baseline_End"     # Marker name for baseline end
BASELINE_MARKER_START_EPRIME = "Baseline_Start" # Marker name in E-Prime log for baseline start
BASELINE_MARKER_END_EPRIME = "Baseline_End"     # Marker name in E-Prime log for baseline end
BASELINE_DURATION_FALLBACK_SEC = 60.0    # Fallback baseline duration if markers not found

# Specific for Work Packages / Orchestrator logic
FAI_ELECTRODE_PAIRS = [('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8')] # Example, used by PSDAnalyzer
FAI_ELECTRODE_PAIRS_FOR_WP4 = ('F3', 'F4') # Specific pair for WP4 correlation (e.g. F3 vs F4)

STIMULUS_DURATION_SECONDS = 4.0 # Example, used by EDAAnalyzer if it were active
ECG_RPEAK_METHOD = 'neurokit' # Default R-peak detection method for ECGPreprocessor