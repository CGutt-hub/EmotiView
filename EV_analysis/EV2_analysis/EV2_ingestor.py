import os
import glob
import pickle
import numpy as np
import pandas as pd
import kagglehub

def log_msg(msg):  print(f"[EV2] INFO: {msg}")
def log_warn(msg): print(f"[EV2] WARNING: {msg}")

# Konstanten für die Pipeline
SAMPLING_RATE = 128.0  # DEAP preprocessed Daten sind auf 128 Hz downgesampelt

# 1. Download des Datensatzes über Kagglehub
log_msg("Starte Download des DEAP Datensatzes...")
download_path = kagglehub.dataset_download("manh123df/deap-dataset")

# 2. Basis-Zielordner definieren
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
base_target_dir = os.path.join(script_dir, "EV2_dataset")
os.makedirs(base_target_dir, exist_ok=True)

# 3. Metadaten-Konstruktion aus Vorgabe
DEAP_EEG_CHANNELS = [
    'Fp1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',
    'CP5', 'CP1', 'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',
    'P4',  'P8',  'PO4', 'O2',  'T8',  'CP2', 'CP6', 'C4',
    'FC2', 'FC6', 'F4',  'F8',  'AF4', 'Fp2', 'Fz',  'Cz',
]
EDA_CH = 36   
BVP_CH = 38   
N_EEG  = 32   

# 4. Suche nach den Quelldateien
source_files = glob.glob(os.path.join(download_path, "**", "*s[0-9]*.dat"), recursive=True)
if not source_files:
    source_files = glob.glob(os.path.join(download_path, "**", "*s[0-9]*.pkl"), recursive=True)

if not source_files:
    log_warn("Keine passenden Quelldateien gefunden.")
    exit()

# 5. Modifizierte Hilfsfunktion für Pipeline-Konventionen (time, condition, epoch_id)
def transform_signal_to_df(data_slice, channel_names, n_trials, n_samples):
    records = []
    for trial_idx in range(n_trials):
        trial_data = data_slice[trial_idx, :, :] # Shape: (Kanäle, 8064)
        
        # Basis-DataFrame für das aktuelle Video (Trial) erstellen
        df_trial = pd.DataFrame(trial_data.T, columns=channel_names)
        
        # Pipeline-Metriken berechnen und einfügen
        # sample_index wird durch die Sampling Rate geteilt, um physikalische Sekunden (time) zu erhalten
        df_trial.insert(0, 'time', np.arange(n_samples) / SAMPLING_RATE)
        
        # trial_index wird in zwei identische Spalten aufgeteilt (condition und epoch_id)
        df_trial.insert(0, 'epoch_id', trial_idx)
        df_trial.insert(0, 'condition', trial_idx)
        
        records.append(df_trial)
        
    return pd.concat(records, ignore_index=True)

# 6. Verarbeitung und Export nach der kanonischen Ordnerstruktur
for file_path in source_files:
    base_name = os.path.basename(file_path)
    digits = "".join(filter(str.isdigit, base_name))
    
    if not digits:
        continue
    
    subject_int = int(digits)
    three_digit_id = f"{subject_int:03d}"  # Erzeugt "001", "002" usw.
    
    subject_dir_name = f"EV2_{three_digit_id}"
    subject_folder = os.path.join(base_target_dir, subject_dir_name)
    os.makedirs(subject_folder, exist_ok=True)
    
    try:
        with open(file_path, 'rb') as f:
            deap_data = pickle.load(f, encoding='latin1')
            
        raw_matrix = deap_data['data']      # Shape: (40, 40, 8064)
        labels_matrix = deap_data['labels']  # Shape: (40, 4)
        
        n_trials, _, n_samples = raw_matrix.shape
        
        # A) EEG extrahieren & speichern
        eeg_data = raw_matrix[:, :N_EEG, :]
        df_eeg = transform_signal_to_df(eeg_data, DEAP_EEG_CHANNELS, n_trials, n_samples)
        df_eeg.to_parquet(os.path.join(subject_folder, f"{subject_dir_name}_eeg.parquet"), index=False)
        
        # B) EDA extrahieren & speichern
        eda_data = raw_matrix[:, EDA_CH:EDA_CH+1, :]
        df_eda = transform_signal_to_df(eda_data, ['EDA'], n_trials, n_samples)
        df_eda.to_parquet(os.path.join(subject_folder, f"{subject_dir_name}_eda.parquet"), index=False)
        
        # C) BVP extrahieren & speichern
        bvp_data = raw_matrix[:, BVP_CH:BVP_CH+1, :]
        df_bvp = transform_signal_to_df(bvp_data, ['HRV'], n_trials, n_samples)
        df_bvp.to_parquet(os.path.join(subject_folder, f"{subject_dir_name}_hrv.parquet"), index=False)
        
        # D) LABELS extrahieren & speichern
        # Für das Labels-File spiegeln wir die Konvention ebenfalls, damit Joins einfacher sind
        df_labels = pd.DataFrame(labels_matrix, columns=['valence', 'arousal', 'dominance', 'liking'])
        df_labels.insert(0, 'epoch_id', np.arange(n_trials))
        df_labels.insert(0, 'condition', np.arange(n_trials))
        df_labels.to_parquet(os.path.join(subject_folder, f"{subject_dir_name}_sam.parquet"), index=False)
        
        log_msg(f"Erfolgreich exportiert (128Hz-Zeitachse): {subject_dir_name}/")
        
    except Exception as e:
        log_warn(f"Fehler bei Probanden-Datei {base_name}: {str(e)}")

log_msg("Kanonischer Export mit Pipeline-Metadaten abgeschlossen!")