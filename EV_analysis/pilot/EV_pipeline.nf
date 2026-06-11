#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// =========================================
// EmotiView Nextflow Pipeline
// =========================================

// Import all modules from EV_modules.nf

include { 
	participant_discovery; finalize_participant; finalize_l2;
    txt_reader; xdf_reader; extracting_processor;
    eda_rejection_processor; hrv_rejection_processor;
    eda_filtering_processor; ecg_filtering_processor; eeg_filtering_processor; fnirs_filtering_processor; peak_detection_processor; referencing_processor; log_transform_processor; tddr_processor; linear_transform_processor; events_processor; tree_processor; eda_epoching_processor; eda_epoching_processor as eda_windowing_processor; ecg_epoching_processor; ecg_epoching_processor as ecg_windowing_processor; eeg_epoching_processor; eeg_epoching_processor as eeg_windowing_processor; fnirs_epoching_processor;
    ic_analyzer; interval_analyzer; amplitude_analyzer; fnirs_hbc1_bootstrap_analyzer; fnirs_hbc2_bootstrap_analyzer; fnirs_hbc3_bootstrap_analyzer; psd_analyzer; psd_fai_analyzer; eeg_roi_group_analyzer; eeg_roi_psd1_analyzer; eeg_roi_psd2_analyzer; eeg_roi_psd3_analyzer; eeg_spectrum_analyzer; group_analyzer; bootstrap_analyzer as eda_bootstrap_analyzer; bootstrap_analyzer as hrv_bootstrap_analyzer; eeg_alpha_bootstrap_analyzer; eeg_beta_bootstrap_analyzer; eeg_theta_bootstrap_analyzer; eeg_parietal_alpha_bootstrap_analyzer; eeg_parietal_beta_bootstrap_analyzer; eeg_parietal_theta_bootstrap_analyzer; fai1_analyzer; fai2_analyzer; fai3_analyzer; panas_analyzer; bisbas_analyzer; sam_analyzer; be7_analyzer; ea11_analyzer;
    sam_concatenating_processor; be7_concatenating_processor; ea11_concatenating_processor; hrv_concatenating_processor; eda_concatenating_processor; eeg_psd_concatenating_processor; fai_concatenating_processor; fnirs_concatenating_processor; fnirs_posneg_concatenating_processor; eeg_alpha_concatenating_processor; eeg_beta_concatenating_processor; eeg_theta_concatenating_processor; eeg_psd_result_concatenating_processor; eeg_parietal_alpha_concatenating_processor; eeg_parietal_beta_concatenating_processor; eeg_parietal_theta_concatenating_processor; eeg_parietal_psd_result_concatenating_processor; eeg_spectrum_concatenating_processor; eda_amp_concatenating_processor; hrv_interv_concatenating_processor;
    eeg_psd_ols_processor; eeg_psd_contrast_processor;
    fai_merger;
    eeg_psd1_outlier_processor; eeg_psd2_outlier_processor; eeg_psd3_outlier_processor;
    panas_result_collector; bisbas_result_collector; sam_result_collector; be7_result_collector; ea11_result_collector; hrv_result_collector; eda_result_collector; eeg_psd_result_collector; eeg_contrast_result_collector; eeg_fai_result_collector; eeg_psd_parietal_result_collector; eeg_frontal_row_filter; eeg_parietal_row_filter; fnirs_result_collector; fnirs_posneg_result_collector; eeg_spectrum_result_collector; anova_combined_result_collector; multimodal_correl_result_collector;
    fnirs_raw_concatenating_processor; fai1_epoch_analyzer; fai2_epoch_analyzer; fai3_epoch_analyzer; fai_epoch_concatenating_processor; anova_merger; anova_combined_analyzer; multimodal_correl_analyzer;
	aux_file_finder; eda_file_finder; ecg_file_finder; trigger_file_finder; eeg_file_finder; eeg_cleaned_file_finder; fnirs_file_finder; sam1_file_finder; sam2_file_finder; sam3_file_finder; be71_file_finder; be72_file_finder; be73_file_finder; ea111_file_finder; ea112_file_finder; ea113_file_finder; eda1_file_finder; eda2_file_finder; eda3_file_finder; hrv1_file_finder; hrv2_file_finder; hrv3_file_finder; psd1_file_finder; psd2_file_finder; psd3_file_finder; psd1_raw_file_finder; psd2_raw_file_finder; psd3_raw_file_finder; eeg_roi1_file_finder; eeg_roi2_file_finder; eeg_roi3_file_finder; eeg_roi_psd1_epoch_file_finder; eeg_roi_psd2_epoch_file_finder; eeg_roi_psd3_epoch_file_finder; eeg_spectrum1_file_finder; eeg_spectrum2_file_finder; eeg_spectrum3_file_finder; eeg_alpha_bs1_file_finder; eeg_alpha_bs2_file_finder; eeg_alpha_bs3_file_finder; eeg_beta_bs1_file_finder; eeg_beta_bs2_file_finder; eeg_beta_bs3_file_finder; eeg_theta_bs1_file_finder; eeg_theta_bs2_file_finder; eeg_theta_bs3_file_finder; eeg_parietal_alpha_bs1_file_finder; eeg_parietal_alpha_bs2_file_finder; eeg_parietal_alpha_bs3_file_finder; eeg_parietal_beta_bs1_file_finder; eeg_parietal_beta_bs2_file_finder; eeg_parietal_beta_bs3_file_finder; eeg_parietal_theta_bs1_file_finder; eeg_parietal_theta_bs2_file_finder; eeg_parietal_theta_bs3_file_finder; fnirs1_file_finder; fnirs2_file_finder; fnirs3_file_finder; fnirs_hbc1_agg_file_finder; fnirs_hbc2_agg_file_finder; fnirs_hbc3_agg_file_finder; eda_amp1_file_finder; eda_amp2_file_finder; eda_amp3_file_finder; hrv_interv1_file_finder; hrv_interv2_file_finder; hrv_interv3_file_finder;
	lgcrct_l2_analyzer;
} from './EV_modules.nf'

// ----------- WORKFLOW DEFINITION -----------
workflow {
	// ----------- PARTICIPANT LEVEL ANALYSIS -----------
	// Step 1: Discover participants and create output folders
	participant_discovery(params.input_dir, params.output_dir, params.participant_pattern)
	
	// Get synchronized participant context and split into separate channels
	participant_context = participant_discovery.out.participant_context
	participant_id = participant_context.map { it[0] }
	output_folder = participant_context.map { it[1] }

	// Step 2: Collect raw signals
	// txt_reader produces a parquet with questionnaire/raw text data
	txt_data = txt_reader(params.python_exe, params.txt_reader_script, 
		participant_id.map { id -> "${workflow.launchDir}/${params.input_dir}/${id}/${id}.txt" }, 
		"${params.txt_encoding}")

	// xdf_reader produces per-stream fif/parquet files
	xdf_data = xdf_reader(params.python_exe, params.xdf_reader_script, 
		participant_id.map { id -> "${workflow.launchDir}/${params.input_dir}/${id}/${id}.xdf" }, 
		"")

	// Extract streams from XDF auxilliary channels (EEG stream with aux)
    aux_stream = aux_file_finder(params.python_exe, params.file_finder_script, xdf_data, "type:EEG.fif aux")
	aux_extr = extracting_processor(params.python_exe, params.extracting_processor_script, aux_stream, params.extraction_columns)

	eda_stream     = eda_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr1.parquet eda")
	ecg_stream     = ecg_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr2.parquet ecg")
	trigger_stream = trigger_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr3.parquet trigger")
	eeg_stream     = eeg_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr4.fif eeg")

	// Extract fNIRS stream from XDF
	fnirs_stream = fnirs_file_finder(params.python_exe, params.file_finder_script, xdf_data, "type:NIRS.fif fnirs")

	// Step 3: Modular signal processing chains
	// Events chain: trigger -> events
	tree_struct = tree_processor(params.python_exe, params.tree_processor_script, txt_data, "${params.entry_delim} ${params.depth_delim} ${params.kv_delim}")
	tree_events_inputs = participant_id
	    .join(trigger_stream.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_struct.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	tree_events = events_processor(params.python_exe, params.events_processor_script, tree_events_inputs, "${params.conds}")

	// EDA chain: filter -> artifact rejection
	eda_filtered = eda_filtering_processor(params.python_exe, params.filtering_processor_script, eda_stream, "${params.eda_l_freq} ${params.eda_h_freq} ${params.eda_channel} ${params.sfreq} ${params.ftype}")
	
	// Artifact rejection: remove samples with excessive amplitude/gradient
	eda_rejected = eda_rejection_processor(params.python_exe, params.rejection_processor_script, eda_filtered, "None amplitude ${params.eda_rejection_threshold}")

	// ECG chain: filter -> artifact rejection -> peak detection
	ecg_filtered = ecg_filtering_processor(params.python_exe, params.filtering_processor_script, ecg_stream, "${params.ecg_l_freq} ${params.ecg_h_freq} ${params.ecg_channel} ${params.sfreq} ${params.ftype}")
	
	// Artifact rejection: remove samples with excessive amplitude/gradient
	ecg_rejected = hrv_rejection_processor(params.python_exe, params.rejection_processor_script, ecg_filtered, "None amplitude ${params.ecg_rejection_threshold}")
	ecg_peaks = peak_detection_processor(params.python_exe, params.peak_detection_processor_script, ecg_rejected, "${params.ecg_channel} ${params.sfreq} ecg")
    
	// EEG chain: referencing -> filter -> ICA (optional channel exclusion for artifact-prone/fNIRS positions)
	eeg_reref = referencing_processor(params.python_exe, params.referencing_processor_script, eeg_stream, "${params.eeg_reference}")
	eeg_filtered = eeg_filtering_processor(params.python_exe, params.filtering_processor_script, eeg_reref, "${params.eeg_l_freq} ${params.eeg_h_freq}")
    ic_analyzed = ic_analyzer(params.python_exe, params.ic_analyzer_script, eeg_filtered, "0.99 None ${params.eeg_ica_exclude ?: '[]'} ${params.eeg_ica_interpolate ?: '[]'}")
	eeg_cleaned = eeg_cleaned_file_finder(params.python_exe, params.file_finder_script, ic_analyzed, "*ica.fif eeg_cleaned")

	// fNIRS chain: log transform -> TDDR robust correction -> regression (short-channel) -> linear unmixing (MBLL) -> filter
	fnirs_log = log_transform_processor(params.python_exe, params.log_transform_script, fnirs_stream, "${params.fnirs_baseline_sec}")
	fnirs_tddr = tddr_processor(params.python_exe, params.tddr_processor_script, fnirs_log, "")
	// Short channels excluded via fnirs_channels param (long-channel pairs only); no regression step needed
	fnirs_hbo = linear_transform_processor(params.python_exe, params.linear_transform_script, fnirs_tddr, "mbll ${params.fnirs_ppf} ${params.fnirs_channels}")
	fnirs_filtered = fnirs_filtering_processor(params.python_exe, params.filtering_processor_script, fnirs_hbo, "${params.fnirs_l_freq} ${params.fnirs_h_freq}")

	// Step 4: Epoching - use join to match data with events by participant ID
	eda_epoch_inputs = participant_id
	    .join(eda_rejected.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_events.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	eda_epoched = eda_epoching_processor(params.python_exe, params.epoching_processor_script, eda_epoch_inputs, "")

	ecg_epoch_inputs = participant_id
	    .join(ecg_peaks.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_events.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	ecg_epoched = ecg_epoching_processor(params.python_exe, params.epoching_processor_script, ecg_epoch_inputs, "")

	eeg_epoch_inputs = participant_id
	    .join(eeg_cleaned.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_events.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	eeg_epoched = eeg_epoching_processor(params.python_exe, params.epoching_processor_script, eeg_epoch_inputs, "")
	// Sliding-window sub-epochs for PSD (increases ANOVA df from ~6 to ~80+ per condition)
	eeg_windowed = eeg_windowing_processor(params.python_exe, params.epoching_processor_script, eeg_epoched, "sliding ${params.bootstrap_window_size} ${params.bootstrap_step_size}")

	fnirs_epoch_inputs = participant_id
	    .join(fnirs_filtered.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_events.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	fnirs_epoched = fnirs_epoching_processor(params.python_exe, params.epoching_processor_script, fnirs_epoch_inputs, "")

	// Step 5: Analysis
	// Questionnaire analysis
	panas_analyzed = panas_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.panas_param} panas")
	bisbas_analyzed = bisbas_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.bisbas_param} bisbas")

	// Single-item questionnaires that get analyzed per condition
	sam_analyzed = sam_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.sam_param} sam")
	sam_files = participant_id
		.join(sam1_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam1.parquet sam").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(sam2_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam2.parquet sam").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(sam3_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam3.parquet sam").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	sam_concat = sam_concatenating_processor(params.python_exe, params.concatenating_processor_script, sam_files, "sam terminal")

	be7_analyzed = be7_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.be7_param} be7")
	be7_files = participant_id
		.join(be71_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be71.parquet be7").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(be72_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be72.parquet be7").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(be73_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be73.parquet be7").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	be7_concat = be7_concatenating_processor(params.python_exe, params.concatenating_processor_script, be7_files, "be7 terminal")
	
	ea11_analyzed = ea11_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.ea11_param} ea11")
	ea11_files = participant_id
		.join(ea111_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea111.parquet ea11").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(ea112_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea112.parquet ea11").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(ea113_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea113.parquet ea11").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	ea11_concat = ea11_concatenating_processor(params.python_exe, params.concatenating_processor_script, ea11_files, "ea11 terminal")
	
	// Physiological analyses (following test analysis procedural tree)
	// EDA analysis: amplitude (peak-baseline per trial) -> bootstrap for within-participant CIs
	// No windowing: EDA reacts slowly so sub-windows would produce highly autocorrelated samples.
	// One amplitude value per trial gives 9 honest observations (3 conditions × 3 trials).
	eda_amplitude = amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, eda_epoched, "peak_baseline None 'Conductance Change (uS)'")
	eda_analyzed = eda_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eda_amplitude, "condition epoch_id value ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.eda_y_lim} 'Conductance Change (uS)'")
	eda_files = participant_id
		.join(eda1_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*bs1.parquet eda").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda2_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*bs2.parquet eda").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda3_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*bs3.parquet eda").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eda_concat = eda_concatenating_processor(params.python_exe, params.concatenating_processor_script, eda_files, "eda terminal")
	
	// HRV analysis on ECG epochs: windowing -> RMSSD per sub-window -> bootstrap for robust within-participant CIs
	// Windowing on the epoched peak series FIRST so RMSSD is computed per sub-window (more bootstrap samples)
	hrv_windowed = ecg_windowing_processor(params.python_exe, params.epoching_processor_script, ecg_epoched, "sliding ${params.bootstrap_window_size} ${params.bootstrap_step_size}")
	hrv_intervals = interval_analyzer(params.python_exe, params.interval_analyzer_script, hrv_windowed, "peak_sample None 'Value (ms)' RMSSD")
	hrv_analyzed = hrv_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, hrv_intervals, "condition epoch_id value ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.hrv_y_lim} 'RMSSD (ms)'")
	hrv_files = participant_id
		.join(hrv1_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*bs1.parquet hrv").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(hrv2_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*bs2.parquet hrv").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(hrv3_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*bs3.parquet hrv").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	hrv_concat = hrv_concatenating_processor(params.python_exe, params.concatenating_processor_script, hrv_files, "hrv terminal")
	
	// EEG analyses: explicit ROI-based PSD chain
	// (Step 1 = preprocessing complete above: filtered → ICA → epoched)

	// Step 2: ROI PSD — Welch PSD on full epoch time-series, electrodes averaged per ROI per epoch
	// Single call on the multi-condition epoched data; psd_analyzer groups channels into regions internally
	eeg_roi_psd = eeg_roi_psd1_analyzer(params.python_exe, params.psd_analyzer_script, eeg_windowed,
		"${params.bands_config} None ${params.eeg_rois} ${params.psd_y_lim}")

	// Step 3: Per-condition epoch PSD files (sorted alphabetically: NEG=1, NEU=2, POS=3)
	eeg_psd1_epochs = eeg_roi_psd1_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd1.parquet eegpsd")
	eeg_psd2_epochs = eeg_roi_psd2_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd2.parquet eegpsd")
	eeg_psd3_epochs = eeg_roi_psd3_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd3.parquet eegpsd")

	// Step 3b: Frequency power spectrum — per-ROI mean PSD (µV²/Hz), computed directly from epoched EEG
	eeg_spectrum_analyzed = eeg_spectrum_analyzer(params.python_exe, params.spectrum_analyzer_script, eeg_epoched,
		"None ${params.eeg_rois} 45")
	eeg_spectrum1_raw = eeg_spectrum1_file_finder(params.python_exe, params.file_finder_script, eeg_spectrum_analyzed, "*spectrum1.parquet eegspectrum")
	eeg_spectrum2_raw = eeg_spectrum2_file_finder(params.python_exe, params.file_finder_script, eeg_spectrum_analyzed, "*spectrum2.parquet eegspectrum")
	eeg_spectrum3_raw = eeg_spectrum3_file_finder(params.python_exe, params.file_finder_script, eeg_spectrum_analyzed, "*spectrum3.parquet eegspectrum")
	eeg_spectrum_files = participant_id
		.join(eeg_spectrum1_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_spectrum2_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_spectrum3_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_spectrum_concat = eeg_spectrum_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_spectrum_files, "eeg_spectrum terminal")

	// Step 5: Outlier detection — remove epochs with extreme band power per condition
	eeg_psd1_clean = eeg_psd1_outlier_processor(params.python_exe, params.rejection_processor_script, eeg_psd1_epochs,
		"${params.eeg_band_list} zscore ${params.psd_outlier_threshold}")
	eeg_psd2_clean = eeg_psd2_outlier_processor(params.python_exe, params.rejection_processor_script, eeg_psd2_epochs,
		"${params.eeg_band_list} zscore ${params.psd_outlier_threshold}")
	eeg_psd3_clean = eeg_psd3_outlier_processor(params.python_exe, params.rejection_processor_script, eeg_psd3_epochs,
		"${params.eeg_band_list} zscore ${params.psd_outlier_threshold}")

	// Step 6: Concatenate clean condition files — single epoch-level table for follow-up stats (t-test etc.)
	eeg_psd_clean_files = participant_id
		.join(eeg_psd1_clean.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_psd2_clean.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_psd3_clean.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_psd_epochs = eeg_psd_concatenating_processor(params.python_exe, params.concatenating_processor_script,
		eeg_psd_clean_files, "eeg_psd_epochs")

	// Step 6b: Split by ROI before bootstrapping so Frontal and Parietal are independent
	eeg_frontal_epochs = eeg_frontal_row_filter(params.python_exe, params.row_filter_processor_script, eeg_psd_epochs, "region Frontal true")
	eeg_parietal_epochs = eeg_parietal_row_filter(params.python_exe, params.row_filter_processor_script, eeg_psd_epochs, "region Parietal true")

	// Step 7: Bootstrap CIs per condition per frequency band (Frontal ROI)
	eeg_alpha_bootstrapped = eeg_alpha_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_frontal_epochs,
		"condition epoch_id alpha ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Alpha Power (µV²/Hz)' terminal")
	eeg_beta_bootstrapped = eeg_beta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_frontal_epochs,
		"condition epoch_id beta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Beta Power (µV²/Hz)' terminal")
	eeg_theta_bootstrapped = eeg_theta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_frontal_epochs,
		"condition epoch_id theta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Theta Power (µV²/Hz)' terminal")

	// Step 7b: Bootstrap CIs per condition per band (Parietal ROI)
	eeg_parietal_alpha_bootstrapped = eeg_parietal_alpha_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_parietal_epochs,
		"condition epoch_id alpha ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Alpha Power (µV²/Hz)' terminal")
	eeg_parietal_beta_bootstrapped = eeg_parietal_beta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_parietal_epochs,
		"condition epoch_id beta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Beta Power (µV²/Hz)' terminal")
	eeg_parietal_theta_bootstrapped = eeg_parietal_theta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_parietal_epochs,
		"condition epoch_id theta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Theta Power (µV²/Hz)' terminal")

	// Step 8: Within-participant OLS — beta per condition per ROI×band (epochs as observations)
	// Equivalent to a GLM: gives effect size (beta) and standard error per condition per channel
	eeg_psd_ols = eeg_psd_ols_processor(params.python_exe, params.ols_processor_script, eeg_psd_epochs,
		"eeg_psd_ols")

	// Step 9: Pairwise contrasts — weighted sums of betas give t-values for each comparison
	// e.g. POS-NEU tests whether positive > neutral within this participant
	eeg_psd_contrasts = eeg_psd_contrast_processor(params.python_exe, params.contrast_processor_script, eeg_psd_ols,
		"${params.eeg_psd_contrasts} eeg_psd_contrast terminal")

	// FAI (F3–F4 frontal alpha asymmetry): separate channel-mode PSD — needs per-electrode resolution
	psd_fai_analyzed = psd_fai_analyzer(params.python_exe, params.psd_analyzer_script, eeg_windowed,
		"${params.bands_config} None None ${params.psd_y_lim}")

	// Raw PSD files (per-channel data) for FAI asymmetry analysis
	psd1_raw = psd1_raw_file_finder(params.python_exe, params.file_finder_script, psd_fai_analyzed, "*psd1.parquet faipsd")
	psd2_raw = psd2_raw_file_finder(params.python_exe, params.file_finder_script, psd_fai_analyzed, "*psd2.parquet faipsd")
	psd3_raw = psd3_raw_file_finder(params.python_exe, params.file_finder_script, psd_fai_analyzed, "*psd3.parquet faipsd")

	// PSD FAI (asymmetry): process each condition's raw PSD separately, then concatenate
	fai1_analyzed = fai1_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd1_raw,
		"${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None")
	fai2_analyzed = fai2_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd2_raw,
		"${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None")
	fai3_analyzed = fai3_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd3_raw,
		"${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None")
	psd_fai_files = participant_id
		.join(fai1_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fai2_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fai3_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	psd_fai_concat = fai_concatenating_processor(params.python_exe, params.concatenating_processor_script, psd_fai_files, "psd_fai terminal")
	
	// fNIRS: ROI analysis
	fnirs_analyzed = group_analyzer(params.python_exe, params.group_analyzer_script, fnirs_epoched, "${params.fnirs_rois} ${params.fnirs_y_lim} ROI 'Hb Change (uM)' hbc ${params.fnirs_epoch_baseline_sec} ${params.fnirs_chromophores}")
	
	// Get the individual condition files from group_analyzer output
	fnirs_hbc1_raw = fnirs1_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc1.parquet fnirshbc")
	fnirs_hbc2_raw = fnirs2_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc2.parquet fnirshbc")
	fnirs_hbc3_raw = fnirs3_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc3.parquet fnirshbc")
	
	// Bootstrap CIs per ROI per condition (auto_multi bootstraps each ROI column independently)
	fnirs_hbc1_agg = fnirs_hbc1_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, fnirs_hbc1_raw, "condition epoch_id auto_multi ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.fnirs_y_lim} 'Hb Change (uM)'")
	fnirs_hbc2_agg = fnirs_hbc2_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, fnirs_hbc2_raw, "condition epoch_id auto_multi ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.fnirs_y_lim} 'Hb Change (uM)'")
	fnirs_hbc3_agg = fnirs_hbc3_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, fnirs_hbc3_raw, "condition epoch_id auto_multi ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.fnirs_y_lim} 'Hb Change (uM)'")
	
	// Extract actual data files from bootstrap_analyzer signal outputs (pattern: *bs1 = the one condition per run)
	fnirs_hbc1_agg_file = fnirs_hbc1_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc1_agg, "*bs1.parquet fnirsagg")
	fnirs_hbc2_agg_file = fnirs_hbc2_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc2_agg, "*bs1.parquet fnirsagg")
	fnirs_hbc3_agg_file = fnirs_hbc3_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc3_agg, "*bs1.parquet fnirsagg")

	fnirs_files = participant_id
		.join(fnirs_hbc1_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_hbc2_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_hbc3_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	fnirs_concat = fnirs_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_files, "fnirsagg terminal")

	// POS+NEG only: skip NEU (hbc2) to prevent its high variance from obscuring the other conditions
	fnirs_posneg_files = participant_id
		.join(fnirs_hbc1_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_hbc3_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2 -> [f1, f2] }
	fnirs_posneg_concat = fnirs_posneg_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_posneg_files, "fnirsagg terminal")
	
	// PSD: file-find per-condition from each band's bootstrap → per-band concat → cross-band concat
	eeg_alpha_files = participant_id
		.join(eeg_alpha_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_alpha_bootstrapped, "*bs1.parquet eeg_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_alpha_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_alpha_bootstrapped, "*bs2.parquet eeg_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_alpha_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_alpha_bootstrapped, "*bs3.parquet eeg_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_alpha_concat = eeg_alpha_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_alpha_files, "alpha terminal")

	eeg_beta_files = participant_id
		.join(eeg_beta_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_beta_bootstrapped, "*bs1.parquet eeg_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_beta_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_beta_bootstrapped, "*bs2.parquet eeg_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_beta_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_beta_bootstrapped, "*bs3.parquet eeg_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_beta_concat = eeg_beta_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_beta_files, "beta terminal")

	eeg_theta_files = participant_id
		.join(eeg_theta_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_theta_bootstrapped, "*bs1.parquet eeg_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_theta_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_theta_bootstrapped, "*bs2.parquet eeg_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_theta_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_theta_bootstrapped, "*bs3.parquet eeg_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_theta_concat = eeg_theta_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_theta_files, "theta terminal")

	// Cross-band: merge Frontal alpha + beta + theta into grouped bar chart
	eeg_psd_cross_files = participant_id
		.join(eeg_alpha_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_beta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_theta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_psd_concat = eeg_psd_result_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_psd_cross_files, "eeg_psd terminal")

	// Parietal: file-find per-condition from each band's bootstrap → per-band concat → cross-band concat
	eeg_parietal_alpha_files = participant_id
		.join(eeg_parietal_alpha_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_alpha_bootstrapped, "*bs1.parquet eeg_parietal_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_alpha_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_alpha_bootstrapped, "*bs2.parquet eeg_parietal_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_alpha_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_alpha_bootstrapped, "*bs3.parquet eeg_parietal_alpha_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_parietal_alpha_concat = eeg_parietal_alpha_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_parietal_alpha_files, "alpha terminal")

	eeg_parietal_beta_files = participant_id
		.join(eeg_parietal_beta_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_beta_bootstrapped, "*bs1.parquet eeg_parietal_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_beta_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_beta_bootstrapped, "*bs2.parquet eeg_parietal_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_beta_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_beta_bootstrapped, "*bs3.parquet eeg_parietal_beta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_parietal_beta_concat = eeg_parietal_beta_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_parietal_beta_files, "beta terminal")

	eeg_parietal_theta_files = participant_id
		.join(eeg_parietal_theta_bs1_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_theta_bootstrapped, "*bs1.parquet eeg_parietal_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_theta_bs2_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_theta_bootstrapped, "*bs2.parquet eeg_parietal_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_theta_bs3_file_finder(params.python_exe, params.file_finder_script, eeg_parietal_theta_bootstrapped, "*bs3.parquet eeg_parietal_theta_bs").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_parietal_theta_concat = eeg_parietal_theta_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_parietal_theta_files, "theta terminal")

	eeg_parietal_psd_cross_files = participant_id
		.join(eeg_parietal_alpha_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_beta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_parietal_theta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_psd_parietal_concat = eeg_parietal_psd_result_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_parietal_psd_cross_files, "eeg_psd_parietal terminal")

	// Step 5b: Collect final results with clean names into results/ folder
	panas_result = panas_result_collector(params.python_exe, params.result_collector_script, panas_analyzed, "panas result terminal")
	bisbas_result = bisbas_result_collector(params.python_exe, params.result_collector_script, bisbas_analyzed, "bisbas result terminal")
	sam_result = sam_result_collector(params.python_exe, params.result_collector_script, sam_concat, "sam result terminal")
	be7_result = be7_result_collector(params.python_exe, params.result_collector_script, be7_concat, "be7 result terminal")
	ea11_result = ea11_result_collector(params.python_exe, params.result_collector_script, ea11_concat, "ea11 result terminal")
	hrv_result = hrv_result_collector(params.python_exe, params.result_collector_script, hrv_concat, "hrv result terminal")
	eda_result = eda_result_collector(params.python_exe, params.result_collector_script, eda_concat, "eda result terminal")
	eeg_psd_result = eeg_psd_result_collector(params.python_exe, params.result_collector_script, eeg_psd_concat, "eeg_psd result terminal")
	eeg_psd_parietal_result = eeg_psd_parietal_result_collector(params.python_exe, params.result_collector_script, eeg_psd_parietal_concat, "eeg_psd_parietal result terminal")
	eeg_spectrum_result = eeg_spectrum_result_collector(params.python_exe, params.result_collector_script, eeg_spectrum_concat, "eeg_spectrum result terminal")
	eeg_contrast_result = eeg_contrast_result_collector(params.python_exe, params.result_collector_script, eeg_psd_contrasts, "eeg_contrast result terminal")
	eeg_fai_result = eeg_fai_result_collector(params.python_exe, params.result_collector_script, psd_fai_concat, "eeg_fai result terminal")
	fnirs_result = fnirs_result_collector(params.python_exe, params.result_collector_script, fnirs_concat, "fnirs result terminal")
	fnirs_posneg_result = fnirs_posneg_result_collector(params.python_exe, params.result_collector_script, fnirs_posneg_concat, "fnirs_posneg result terminal")

// Step 5c: ANOVA per modality — one-way ANOVA across conditions per DV
    // EDA ANOVA: find per-condition amplitude epoch files → concat → ANOVA
    eda_amp1 = eda_amp1_file_finder(params.python_exe, params.file_finder_script, eda_amplitude, "*amp1.parquet eda_amp")
    eda_amp2 = eda_amp2_file_finder(params.python_exe, params.file_finder_script, eda_amplitude, "*amp2.parquet eda_amp")
    eda_amp3 = eda_amp3_file_finder(params.python_exe, params.file_finder_script, eda_amplitude, "*amp3.parquet eda_amp")
    eda_amp_files = participant_id
            .join(eda_amp1.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(eda_amp2.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(eda_amp3.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, f1, f2, f3 -> [f1, f2, f3] }
    eda_amp_concat = eda_amp_concatenating_processor(params.python_exe, params.concatenating_processor_script, eda_amp_files, "eda_amp terminal")

    // HRV: find per-condition interval epoch files → concat
    hrv_interv1 = hrv_interv1_file_finder(params.python_exe, params.file_finder_script, hrv_intervals, "*interv1.parquet hrv_interv")
    hrv_interv2 = hrv_interv2_file_finder(params.python_exe, params.file_finder_script, hrv_intervals, "*interv2.parquet hrv_interv")
    hrv_interv3 = hrv_interv3_file_finder(params.python_exe, params.file_finder_script, hrv_intervals, "*interv3.parquet hrv_interv")
    hrv_interv_files = participant_id
            .join(hrv_interv1.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(hrv_interv2.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(hrv_interv3.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, f1, f2, f3 -> [f1, f2, f3] }
    hrv_interv_concat = hrv_interv_concatenating_processor(params.python_exe, params.concatenating_processor_script, hrv_interv_files, "hrv_interv terminal")

    // fNIRS: concatenate raw per-condition ROI epoch files
    fnirs_raw_files = participant_id
            .join(fnirs_hbc1_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fnirs_hbc2_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fnirs_hbc3_raw.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, f1, f2, f3 -> [f1, f2, f3] }
    fnirs_raw_concat = fnirs_raw_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_raw_files, "fnirshbc terminal")

    // FAI: compute per-epoch FAI from channel-level PSD files (epoch_output=true)
    fai1_epoch = fai1_epoch_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd1_raw,
        "${params.electrode_pairs} log ${params.fai_band_name} None None true terminal")
    fai2_epoch = fai2_epoch_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd2_raw,
        "${params.electrode_pairs} log ${params.fai_band_name} None None true terminal")
    fai3_epoch = fai3_epoch_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd3_raw,
        "${params.electrode_pairs} log ${params.fai_band_name} None None true terminal")
    fai_epoch_files = participant_id
            .join(fai1_epoch.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fai2_epoch.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fai3_epoch.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, f1, f2, f3 -> [f1, f2, f3] }
    fai_epoch_table = fai_epoch_concatenating_processor(params.python_exe, params.concatenating_processor_script, fai_epoch_files, "fai_epochs terminal")

    // Step 5d: Combine all epoch tables (with injected 'source' label) → single ANOVA
    // concatenating_processor label:path syntax injects a 'source' column; anova_analyzer
    // group_by=source then runs one-way ANOVA per modality group in a single combined table.
    // fNIRS included: fnirs_raw_concat holds epoch-level aggregated ROI data matching other modalities.
    anova_epoch_files = participant_id
            .join(eda_amp_concat.map     { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(hrv_interv_concat.map  { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fnirs_raw_concat.map   { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(eeg_frontal_epochs.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(eeg_parietal_epochs.map{ f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(fai_epoch_table.map    { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, a, b, c, d, e, fai -> [a, b, c, d, e, fai] }
    anova_epochs_concat = anova_merger(params.python_exe, params.concatenating_processor_script, anova_epoch_files, "labels=EDA,HRV,fNIRS,EEG_Frontal,EEG_Parietal,FAI anova_epochs terminal")
    anova_combined = anova_combined_analyzer(params.python_exe, params.anova_analyzer_script, anova_epochs_concat, "auto condition false None source terminal")
    anova_combined_result = anova_combined_result_collector(params.python_exe, params.result_collector_script, anova_combined, "anova_combined result terminal")

    // Step 5e: Multimodal cross-correlation heatmap (EEG frontal + HRV joined on condition-level means)
    multimodal_files = participant_id
            .join(eeg_frontal_epochs.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .join(hrv_interv_concat.map  { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
            .map { pid, eeg, hrv -> [eeg, hrv] }
    multimodal_correl = multimodal_correl_analyzer(params.python_exe, params.correl_analyzer_script, multimodal_files, "None terminal")
    multimodal_correl_result = multimodal_correl_result_collector(params.python_exe, params.result_collector_script, multimodal_correl, "multimodal_correl result terminal")

    // Step 6: Finalize participants — triggers per participant when all 16 result collectors complete.
    // Raw epoch data is auto-exported to tables/ by workflow_wrapper.nf (no explicit wiring needed).
    finalize_participant(
            panas_result.mix(bisbas_result, sam_result, be7_result, ea11_result,
                             hrv_result, eda_result, eeg_psd_result, eeg_psd_parietal_result, eeg_contrast_result,
                             eeg_fai_result, fnirs_result, fnirs_posneg_result, eeg_spectrum_result,
                             anova_combined_result,
                             multimodal_correl_result),
            16,
		participant_context
	)

	// Optional exploratory LGC-RCT branch (cross-subject LOSO) on pilot EEG epochs.
	// Runs only when enabled in EV_parameters.config.
	if (params.enable_lgcrct_branch && params.l2_analyses) {
		lgcrct_l2 = lgcrct_l2_analyzer(params.python_exe, params.lgcrct_loso_script,
			eeg_epoched.collect(),
			"--mode ${params.lgcrct_mode} --targets ${params.lgcrct_targets} --sfreq ${params.sfreq} --window-sec ${params.lgcrct_window_sec} --step-sec ${params.lgcrct_step_sec} --band ${params.lgcrct_band} --threshold ${params.lgcrct_threshold} --use-lgc ${params.lgcrct_use_lgc} --half-window ${params.lgcrct_half_window} --cov-estimator ${params.lgcrct_cov_estimator} --lgc-mean ${params.lgcrct_lgc_mean} group_log result terminal")
		finalize_l2(lgcrct_l2)
	}

}
