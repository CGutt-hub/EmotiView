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
    eda_filtering_processor; ecg_filtering_processor; eeg_filtering_processor; fnirs_filtering_processor; peak_detection_processor; referencing_processor; log_transform_processor; tddr_processor; regression_processor; linear_transform_processor; events_processor; tree_processor; eda_epoching_processor; eda_epoching_processor as eda_windowing_processor; ecg_epoching_processor; ecg_epoching_processor as ecg_windowing_processor; eeg_epoching_processor; fnirs_epoching_processor;
    ic_analyzer; interval_analyzer; amplitude_analyzer; fnirs_hbc1_amplitude_analyzer; fnirs_hbc2_amplitude_analyzer; fnirs_hbc3_amplitude_analyzer; psd_analyzer; psd_fai_analyzer; eeg_roi_group_analyzer; eeg_roi_psd1_analyzer; eeg_roi_psd2_analyzer; eeg_roi_psd3_analyzer; group_analyzer; bootstrap_analyzer as eda_bootstrap_analyzer; bootstrap_analyzer as hrv_bootstrap_analyzer; eeg_alpha_bootstrap_analyzer; eeg_beta_bootstrap_analyzer; eeg_theta_bootstrap_analyzer; fai1_analyzer; fai2_analyzer; fai3_analyzer; panas_analyzer; bisbas_analyzer; sam_analyzer; be7_analyzer; ea11_analyzer;
    sam_concatenating_processor; be7_concatenating_processor; ea11_concatenating_processor; hrv_concatenating_processor; eda_concatenating_processor; eeg_psd_concatenating_processor; fai_concatenating_processor; fnirs_concatenating_processor; fnirs_asym_concatenating_processor; physio_concatenating_processor; eeg_alpha_concatenating_processor; eeg_beta_concatenating_processor; eeg_theta_concatenating_processor; eeg_psd_result_concatenating_processor;
    eeg_psd_ols_processor; eeg_psd_contrast_processor;
    eda_ols_processor; hrv_ols_processor; condition_profile_processor; condition_profile_correl_analyzer;
    fai_merger;
    fnirs_asym1_analyzer; fnirs_asym2_analyzer; fnirs_asym3_analyzer;
    eeg_psd1_outlier_processor; eeg_psd2_outlier_processor; eeg_psd3_outlier_processor;
    panas_result_collector; bisbas_result_collector; sam_result_collector; be7_result_collector; ea11_result_collector; hrv_result_collector; eda_result_collector; eeg_psd_result_collector; eeg_contrast_result_collector; eeg_fai_result_collector; fnirs_result_collector; fnirs_fai_result_collector; physio_result_collector;
    condition_profile_result_collector; cross_modal_correl_result_collector;
    aux_file_finder; eda_file_finder; ecg_file_finder; trigger_file_finder; eeg_file_finder; eeg_cleaned_file_finder; fnirs_file_finder; sam1_file_finder; sam2_file_finder; sam3_file_finder; be71_file_finder; be72_file_finder; be73_file_finder; ea111_file_finder; ea112_file_finder; ea113_file_finder; eda1_file_finder; eda2_file_finder; eda3_file_finder; hrv1_file_finder; hrv2_file_finder; hrv3_file_finder; psd1_file_finder; psd2_file_finder; psd3_file_finder; psd1_raw_file_finder; psd2_raw_file_finder; psd3_raw_file_finder; eeg_roi1_file_finder; eeg_roi2_file_finder; eeg_roi3_file_finder; eeg_roi_psd1_epoch_file_finder; eeg_roi_psd2_epoch_file_finder; eeg_roi_psd3_epoch_file_finder; eeg_alpha_bs1_file_finder; eeg_alpha_bs2_file_finder; eeg_alpha_bs3_file_finder; eeg_beta_bs1_file_finder; eeg_beta_bs2_file_finder; eeg_beta_bs3_file_finder; eeg_theta_bs1_file_finder; eeg_theta_bs2_file_finder; eeg_theta_bs3_file_finder; fnirs1_file_finder; fnirs2_file_finder; fnirs3_file_finder; fnirs_asym_input1_file_finder; fnirs_asym_input2_file_finder; fnirs_asym_input3_file_finder; fnirs_hbc1_agg_file_finder; fnirs_hbc2_agg_file_finder; fnirs_hbc3_agg_file_finder;
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
    ic_analyzed = ic_analyzer(params.python_exe, params.ic_analyzer_script, eeg_filtered, params.eeg_ica_exclude ? "0.99 None ${params.eeg_ica_exclude}" : "")
	eeg_cleaned = eeg_cleaned_file_finder(params.python_exe, params.file_finder_script, ic_analyzed, "*ica.fif eeg_cleaned")

	// fNIRS chain: log transform -> TDDR robust correction -> regression (short-channel) -> linear unmixing (MBLL) -> filter
	fnirs_log = log_transform_processor(params.python_exe, params.log_transform_script, fnirs_stream, "${params.fnirs_baseline_sec}")
	fnirs_tddr = tddr_processor(params.python_exe, params.tddr_processor_script, fnirs_log, "")
	fnirs_regr = regression_processor(params.python_exe, params.regression_processor_script, fnirs_tddr, "short_channel")
	fnirs_hbo = linear_transform_processor(params.python_exe, params.linear_transform_script, fnirs_regr, "mbll ${params.fnirs_ppf} ${params.fnirs_channels}")
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
	// EDA analysis: windowing -> amplitude (peak-baseline per sub-window) -> bootstrap for robust within-participant CIs
	// Windowing runs on the time-series epochs FIRST to create sub-epochs, then amplitude is extracted per sub-window
	eda_windowed = eda_windowing_processor(params.python_exe, params.epoching_processor_script, eda_epoched, "sliding ${params.bootstrap_window_size} ${params.bootstrap_step_size}")
	eda_amplitude = amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, eda_windowed, "peak_baseline None 'Conductance Change (uS)'")
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
	eeg_roi_psd = eeg_roi_psd1_analyzer(params.python_exe, params.psd_analyzer_script, eeg_epoched,
		"${params.bands_config} None ${params.eeg_rois} ${params.psd_y_lim}")

	// Step 3: Per-condition epoch PSD files (sorted alphabetically: NEG=1, NEU=2, POS=3)
	eeg_psd1_epochs = eeg_roi_psd1_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd1.parquet eegpsd")
	eeg_psd2_epochs = eeg_roi_psd2_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd2.parquet eegpsd")
	eeg_psd3_epochs = eeg_roi_psd3_epoch_file_finder(params.python_exe, params.file_finder_script, eeg_roi_psd, "*psd3.parquet eegpsd")

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

	// Step 7: Bootstrap CIs per condition per frequency band
	eeg_alpha_bootstrapped = eeg_alpha_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_psd_epochs,
		"condition epoch_id alpha ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Alpha Power (µV²/Hz)' terminal")
	eeg_beta_bootstrapped = eeg_beta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_psd_epochs,
		"condition epoch_id beta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Beta Power (µV²/Hz)' terminal")
	eeg_theta_bootstrapped = eeg_theta_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script, eeg_psd_epochs,
		"condition epoch_id theta ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} ${params.psd_y_lim} 'Theta Power (µV²/Hz)' terminal")

	// Step 8: Within-participant OLS — beta per condition per ROI×band (epochs as observations)
	// Equivalent to a GLM: gives effect size (beta) and standard error per condition per channel
	eeg_psd_ols = eeg_psd_ols_processor(params.python_exe, params.ols_processor_script, eeg_psd_epochs,
		"eeg_psd_ols")

	// Step 9: Pairwise contrasts — weighted sums of betas give t-values for each comparison
	// e.g. POS-NEU tests whether positive > neutral within this participant
	eeg_psd_contrasts = eeg_psd_contrast_processor(params.python_exe, params.contrast_processor_script, eeg_psd_ols,
		"${params.eeg_psd_contrasts} eeg_psd_contrast terminal")

	// Step 10: Cross-modal consistency — OLS condition betas for EDA and HRV
	// Using pre-bootstrap epoch-level data so betas reflect the same epoch granularity as EEG
	// eda_amplitude: condition + epoch_id + value (SCR amplitude per sliding sub-window)
	// hrv_intervals: condition + epoch_id + value (RMSSD per sliding sub-window)
	eda_ols = eda_ols_processor(params.python_exe, params.ols_processor_script, eda_amplitude,
		"eda_ols")
	hrv_ols = hrv_ols_processor(params.python_exe, params.ols_processor_script, hrv_intervals,
		"hrv_ols")

	// Step 11: Pivot all three OLS outputs wide (1 row per condition, 1 col per modality×channel)
	// eeg_psd_ols has Frontal_alpha, Parietal_alpha, Frontal_beta... as channels
	// eda_ols / hrv_ols each have a single 'value' channel — renamed to eda_value / hrv_value
	cross_modal_files = participant_id
		.join(eeg_psd_ols.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda_ols.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(hrv_ols.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	cross_modal_wide = condition_profile_processor(params.python_exe, params.condition_profile_processor_script,
		cross_modal_files, "eeg eda hrv condition_profile group_log")

	// Step 12: Pairwise Pearson r across all modality signals (3 conditions as observations)
	// Interpretation: high |r| between eda_value and eeg_Frontal_alpha means the condition
	// that activates the sympathetic system also suppresses frontal alpha — an integration signal
	cross_modal_correl = condition_profile_correl_analyzer(params.python_exe, params.correl_analyzer_script,
		cross_modal_wide, "None group_log")

	// FAI (F3–F4 frontal alpha asymmetry): separate channel-mode PSD — needs per-electrode resolution
	psd_fai_analyzed = psd_fai_analyzer(params.python_exe, params.psd_analyzer_script, eeg_epoched,
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
	
	// Aggregate ROI epochs for plotting (mean ± SEM per condition per region)
	fnirs_hbc1_agg = fnirs_hbc1_amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, fnirs_hbc1_raw, "mean ${params.fnirs_y_lim} 'Hb Change (uM)'")
	fnirs_hbc2_agg = fnirs_hbc2_amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, fnirs_hbc2_raw, "mean ${params.fnirs_y_lim} 'Hb Change (uM)'")
	fnirs_hbc3_agg = fnirs_hbc3_amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, fnirs_hbc3_raw, "mean ${params.fnirs_y_lim} 'Hb Change (uM)'")
	
	// Extract actual data files from amplitude_analyzer signal outputs (pattern: *amp1 = the one condition per run)
	fnirs_hbc1_agg_file = fnirs_hbc1_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc1_agg, "*amp1.parquet fnirsagg")
	fnirs_hbc2_agg_file = fnirs_hbc2_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc2_agg, "*amp1.parquet fnirsagg")
	fnirs_hbc3_agg_file = fnirs_hbc3_agg_file_finder(params.python_exe, params.file_finder_script, fnirs_hbc3_agg, "*amp1.parquet fnirsagg")

	fnirs_files = participant_id
		.join(fnirs_hbc1_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_hbc2_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_hbc3_agg_file.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	fnirs_concat = fnirs_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_files, "fnirsagg terminal")
	
	// fNIRS FAI: Compute R-L hemispheric asymmetry from aggregated ROI data
	fnirs_fai1 = fnirs_asym1_analyzer(params.python_exe, params.asymmetry_analyzer_script, fnirs_hbc1_agg,
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)'")
	fnirs_fai2 = fnirs_asym2_analyzer(params.python_exe, params.asymmetry_analyzer_script, fnirs_hbc2_agg,
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)'")
	fnirs_fai3 = fnirs_asym3_analyzer(params.python_exe, params.asymmetry_analyzer_script, fnirs_hbc3_agg,
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)'")
	fnirs_fai_files = participant_id
		.join(fnirs_fai1.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_fai2.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_fai3.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	fnirs_fai_concat = fnirs_asym_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_fai_files, "hbc_fai terminal")

	// Combined physio: HRV + EDA merged (conditions as x-ticks, modality as series)
	physio_files = participant_id
		.join(hrv_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2 -> [f1, f2] }
	physio_concat = physio_concatenating_processor(params.python_exe, params.concatenating_processor_script, physio_files, "physio terminal")

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

	// Cross-band: merge alpha + beta + theta into grouped bar chart (conditions × bands)
	eeg_psd_cross_files = participant_id
		.join(eeg_alpha_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_beta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eeg_theta_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eeg_psd_concat = eeg_psd_result_concatenating_processor(params.python_exe, params.concatenating_processor_script, eeg_psd_cross_files, "eeg_psd terminal")

	// Step 5b: Collect final results with clean names into results/ folder
	panas_result = panas_result_collector(params.python_exe, params.result_collector_script, panas_analyzed, "panas result terminal")
	bisbas_result = bisbas_result_collector(params.python_exe, params.result_collector_script, bisbas_analyzed, "bisbas result terminal")
	sam_result = sam_result_collector(params.python_exe, params.result_collector_script, sam_concat, "sam result terminal")
	be7_result = be7_result_collector(params.python_exe, params.result_collector_script, be7_concat, "be7 result terminal")
	ea11_result = ea11_result_collector(params.python_exe, params.result_collector_script, ea11_concat, "ea11 result terminal")
	physio_result = physio_result_collector(params.python_exe, params.result_collector_script, physio_concat, "physio result terminal")
	eeg_psd_result = eeg_psd_result_collector(params.python_exe, params.result_collector_script, eeg_psd_concat, "eeg_psd result terminal")
	eeg_contrast_result = eeg_contrast_result_collector(params.python_exe, params.result_collector_script, eeg_psd_contrasts, "eeg_contrast result terminal")
	eeg_fai_result = eeg_fai_result_collector(params.python_exe, params.result_collector_script, psd_fai_concat, "eeg_fai result terminal")
	fnirs_result = fnirs_result_collector(params.python_exe, params.result_collector_script, fnirs_concat, "fnirs result terminal")
	fnirs_fai_result = fnirs_fai_result_collector(params.python_exe, params.result_collector_script, fnirs_fai_concat, "fnirs_fai result terminal")

	// Step 6: Finalize participants — triggers per participant when all 13 result collectors complete.
	finalize_participant(
		panas_result.mix(bisbas_result, sam_result, be7_result, ea11_result,
		                 physio_result, eeg_psd_result, eeg_contrast_result,
						 eeg_fai_result, fnirs_result, fnirs_fai_result),
		11,
		participant_context
	)

	// Step 7: Collect L2 results with clean names into EV_l2/results/
	l2_profile_result = condition_profile_result_collector(params.python_exe, params.result_collector_script, cross_modal_wide, "condition_profile group_log result")
	l2_correl_result = cross_modal_correl_result_collector(params.python_exe, params.result_collector_script, cross_modal_correl, "cross_modal_correl group_log result")

	// Step 8: Finalize group-level (l2) outputs — commits EV_l2 once all l2 result collectors are done.
	finalize_l2(l2_profile_result.mix(l2_correl_result))
}
