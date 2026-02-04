#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// =========================================
// EmotiView Nextflow Pipeline
// =========================================

// Import all modules from EV_modules.nf

include { 
    participant_discovery; finalize_participant;
    txt_reader; xdf_reader; extracting_processor;
    eda_filtering_processor; ecg_filtering_processor; eeg_filtering_processor; fnirs_filtering_processor; peak_detection_processor; referencing_processor; log_transform_processor; tddr_processor; linear_transform_processor; events_processor; tree_processor; eda_epoching_processor; ecg_epoching_processor; eeg_epoching_processor; fnirs_epoching_processor;
    ic_analyzer; interval_analyzer; amplitude_analyzer; psd_analyzer; group_analyzer; fai1_analyzer; fai2_analyzer; fai3_analyzer; panas_analyzer; bisbas_analyzer; sam_analyzer; be7_analyzer; ea11_analyzer;
    sam_concatenating_processor; be7_concatenating_processor; ea11_concatenating_processor; hrv_concatenating_processor; eda_concatenating_processor; psd_concatenating_processor; fai_concatenating_processor; fnirs_concatenating_processor; fnirs_asym_concatenating_processor;
    fai_merger;
    fnirs_asym1_analyzer; fnirs_asym2_analyzer; fnirs_asym3_analyzer;
    hrv_relative_analyzer; eda_relative_analyzer; psd_relative_analyzer; fai_relative_analyzer; fnirs_relative_analyzer; fnirs_asym_relative_analyzer; fai_combined_relative_analyzer;
    panas_plotter; bisbas_plotter; sam_plotter; be7_plotter; ea11_plotter; hrv_plotter; eda_plotter; psd_plotter; fai_plotter; fnirs_plotter; fnirs_asym_plotter; fai_combined_plotter;
    hrv_rel_plotter; eda_rel_plotter; psd_rel_plotter; fai_rel_plotter; fnirs_rel_plotter; fnirs_asym_rel_plotter; fai_combined_rel_plotter;
    aux_file_finder; eda_file_finder; ecg_file_finder; trigger_file_finder; eeg_file_finder; eeg_cleaned_file_finder; fnirs_file_finder; sam1_file_finder; sam2_file_finder; sam3_file_finder; be71_file_finder; be72_file_finder; be73_file_finder; ea111_file_finder; ea112_file_finder; ea113_file_finder; eda1_file_finder; eda2_file_finder; eda3_file_finder; hrv1_file_finder; hrv2_file_finder; hrv3_file_finder; psd1_file_finder; psd2_file_finder; psd3_file_finder; psd1_raw_file_finder; psd2_raw_file_finder; psd3_raw_file_finder; fnirs1_file_finder; fnirs2_file_finder; fnirs3_file_finder; fnirs_asym_input1_file_finder; fnirs_asym_input2_file_finder; fnirs_asym_input3_file_finder;
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
    aux_stream = aux_file_finder(params.python_exe, params.file_finder_script, xdf_data, "type:EEG.fif")
	aux_extr = extracting_processor(params.python_exe, params.extracting_processor_script, aux_stream, params.extraction_columns)

	eda_stream     = eda_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr1.parquet")
	ecg_stream     = ecg_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr2.parquet")
	trigger_stream = trigger_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr3.parquet")
	eeg_stream     = eeg_file_finder(params.python_exe, params.file_finder_script, aux_extr, "*extr4.fif")

	// Extract fNIRS stream from XDF
	fnirs_stream = fnirs_file_finder(params.python_exe, params.file_finder_script, xdf_data, "type:NIRS.fif")

	// Step 3: Modular signal processing chains
	// Events chain: trigger → events
	tree_struct = tree_processor(params.python_exe, params.tree_processor_script, txt_data, "${params.entry_delim} ${params.depth_delim} ${params.kv_delim}")
	tree_events_inputs = participant_id
	    .join(trigger_stream.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .join(tree_struct.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
	    .map { pid, f1, f2 -> [f1, f2] }
	tree_events = events_processor(params.python_exe, params.events_processor_script, tree_events_inputs, "${params.conds}")
	
    // EDA chain: filter only
	eda_filtered = eda_filtering_processor(params.python_exe, params.filtering_processor_script, eda_stream, "${params.eda_l_freq} ${params.eda_h_freq} ${params.eda_channel} ${params.sfreq} ${params.ftype}")

	// ECG chain: filter → peak detection
	ecg_filtered = ecg_filtering_processor(params.python_exe, params.filtering_processor_script, ecg_stream, "${params.ecg_l_freq} ${params.ecg_h_freq} ${params.ecg_channel} ${params.sfreq} ${params.ftype}")
	ecg_peaks = peak_detection_processor(params.python_exe, params.peak_detection_processor_script, ecg_filtered, "${params.ecg_channel} ${params.sfreq} ecg")
    
	// EEG chain: referencing → filter → ICA
	eeg_reref = referencing_processor(params.python_exe, params.referencing_processor_script, eeg_stream, "${params.eeg_reference}")
	eeg_filtered = eeg_filtering_processor(params.python_exe, params.filtering_processor_script, eeg_reref, "${params.eeg_l_freq} ${params.eeg_h_freq}")
    ic_analyzed = ic_analyzer(params.python_exe, params.ic_analyzer_script, eeg_filtered, "")
	eeg_cleaned = eeg_cleaned_file_finder(params.python_exe, params.file_finder_script, ic_analyzed, "*ica_cleaned.fif")

	// fNIRS chain: log transform → TDDR robust correction → linear unmixing (MBLL) → filter
	fnirs_log = log_transform_processor(params.python_exe, params.log_transform_script, fnirs_stream, "${params.fnirs_baseline_sec}")
	fnirs_tddr = tddr_processor(params.python_exe, params.tddr_processor_script, fnirs_log, "")
	fnirs_hbo = linear_transform_processor(params.python_exe, params.linear_transform_script, fnirs_tddr, "mbll ${params.fnirs_ppf} ${params.fnirs_channels}")
	fnirs_filtered = fnirs_filtering_processor(params.python_exe, params.filtering_processor_script, fnirs_hbo, "${params.fnirs_l_freq} ${params.fnirs_h_freq}")

	// Step 4: Epoching - use join to match data with events by participant ID
	eda_epoch_inputs = participant_id
	    .join(eda_filtered.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
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
		.join(sam1_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam1.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(sam2_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam2.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(sam3_file_finder(params.python_exe, params.file_finder_script, sam_analyzed, "*sam3.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	sam_concat = sam_concatenating_processor(params.python_exe, params.concatenating_processor_script, sam_files, "sam_concat")

	be7_analyzed = be7_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.be7_param} be7")
	be7_files = participant_id
		.join(be71_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be71.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(be72_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be72.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(be73_file_finder(params.python_exe, params.file_finder_script, be7_analyzed, "*be73.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	be7_concat = be7_concatenating_processor(params.python_exe, params.concatenating_processor_script, be7_files, "be7_concat")
	
	ea11_analyzed = ea11_analyzer(params.python_exe, params.quest_analyzer_script, tree_struct, "${params.ea11_param} ea11")
	ea11_files = participant_id
		.join(ea111_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea111.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(ea112_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea112.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(ea113_file_finder(params.python_exe, params.file_finder_script, ea11_analyzed, "*ea113.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	ea11_concat = ea11_concatenating_processor(params.python_exe, params.concatenating_processor_script, ea11_files, "ea11_concat")
	
	// Physiological analyses (following test analysis procedural tree)
	// EDA analysis: amplitude (peak-baseline)
	eda_analyzed = amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, eda_epoched, "peak_baseline ${params.eda_y_lim} 'Conductance Change (μS)' eda")
	eda_files = participant_id
		.join(eda1_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*eda1.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda2_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*eda2.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(eda3_file_finder(params.python_exe, params.file_finder_script, eda_analyzed, "*eda3.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	eda_concat = eda_concatenating_processor(params.python_exe, params.concatenating_processor_script, eda_files, "eda_concat")
	
	// HRV analysis on ECG epochs
	hrv_analyzed = interval_analyzer(params.python_exe, params.interval_analyzer_script, ecg_epoched, "peak_sample ${params.hrv_y_lim} 'Value (ms)' hrv ${params.hrv_metrics_mode}")
	hrv_files = participant_id
		.join(hrv1_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*hrv1.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(hrv2_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*hrv2.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(hrv3_file_finder(params.python_exe, params.file_finder_script, hrv_analyzed, "*hrv3.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	hrv_concat = hrv_concatenating_processor(params.python_exe, params.concatenating_processor_script, hrv_files, "hrv_concat")
	
	// EEG analyses: PSD
	psd_analyzed = psd_analyzer(params.python_exe, params.psd_analyzer_script, eeg_epoched, "${params.bands_config} None ${params.psd_y_lim}")
	
	// Raw PSD files (per-channel data) for FAI asymmetry analysis
	psd1_raw = psd1_raw_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd1.parquet")
	psd2_raw = psd2_raw_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd2.parquet")
	psd3_raw = psd3_raw_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd3.parquet")
	
	// Plot-ready files for PSD concatenation
	psd1 = psd1_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd1_plot.parquet")
	psd2 = psd2_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd2_plot.parquet")
	psd3 = psd3_file_finder(params.python_exe, params.file_finder_script, psd_analyzed, "*psd3_plot.parquet")
	psd_files = participant_id
		.join(psd1.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(psd2.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(psd3.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	psd_concat = psd_concatenating_processor(params.python_exe, params.concatenating_processor_script, psd_files, "psd_concat")
	
	// PSD FAI (asymmetry): process each condition's raw PSD separately, then concatenate
	fai1_analyzed = fai1_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd1_raw, "${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None psd_fai")
	fai2_analyzed = fai2_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd2_raw, "${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None psd_fai")
	fai3_analyzed = fai3_analyzer(params.python_exe, params.asymmetry_analyzer_script, psd3_raw, "${params.electrode_pairs} log ${params.fai_band_name} ${params.fai_y_lim} None psd_fai")
	psd_fai_files = participant_id
		.join(fai1_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fai2_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fai3_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	psd_fai_concat = fai_concatenating_processor(params.python_exe, params.concatenating_processor_script, psd_fai_files, "psd_fai_concat")
	
	// fNIRS: ROI analysis
	fnirs_analyzed = group_analyzer(params.python_exe, params.group_analyzer_script, fnirs_epoched, "${params.fnirs_rois} ${params.fnirs_y_lim} ROI 'Hb Change (μM)' hbc ${params.fnirs_epoch_baseline_sec} ${params.fnirs_chromophores}")
	fnirs_files = participant_id
		.join(fnirs1_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc1.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs2_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc2.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs3_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc3.parquet").map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	fnirs_concat = fnirs_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_files, "hbc_concat")
	
	// fNIRS FAI: Compute R-L hemispheric asymmetry from ROI data (uses asymmetry_analyzer with mode=diff)
	fnirs_fai1 = fnirs_asym1_analyzer(params.python_exe, params.asymmetry_analyzer_script,
		fnirs_asym_input1_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc1.parquet"),
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)' fnirs_fai")
	fnirs_fai2 = fnirs_asym2_analyzer(params.python_exe, params.asymmetry_analyzer_script,
		fnirs_asym_input2_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc2.parquet"),
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)' fnirs_fai")
	fnirs_fai3 = fnirs_asym3_analyzer(params.python_exe, params.asymmetry_analyzer_script,
		fnirs_asym_input3_file_finder(params.python_exe, params.file_finder_script, fnirs_analyzed, "*hbc3.parquet"),
		"${params.fnirs_asym_pairs} diff None ${params.fnirs_asym_y_lim} 'HbO2 Asymmetry (R-L)' hbc_fai")
	fnirs_fai_files = participant_id
		.join(fnirs_fai1.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_fai2.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.join(fnirs_fai3.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] })
		.map { pid, f1, f2, f3 -> [f1, f2, f3] }
	fnirs_fai_concat = fnirs_asym_concatenating_processor(params.python_exe, params.concatenating_processor_script, fnirs_fai_files, "hbc_fai_concat")
	
	// Relative normalization to neutral baseline
	// Apply relative normalization (subtract NEU baseline from each condition, excluding NEU from output)
	hrv_rel = hrv_relative_analyzer(params.python_exe, params.relative_analyzer_script, hrv_concat, "NEU ${params.hrv_rel_y_lim}")
	eda_rel = eda_relative_analyzer(params.python_exe, params.relative_analyzer_script, eda_concat, "NEU ${params.eda_rel_y_lim}")
	psd_rel = psd_relative_analyzer(params.python_exe, params.relative_analyzer_script, psd_concat, "NEU ${params.psd_rel_y_lim}")
	psd_fai_rel = fai_relative_analyzer(params.python_exe, params.relative_analyzer_script, psd_fai_concat, "NEU ${params.fai_rel_y_lim}")
	fnirs_rel = fnirs_relative_analyzer(params.python_exe, params.relative_analyzer_script, fnirs_concat, "NEU ${params.fnirs_rel_y_lim}")
	fnirs_fai_rel = fnirs_asym_relative_analyzer(params.python_exe, params.relative_analyzer_script, fnirs_fai_concat, "NEU ${params.fnirs_asym_rel_y_lim}")

	// Step 6: Plotting - join data with context by participant ID to ensure correct matching
	// Multi-item questionnaire plots
	panas_with_ctx = panas_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	panas_plot = panas_plotter(params.python_exe, params.plotter_script, 
		panas_with_ctx.map { pid, f, folder -> f },
		panas_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_panas" })
	
	bisbas_with_ctx = bisbas_analyzed.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	bisbas_plot = bisbas_plotter(params.python_exe, params.plotter_script, 
		bisbas_with_ctx.map { pid, f, folder -> f },
		bisbas_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_bisbas" })
    
	// Condition-based questionnaire plots
	sam_with_ctx = sam_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	sam_plot = sam_plotter(params.python_exe, params.plotter_script, 
		sam_with_ctx.map { pid, f, folder -> f },
		sam_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_sam" })
	
	be7_with_ctx = be7_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	be7_plot = be7_plotter(params.python_exe, params.plotter_script, 
		be7_with_ctx.map { pid, f, folder -> f },
		be7_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_be7" })
	
	ea11_with_ctx = ea11_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	ea11_plot = ea11_plotter(params.python_exe, params.plotter_script, 
		ea11_with_ctx.map { pid, f, folder -> f },
		ea11_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_ea11" })

	// Physiological plots
	hrv_with_ctx = hrv_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	hrv_plot = hrv_plotter(params.python_exe, params.plotter_script, 
		hrv_with_ctx.map { pid, f, folder -> f },
		hrv_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hrv" })
	
	eda_with_ctx = eda_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	eda_plot = eda_plotter(params.python_exe, params.plotter_script, 
		eda_with_ctx.map { pid, f, folder -> f },
		eda_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_eda" })
	
	psd_with_ctx = psd_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	psd_plot = psd_plotter(params.python_exe, params.plotter_script, 
		psd_with_ctx.map { pid, f, folder -> f },
		psd_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_psd" })
	
	psd_fai_with_ctx = psd_fai_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	psd_fai_plot = fai_plotter(params.python_exe, params.plotter_script, 
		psd_fai_with_ctx.map { pid, f, folder -> f },
		psd_fai_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_psd_fai" })
	
	fnirs_with_ctx = fnirs_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	fnirs_plot = fnirs_plotter(params.python_exe, params.plotter_script, 
		fnirs_with_ctx.map { pid, f, folder -> f },
		fnirs_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hbc" })

	fnirs_fai_with_ctx = fnirs_fai_concat.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	fnirs_fai_plot = fnirs_asym_plotter(params.python_exe, params.plotter_script, 
		fnirs_fai_with_ctx.map { pid, f, folder -> f },
		fnirs_fai_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hbc_fai" })

	// Relative-normalized physiological plots
	hrv_rel_with_ctx = hrv_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	hrv_rel_plot = hrv_rel_plotter(params.python_exe, params.plotter_script, 
		hrv_rel_with_ctx.map { pid, f, folder -> f },
		hrv_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hrv_rel" })
	
	eda_rel_with_ctx = eda_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	eda_rel_plot = eda_rel_plotter(params.python_exe, params.plotter_script, 
		eda_rel_with_ctx.map { pid, f, folder -> f },
		eda_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_eda_rel" })
	
	psd_rel_with_ctx = psd_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	psd_rel_plot = psd_rel_plotter(params.python_exe, params.plotter_script, 
		psd_rel_with_ctx.map { pid, f, folder -> f },
		psd_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_psd_rel" })
	
	psd_fai_rel_with_ctx = psd_fai_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	psd_fai_rel_plot = fai_rel_plotter(params.python_exe, params.plotter_script, 
		psd_fai_rel_with_ctx.map { pid, f, folder -> f },
		psd_fai_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_psd_fai_rel" })
	
	fnirs_rel_with_ctx = fnirs_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	fnirs_rel_plot = fnirs_rel_plotter(params.python_exe, params.plotter_script, 
		fnirs_rel_with_ctx.map { pid, f, folder -> f },
		fnirs_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hbc_rel" })

	fnirs_fai_rel_with_ctx = fnirs_fai_rel.map { f -> [f.baseName.toString().split('_')[0..1].join('_'), f] }.join(participant_context)
	fnirs_fai_rel_plot = fnirs_asym_rel_plotter(params.python_exe, params.plotter_script, 
		fnirs_fai_rel_with_ctx.map { pid, f, folder -> f },
		fnirs_fai_rel_with_ctx.map { pid, f, folder -> "${workflow.launchDir}/${folder} ${pid}_hbc_fai_rel" })

	// Collect all terminal plot outputs
	def terminal_plots = [
		panas_plot, bisbas_plot, sam_plot, be7_plot, ea11_plot,
		hrv_plot, eda_plot, psd_plot, psd_fai_plot, fnirs_plot, fnirs_fai_plot,
		hrv_rel_plot, eda_rel_plot, psd_rel_plot, psd_fai_rel_plot,
		fnirs_rel_plot, fnirs_fai_rel_plot
	]
	
	finalize_participant(terminal_plots, participant_context)
}
