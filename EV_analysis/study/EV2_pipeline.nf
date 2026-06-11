// EV2 Secondary Analysis Pipeline
//
// Processes preprocessed EV2 .dat files (Koelstra et al. 2012) to test
// EEG-physiological correlates of continuous valence and arousal ratings.
//
// Input  : data_preprocessed_python/s01.dat … s32.dat  (128 Hz, 40 ch, 40 trials)
// Output : per-participant correlation tables (FAI, HRV, EDA × valence/arousal)
//          + EEG PSD descriptives + EEG FAI per-trial table
//          + PLV (EEG-HRV phase coupling) + WP3/WP4 cross-modal correlations

nextflow.enable.dsl=2

// ── Include all module aliases (defined in EV2_modules.nf) ────────────────
include {
    finalize_participant;
    finalize_l2;
    deap_bootstrap;
    deap_ingestor;
    lgcrct_l2_analyzer;
    result_collector;
    labels_result_collector;
    eeg_file_finder; eda_file_finder; bvp_file_finder; labels_file_finder;
    eeg_spectrum_analyzer; eeg_spectrum_result_collector;
    eeg_fai_psd_analyzer;
    eeg_fai_analyzer;
    eeg_windowing_processor;
    eeg_roi_psd_analyzer;
    eeg_roi_psd_all_file_finder;
    eeg_frontal_row_filter; eeg_parietal_row_filter;
    eeg_frontal_label_binner; eeg_parietal_label_binner;
    eeg_frontal_bootstrap_analyzer; eeg_parietal_bootstrap_analyzer;
    eeg_frontal_bs_file_finder; eeg_parietal_bs_file_finder;
    eeg_frontal_bs_concatenating_processor; eeg_parietal_bs_concatenating_processor;
    eeg_frontal_bootstrap_result_collector; eeg_parietal_bootstrap_result_collector;
    eda_amplitude_analyzer;
    bvp_peak_detector;
    hrv_interval_analyzer;
    fai_correlation_analyzer; eda_correlation_analyzer; hrv_correlation_analyzer;
    multimodal_correl_analyzer; multimodal_correl_result_collector;
    eda_label_binner; hrv_label_binner; fai_label_binner;
    eda_bootstrap_analyzer; hrv_bootstrap_analyzer; fai_bootstrap_analyzer;
    eda_bs_file_finder; hrv_bs_file_finder; fai_bs_file_finder;
    eda_bs_concatenating_processor; hrv_bs_concatenating_processor; fai_bs_concatenating_processor;
    eda_bootstrap_result_collector; hrv_bootstrap_result_collector; fai_bootstrap_result_collector;
    anova_merger; anova_combined_analyzer; anova_combined_result_collector;
    eda_l2_anova_analyzer; hrv_l2_anova_analyzer; fai_l2_anova_analyzer;
    eeg_frontal_ols_processor; eeg_parietal_ols_processor;
    eeg_frontal_contrast_processor; eeg_parietal_contrast_processor;
    eeg_frontal_contrast_result_collector; eeg_parietal_contrast_result_collector;
    eeg_frontal_l2_anova_analyzer; eeg_parietal_l2_anova_analyzer;
    l2_anova_consolidator;
} from './EV2_modules.nf'

// ── Participant ID helpers ─────────────────────────────────────────────────
// All output files are prefixed DEAP_NN_* so split('_')[0..1].join('_') = "DEAP_NN"
def pid = { f -> f.baseName.toString().split('_')[0..1].join('_') }

// ── Workflow ───────────────────────────────────────────────────────────────
workflow {

    // Step 1: Download EV2 dataset via kagglehub (idempotent — uses local cache if present)
    // kaggle_downloader reads kaggle_id from kaggle_config.parquet, calls
    // kagglehub.dataset_download(), finds s*.dat files, and emits one
    // DEAP_NN_dat_path.parquet trigger per participant.
    // .flatten() fans out the 32 trigger files into individual channel items.
    kaggle_config = Channel.value(file("${workflow.launchDir}/${params.kaggle_config}"))

    dat_triggers = deap_bootstrap(params.python_exe, params.ev2_dataset_setup_script,
        kaggle_config, "s*.dat DEAP dat_path 32")
        .flatten()
        // IOInterface stages input parquet(s) alongside produced outputs.
        // Keep only participant trigger files (DEAP_NN_dat_path.parquet).
        .filter { f -> f.name ==~ /DEAP_\d{2}_dat_path\.parquet/ }

    // Normalise participant ID from trigger filename: DEAP_01_dat.parquet → "DEAP_01"
    participant_id = dat_triggers.map { f -> f.baseName.toString().split('_')[0..1].join('_') }

    // Step 2: Ingest each participant's .dat → epoch parquets + signal pointer
    // deap_ingestor reads dat_path from the trigger parquet, loads the .dat file,
    // writes DEAP_NN_{eeg,eda,bvp}_epochs.parquet + labels.parquet into a subfolder,
    // and returns a signal pointer parquet (folder_path) for file_finder.
    deap_signal = deap_ingestor(params.python_exe, params.ev2_ingestor_script,
        dat_triggers, "true 128")

    // Step 3: Extract modality parquets from signal pointer
    eeg_epochs  = eeg_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*eeg_epochs.parquet eeg")
    eda_epochs  = eda_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*eda_epochs.parquet eda")
    bvp_epochs  = bvp_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*bvp_epochs.parquet bvp")
    labels      = labels_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*labels.parquet labels")

    labels_result = labels_result_collector(params.python_exe, params.result_collector_script,
        labels, "labels result terminal")

    // Step 4: EEG analysis
    // 4a. Full-spectrum descriptive (all trials, no variance band — quality check)
    eeg_spectrum = eeg_spectrum_analyzer(params.python_exe, params.spectrum_analyzer_script,
        eeg_epochs, "${params.eeg_psd_channels} None ${params.eeg_max_freq} terminal")
    eeg_spectrum_result = eeg_spectrum_result_collector(params.python_exe, params.result_collector_script,
        eeg_spectrum, "eeg_spectrum result terminal")

    // 4a-pre. Per-trial alpha band power for FAI
    eeg_alpha_psd = eeg_fai_psd_analyzer(params.python_exe, params.psd_analyzer_script,
        eeg_epochs, "{'alpha': [8, 13]} None None")

    // 4b. FAI per-trial flat table (epoch_output=true on per-trial alpha power)
    eeg_fai = eeg_fai_analyzer(params.python_exe, params.asymmetry_analyzer_script,
        eeg_alpha_psd,
        "${params.fai_electrode_pairs} log ${params.fai_band_name} None None true")

    eeg_fai_result = result_collector(params.python_exe, params.result_collector_script,
        eeg_fai, "eeg_fai result terminal")

    // 4c. ROI band-power per trial (Frontal + Parietal; alpha/beta/theta)
    // psd_analyzer region mode: outputs {condition=trial_id, epoch_id, region, alpha, beta, theta}
    // row_filter extracts one ROI each for downstream label-binner and multimodal correlation
    // Sliding-window sub-epochs increase ANOVA df (60s trials → overlapping 30s windows at 10s steps)
    eeg_windowed = eeg_windowing_processor(params.python_exe, params.epoching_processor_script,
        eeg_epochs, "sliding ${params.eeg_window_size} ${params.eeg_step_size}")
    eeg_roi_psd = eeg_roi_psd_analyzer(params.python_exe, params.psd_analyzer_script,
        eeg_windowed, "${params.bands_config} None ${params.eeg_rois} ${params.psd_y_lim}")
    // Expand signal pointer → single combined flat file (all 40 trials × all regions)
    eeg_roi_psd_combined = eeg_roi_psd_all_file_finder(params.python_exe, params.file_finder_script,
        eeg_roi_psd, "*psd_all.parquet eeg_roi_psd_all")
    eeg_frontal_epochs = eeg_frontal_row_filter(params.python_exe, params.row_filter_processor_script,
        eeg_roi_psd_combined, "region Frontal true")
    eeg_parietal_epochs = eeg_parietal_row_filter(params.python_exe, params.row_filter_processor_script,
        eeg_roi_psd_combined, "region Parietal true")

    // Step 5: EDA amplitude per trial
    eda_amplitude = eda_amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script,
        eda_epochs, "eda None terminal")

    // Step 6: HRV per trial
    // 6a. Peak detection on BVP signal (scipy mode; ~120 bpm at rest → min_distance ~0.4s)
    bvp_peaks = bvp_peak_detector(params.python_exe, params.peak_detection_processor_script,
        bvp_epochs, "bvp ${params.ev2_sfreq} scipy None 0.4")

    // 6b. RMSSD per condition (= per trial)
    hrv_rmssd = hrv_interval_analyzer(params.python_exe, params.interval_analyzer_script,
        bvp_peaks, "peak_sample None 'BVP (ms)' RMSSD flat_table terminal")

    // Step 7: Correlate each modality with valence + arousal
    // Join: metrics + labels → correlation_analyzer takes [metrics, labels] as input tuple
    fai_corr_inputs = participant_id
        .join(eeg_fai.map { f -> [pid(f), f] })
        .join(labels.map  { f -> [pid(f), f] })
        .map { p, fai, lab -> [fai, lab] }
    fai_correlation = fai_correlation_analyzer(params.python_exe,
        params.correlation_analyzer_script, fai_corr_inputs,
        "valence,arousal None spearman terminal result")

    eda_corr_inputs = participant_id
        .join(eda_amplitude.map { f -> [pid(f), f] })
        .join(labels.map        { f -> [pid(f), f] })
        .map { p, amp, lab -> [amp, lab] }
    eda_correlation = eda_correlation_analyzer(params.python_exe,
        params.correlation_analyzer_script, eda_corr_inputs,
        "valence,arousal None spearman terminal result")

    hrv_corr_inputs = participant_id
        .join(hrv_rmssd.map { f -> [pid(f), f] })
        .join(labels.map    { f -> [pid(f), f] })
        .map { p, hrv, lab -> [hrv, lab] }
    hrv_correlation = hrv_correlation_analyzer(params.python_exe,
        params.correlation_analyzer_script, hrv_corr_inputs,
        "valence,arousal None spearman terminal result")

    // Step 8: Per-condition bootstrap analysis (valence_high/low + arousal_high/low)
    // ──────────────────────────────────────────────────────────────────────────────
    // Mirrors the EV pilot bootstrap approach: bin the 40 DEAP trials by the same
    // valence / arousal thresholds used at L2, then bootstrap within each bin to
    // produce mean ± CI bar plots that are directly comparable to EV pilot results.
    //
    // EDA — uses amplitude signal pointer (per-trial epoch-level files)
    eda_bin_inputs = participant_id
        .join(eda_amplitude.map { f -> [pid(f), f] })
        .join(labels.map        { f -> [pid(f), f] })
        .map { p, amp, lab -> [amp, lab] }
    eda_binned = eda_label_binner(params.python_exe, params.ev2_label_binner_script,
        eda_bin_inputs,
        "${params.ev2_valence_threshold} ${params.ev2_arousal_threshold} auto ${params.ev2_neutral_band} terminal")
    eda_bootstrap = eda_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script,
        eda_binned,
        "condition epoch_id value ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} None 'Conductance Change (uS)'")
    // Single file_finder collects all available bs*.parquet for each participant at once.
    // Participants with fewer bins (< 6) produce fewer files — all are included without error.
    eda_bs_files = eda_bs_file_finder(params.python_exe, params.file_finder_script,
        eda_bootstrap, "*bs*.parquet eda_bs")
    eda_bs_concat = eda_bs_concatenating_processor(params.python_exe, params.concatenating_processor_script,
        eda_bs_files, "eda_binned terminal")
    eda_bootstrap_result = eda_bootstrap_result_collector(params.python_exe,
        params.result_collector_script, eda_bs_concat, "eda_bootstrap result terminal")

    // HRV — uses flat_table from interval_analyzer
    hrv_bin_inputs = participant_id
        .join(hrv_rmssd.map { f -> [pid(f), f] })
        .join(labels.map    { f -> [pid(f), f] })
        .map { p, hrv, lab -> [hrv, lab] }
    hrv_binned = hrv_label_binner(params.python_exe, params.ev2_label_binner_script,
        hrv_bin_inputs,
        "${params.ev2_valence_threshold} ${params.ev2_arousal_threshold} auto ${params.ev2_neutral_band} terminal")
    hrv_bootstrap = hrv_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script,
        hrv_binned,
        "condition epoch_id value ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} None 'RMSSD (ms)'")
    hrv_bs_files = hrv_bs_file_finder(params.python_exe, params.file_finder_script,
        hrv_bootstrap, "*bs*.parquet hrv_bs")
    hrv_bs_concat = hrv_bs_concatenating_processor(params.python_exe, params.concatenating_processor_script,
        hrv_bs_files, "hrv_binned terminal")
    hrv_bootstrap_result = hrv_bootstrap_result_collector(params.python_exe,
        params.result_collector_script, hrv_bs_concat, "hrv_bootstrap result terminal")

    // FAI — uses epoch_output flat table from asymmetry_analyzer
    fai_bin_inputs = participant_id
        .join(eeg_fai.map  { f -> [pid(f), f] })
        .join(labels.map   { f -> [pid(f), f] })
        .map { p, fai, lab -> [fai, lab] }
    fai_binned = fai_label_binner(params.python_exe, params.ev2_label_binner_script,
        fai_bin_inputs,
        "${params.ev2_valence_threshold} ${params.ev2_arousal_threshold} auto ${params.ev2_neutral_band} terminal")
    fai_bootstrap = fai_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script,
        fai_binned,
        "condition epoch_id value ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} None 'FAI (ln R - ln L)'")
    fai_bs_files = fai_bs_file_finder(params.python_exe, params.file_finder_script,
        fai_bootstrap, "*bs*.parquet fai_bs")
    fai_bs_concat = fai_bs_concatenating_processor(params.python_exe, params.concatenating_processor_script,
        fai_bs_files, "fai_binned terminal")
    fai_bootstrap_result = fai_bootstrap_result_collector(params.python_exe,
        params.result_collector_script, fai_bs_concat, "fai_bootstrap result terminal")

    // Step 8b: EEG ROI band-power bootstrap (Frontal + Parietal, auto_multi_by_col)
    // label_binner auto_multi mode: keeps all numeric columns (alpha, beta, theta).
    // bootstrap_analyzer auto_multi_by_col: iterates columns (bands) first so each output
    // file covers all conditions for one band — x_data=conditions, labels=bands.
    // Matches the pilot's per-band format (x = conditions, colours = bands).
    eeg_frontal_bin_inputs = participant_id
        .join(eeg_frontal_epochs.map { f -> [pid(f), f] })
        .join(labels.map             { f -> [pid(f), f] })
        .map { p, eeg, lab -> [eeg, lab] }
    eeg_frontal_binned = eeg_frontal_label_binner(params.python_exe, params.ev2_label_binner_script,
        eeg_frontal_bin_inputs,
        "${params.ev2_valence_threshold} ${params.ev2_arousal_threshold} auto_multi ${params.ev2_neutral_band} terminal")
    eeg_frontal_bootstrap = eeg_frontal_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script,
        eeg_frontal_binned,
        "condition epoch_id auto_multi_by_col ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} None 'Band Power (uV2/Hz)'")
    eeg_frontal_bs_files = eeg_frontal_bs_file_finder(params.python_exe, params.file_finder_script,
        eeg_frontal_bootstrap, "*bscol*.parquet eeg_frontal_bs")
    eeg_frontal_bs_concat = eeg_frontal_bs_concatenating_processor(params.python_exe,
        params.concatenating_processor_script, eeg_frontal_bs_files, "eeg_frontal_binned terminal")
    eeg_frontal_bootstrap_result = eeg_frontal_bootstrap_result_collector(params.python_exe,
        params.result_collector_script, eeg_frontal_bs_concat, "eeg_frontal_bootstrap result terminal")

    eeg_parietal_bin_inputs = participant_id
        .join(eeg_parietal_epochs.map { f -> [pid(f), f] })
        .join(labels.map              { f -> [pid(f), f] })
        .map { p, eeg, lab -> [eeg, lab] }
    eeg_parietal_binned = eeg_parietal_label_binner(params.python_exe, params.ev2_label_binner_script,
        eeg_parietal_bin_inputs,
        "${params.ev2_valence_threshold} ${params.ev2_arousal_threshold} auto_multi ${params.ev2_neutral_band} terminal")
    eeg_parietal_bootstrap = eeg_parietal_bootstrap_analyzer(params.python_exe, params.bootstrap_analyzer_script,
        eeg_parietal_binned,
        "condition epoch_id auto_multi_by_col ${params.bootstrap_n_boot} ${params.bootstrap_ci_method} ${params.bootstrap_alpha} None 'Band Power (uV2/Hz)'")
    eeg_parietal_bs_files = eeg_parietal_bs_file_finder(params.python_exe, params.file_finder_script,
        eeg_parietal_bootstrap, "*bscol*.parquet eeg_parietal_bs")
    eeg_parietal_bs_concat = eeg_parietal_bs_concatenating_processor(params.python_exe,
        params.concatenating_processor_script, eeg_parietal_bs_files, "eeg_parietal_binned terminal")
    eeg_parietal_bootstrap_result = eeg_parietal_bootstrap_result_collector(params.python_exe,
        params.result_collector_script, eeg_parietal_bs_concat, "eeg_parietal_bootstrap result terminal")

    // Step 8e: EEG pairwise contrasts (within-participant, L1)
    // OLS fits one beta per bin per band; contrast_processor computes weighted t-statistics
    // for the key theoretical comparisons (high vs low for each dimension).
    eeg_frontal_ols = eeg_frontal_ols_processor(params.python_exe, params.ols_processor_script,
        eeg_frontal_binned, "eeg_frontal_ols")
    eeg_frontal_contrast = eeg_frontal_contrast_processor(params.python_exe, params.contrast_processor_script,
        eeg_frontal_ols, "${params.eeg_bin_contrasts} eeg_frontal_contrast terminal")
    eeg_frontal_contrast_result = eeg_frontal_contrast_result_collector(params.python_exe,
        params.result_collector_script, eeg_frontal_contrast, "eeg_frontal_contrast result terminal")

    eeg_parietal_ols = eeg_parietal_ols_processor(params.python_exe, params.ols_processor_script,
        eeg_parietal_binned, "eeg_parietal_ols")
    eeg_parietal_contrast = eeg_parietal_contrast_processor(params.python_exe, params.contrast_processor_script,
        eeg_parietal_ols, "${params.eeg_bin_contrasts} eeg_parietal_contrast terminal")
    eeg_parietal_contrast_result = eeg_parietal_contrast_result_collector(params.python_exe,
        params.result_collector_script, eeg_parietal_contrast, "eeg_parietal_contrast result terminal")

    // Step 8c: Within-participant ANOVA (L1)
    // Concatenates all binned flat tables with source labels, then runs one-way
    // ANOVA per source comparing 6 valence/arousal bins (df2 ~ 34 with 40 trials).
    anova_epoch_files = participant_id
        .join(eda_binned.map          { f -> [pid(f), f] })
        .join(hrv_binned.map          { f -> [pid(f), f] })
        .join(fai_binned.map          { f -> [pid(f), f] })
        .join(eeg_frontal_binned.map  { f -> [pid(f), f] })
        .join(eeg_parietal_binned.map { f -> [pid(f), f] })
        .map { p, e, h, f, ef, ep -> [e, h, f, ef, ep] }
    anova_epochs_concat = anova_merger(params.python_exe, params.concatenating_processor_script,
        anova_epoch_files, "labels=EDA,HRV,FAI,EEG_Frontal,EEG_Parietal anova_epochs terminal")
    anova_combined = anova_combined_analyzer(params.python_exe, params.anova_analyzer_script,
        anova_epochs_concat, "auto condition false None source terminal")
    anova_combined_result = anova_combined_result_collector(params.python_exe,
        params.result_collector_script, anova_combined, "anova_combined result terminal")

    // Step 8d: Multimodal EEG-HRV correlation (per-trial Spearman r, N=40)
    // EEG frontal band power (alpha/beta/theta per trial) x RMSSD per trial.
    // correl_analyzer multi-file mode: aggregates each input on condition (= trial_id),
    // renames 'value' to file-stem_value, joins, then correlates all numeric pairs.
    // Result: Spearman r heatmap for EEG bands x RMSSD across 40 trials.
    multimodal_files = participant_id
        .join(eeg_frontal_epochs.map { f -> [pid(f), f] })
        .join(hrv_rmssd.map          { f -> [pid(f), f] })
        .map { p, eeg, hrv -> [eeg, hrv] }
    multimodal_correl = multimodal_correl_analyzer(params.python_exe, params.correl_analyzer_script,
        multimodal_files, "None terminal")
    multimodal_correl_result = multimodal_correl_result_collector(params.python_exe,
        params.result_collector_script, multimodal_correl, "multimodal_correl result terminal")

    // Step 9: Finalize participants
    // copied labels/FAI result tables per participant.
    // then triggers git commit and finalization for each one (watch mode compatible).
    //
    // l1_results is the single source of truth for which channels are expected.
    // result_count is derived automatically — add/remove a channel here and the
    // barrier threshold updates with it; no magic number to maintain separately.
    participant_context = participant_id
        .map { p -> [p, "${params.output_dir}/${params.project_name}_l1/${p}"] }

    // Raw epoch data is auto-exported to tables/ by workflow_wrapper.nf (no explicit wiring needed).
    def l1_results = [
        fai_correlation, eda_correlation, hrv_correlation,
        labels_result, eeg_fai_result,
        eda_bootstrap_result, hrv_bootstrap_result, fai_bootstrap_result,
        eeg_frontal_bootstrap_result, eeg_parietal_bootstrap_result,
        eeg_frontal_contrast_result, eeg_parietal_contrast_result,
        anova_combined_result, multimodal_correl_result,
        eeg_spectrum_result
    ]
    finalize_participant(
        l1_results[0].mix(*l1_results.drop(1)),
        l1_results.size(),
        participant_context
    )

    // Step 10: L2 group-level analysis
    // L2 ANOVA: one-way F-test across all bins (valence_high/low + arousal_high/low).
    eda_l2_anova = eda_l2_anova_analyzer(params.python_exe, params.anova_analyzer_script,
        eda_binned.collect(),
        "auto condition false None None group_log table terminal")
    hrv_l2_anova = hrv_l2_anova_analyzer(params.python_exe, params.anova_analyzer_script,
        hrv_binned.collect(),
        "auto condition false None None group_log table terminal")
    fai_l2_anova = fai_l2_anova_analyzer(params.python_exe, params.anova_analyzer_script,
        fai_binned.collect(),
        "auto condition false None None group_log table terminal")

    eeg_frontal_l2_anova = eeg_frontal_l2_anova_analyzer(params.python_exe, params.anova_analyzer_script,
        eeg_frontal_binned.collect(),
        "auto condition false None None group_log table terminal")
    eeg_parietal_l2_anova = eeg_parietal_l2_anova_analyzer(params.python_exe, params.anova_analyzer_script,
        eeg_parietal_binned.collect(),
        "auto condition false None None group_log table terminal")
    l2_anova_outputs = eda_l2_anova.mix(hrv_l2_anova).mix(fai_l2_anova).mix(eeg_frontal_l2_anova).mix(eeg_parietal_l2_anova)
    l2_anova_summary = l2_anova_consolidator(params.python_exe, params.l2_anova_consolidator_script,
        l2_anova_outputs.collect(),
        "--consolidate-l2 --modality-map ${params.l2_anova_modality_map} l2_anova_summary group_log result terminal")
    l2_outputs = l2_anova_outputs.mix(l2_anova_summary)
    if (params.enable_lgcrct_branch) {
        lgcrct_l2 = lgcrct_l2_analyzer(params.python_exe, params.lgcrct_loso_script,
            eeg_epochs.collect(),
            "--mode ${params.lgcrct_mode} --targets ${params.lgcrct_targets} --sfreq ${params.ev2_sfreq} --window-sec ${params.lgcrct_window_sec} --step-sec ${params.lgcrct_step_sec} --band ${params.lgcrct_band} --threshold ${params.lgcrct_threshold} --use-lgc ${params.lgcrct_use_lgc} --half-window ${params.lgcrct_half_window} --cov-estimator ${params.lgcrct_cov_estimator} --lgc-mean ${params.lgcrct_lgc_mean} group_log result terminal")
        l2_outputs = l2_outputs.mix(lgcrct_l2)
    }

    finalize_l2(l2_outputs)
}
