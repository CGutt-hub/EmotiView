// EV2 Secondary Analysis Pipeline
//
// Processes preprocessed EV2 .dat files (Koelstra et al. 2012) to test
// EEG-physiological correlates of continuous valence and arousal ratings.
//
// Input  : data_preprocessed_python/s01.dat … s32.dat  (128 Hz, 40 ch, 40 trials)
// Output : per-participant correlation tables (FAI, HRV, EDA × valence/arousal)
//          + EEG PSD descriptives + EEG FAI per-trial table
//
// NOTE   : EEG-HRV PLV analysis is deferred (requires plv_analyzer.py, TBD).

nextflow.enable.dsl=2

// ── Include all module aliases (defined in EV2_modules.nf) ────────────────
include {
    finalize_participant;
    finalize_l2;
    deap_bootstrap;
    deap_ingestor;
    study_l2_analyzer;
    lgcrct_l2_analyzer;
    result_collector;
    labels_result_collector;
    plv_result_collector;
    baseline_result_collector;
    fai_label_exporter;
    eeg_file_finder; eda_file_finder; bvp_file_finder; bvp_baseline_file_finder; labels_file_finder;
    eeg_psd_analyzer;
    eeg_fai_analyzer;
    eda_amplitude_analyzer;
    bvp_peak_detector;
    hrv_interval_analyzer;
    fai_correlation_analyzer; eda_correlation_analyzer; hrv_correlation_analyzer; plv_correlation_analyzer; plv_fai_correlation_analyzer;
    plv_analyzer;
} from './EV2_modules.nf'

// ── Participant ID helpers ─────────────────────────────────────────────────
// All output files are prefixed DEAP_NN_* so split('_')[0..1].join('_') = "DEAP_NN"
def pid = { f -> f.baseName.toString().split('_')[0..1].join('_') }

// ── Workflow ───────────────────────────────────────────────────────────────
workflow {

    // Step 1: Download EV2 dataset via kagglehub (idempotent — uses local cache if present)
    // deap_bootstrap reads kaggle_config.parquet, calls kagglehub.dataset_download(),
    // and emits one DEAP_NN_dat.parquet trigger per participant.
    // .flatten() fans out the 32 trigger files into individual channel items.
    kaggle_config = Channel.value(file("${workflow.launchDir}/${params.kaggle_config}"))

    dat_triggers = deap_bootstrap(params.python_exe, params.ev2_bootstrap_script,
        kaggle_config, "32").flatten()

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
    bvp_baseline_epochs = bvp_baseline_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*bvp_baseline_epochs.parquet bvp")
    labels      = labels_file_finder(params.python_exe, params.file_finder_script,
        deap_signal, "*labels.parquet labels")

    labels_result = labels_result_collector(params.python_exe, params.result_collector_script,
        labels, "labels result terminal")

    // Step 4: EEG analysis
    // 4a. PSD descriptive (all trials → per-condition PSD plot; useful for quality check)
    eeg_psd = eeg_psd_analyzer(params.python_exe, params.phase_analyzer_script,
        eeg_epochs, "${params.eeg_psd_channels} None ${params.eeg_max_freq} terminal")

    // 4b. FAI per-trial flat table (epoch_output=true on condition=trial_NN data)
    eeg_fai = eeg_fai_analyzer(params.python_exe, params.asymmetry_analyzer_script,
        eeg_epochs,
        "${params.fai_electrode_pairs} log ${params.fai_band_name} None None true terminal")

    eeg_fai_result = result_collector(params.python_exe, params.result_collector_script,
        eeg_fai, "eeg_fai result terminal")

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

    // 6c. Baseline RMSSD per trial (3s pre-stimulus baseline segments)
    bvp_baseline_peaks = bvp_peak_detector(params.python_exe, params.peak_detection_processor_script,
        bvp_baseline_epochs, "bvp ${params.ev2_sfreq} scipy None 0.4")
    baseline_rmssd = hrv_interval_analyzer(params.python_exe, params.interval_analyzer_script,
        bvp_baseline_peaks, "peak_sample None 'BVP baseline (ms)' RMSSD flat_table terminal")
    baseline_rmssd_result = baseline_result_collector(params.python_exe, params.result_collector_script,
        baseline_rmssd, "baseline_rmssd result terminal")

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

    // Step 8: PLV during stimulus exposure (EEG-HRV only).
    // We intentionally omit EEG-EDA PLV here because 60s trials are more suitable
    // for robust EEG-HRV coupling than for slow EDA phase-coupling estimates in this paradigm.
    // flat_table mode emits a trial-level table for SAM/FAI correlation.
    plv_inputs = participant_id
        .join(eeg_epochs.map { f -> [pid(f), f] })
        .join(bvp_peaks.map  { f -> [pid(f), f] })
        .map { p, eeg, bvp -> [eeg, bvp] }
    plv_table = plv_analyzer(params.python_exe, params.plv_analyzer_script,
        plv_inputs,
        "[{'type':'continuous','channels':['F3','F4'],'freq_band':[8,13],'sfreq':${params.ev2_sfreq}},{'type':'event','column':'time','sfreq':${params.ev2_sfreq}}] None flat_table terminal")
    plv_table_result = plv_result_collector(params.python_exe, params.result_collector_script,
        plv_table, "plv_table result terminal")

    plv_corr_inputs = participant_id
        .join(plv_table.map { f -> [pid(f), f] })
        .join(labels.map    { f -> [pid(f), f] })
        .map { p, plv, lab -> [plv, lab] }
    plv_correlation = plv_correlation_analyzer(params.python_exe,
        params.correlation_analyzer_script, plv_corr_inputs,
        "valence,arousal None spearman terminal result")

    // WP4: Frontal asymmetry (FAI) vs EEG-HRV PLV metrics.
    fai_labels = fai_label_exporter(params.python_exe, params.ev2_fai_label_export_script,
        eeg_fai, "terminal")
    plv_fai_corr_inputs = participant_id
        .join(plv_table.map  { f -> [pid(f), f] })
        .join(fai_labels.map { f -> [pid(f), f] })
        .map { p, plv, fai -> [plv, fai] }
    plv_fai_correlation = plv_fai_correlation_analyzer(params.python_exe,
        params.correlation_analyzer_script, plv_fai_corr_inputs,
        "fai None spearman terminal result")

    // Step 9: Finalize participants — waits for the correlation outputs plus the
    // copied labels/FAI result tables per participant.
    // then triggers git commit and finalization for each one (watch mode compatible).
    participant_context = participant_id
        .map { p -> [p, "${params.output_dir}/${params.project_name}_l1/${p}"] }

    finalize_participant(
        fai_correlation.mix(eda_correlation, hrv_correlation, plv_correlation, plv_fai_correlation,
            labels_result, eeg_fai_result, baseline_rmssd_result, plv_table_result),
        8,
        participant_context
    )

    // Step 10: Study L2 analysis — build participant summaries, copy the relevant
    // participant result tables into EV2_l1/results/, then run group-level ANOVA.
    // The L2 script scans the finished L1 folders on disk, so the collected FAI
    // channel is only used as a completion barrier.
    l2_summary = study_l2_analyzer(params.python_exe, params.study_l2_analysis_script,
        eeg_fai.collect(), "${workflow.launchDir}/${params.output_dir}/${params.project_name}_l1 ${params.project_name} ${params.ev2_sfreq} 5 5 group_log result terminal")

    l2_outputs = l2_summary
    if (params.enable_lgcrct_branch) {
        lgcrct_l2 = lgcrct_l2_analyzer(params.python_exe, params.lgcrct_loso_script,
            eeg_epochs.collect(),
            "--mode ${params.lgcrct_mode} --targets ${params.lgcrct_targets} --sfreq ${params.ev2_sfreq} --window-sec ${params.lgcrct_window_sec} --step-sec ${params.lgcrct_step_sec} --band ${params.lgcrct_band} --threshold ${params.lgcrct_threshold} --use-lgc ${params.lgcrct_use_lgc} --half-window ${params.lgcrct_half_window} --cov-estimator ${params.lgcrct_cov_estimator} --lgc-mean ${params.lgcrct_lgc_mean} group_log result terminal")
        l2_outputs = l2_outputs.mix(lgcrct_l2)
    }

    finalize_l2(l2_outputs)
}
