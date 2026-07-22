nextflow.enable.dsl=2

// =========================================================================
// 1. STANDARD-WORKFLOW-INFRASTRUKTUR & EVENT-GATES
// =========================================================================
include { data_discovery } from '../../../AnalysisToolbox/gitatbx/.bin/data_discovery.nf'
include { finalize_l1 }    from '../../../AnalysisToolbox/gitatbx/.bin/finalize_l1.nf'
include { finalize_l2 }    from '../../../AnalysisToolbox/gitatbx/.bin/finalize_l2.nf'

// =========================================================================
// 2. INFRASTRUKTUR-MODULE (HORIZONTAL-JOINS AUF PROBANDEN-EBENE)
// =========================================================================
include {
    NATIVE_JOIN as l1_wp1_anova_inputs_join; 
    NATIVE_JOIN as l1_wp2_correl_inputs_join; 
    NATIVE_JOIN as l1_wp3_fai_sam_inputs_join; 
    NATIVE_JOIN as l1_wp3_eda_sam_inputs_join; 
    NATIVE_JOIN as l1_wp3_hrv_sam_inputs_join; 
    NATIVE_JOIN as l1_wp4_fai_eda_inputs_join; 
    NATIVE_JOIN as l1_wp4_fai_hrv_inputs_join;
    NATIVE_JOIN as l1_wp5_val_inputs_join;
    NATIVE_JOIN as l1_wp5_arou_inputs_join
} from '../../../AnalysisToolbox/gitatbx/.bin/NATIVE_JOIN.nf'

// =========================================================================
// 3. INFRASTRUKTUR-MODULE (VERTIKAL-CONCATS ZUR KOHORTEN-AGGREGATION)
// =========================================================================
include {
    NATIVE_CONCAT as l2_wp1_cohort_anova_concat;
    NATIVE_CONCAT as l2_wp2_cohort_correl_concat;
    NATIVE_CONCAT as l2_wp3_cohort_fai_sam_concat;
    NATIVE_CONCAT as l2_wp3_cohort_eda_sam_concat;
    NATIVE_CONCAT as l2_wp3_cohort_hrv_sam_concat;
    NATIVE_CONCAT as l2_wp4_cohort_fai_eda_concat;
    NATIVE_CONCAT as l2_wp4_cohort_fai_hrv_concat;
    NATIVE_CONCAT as l2_wp5_cohort_val_concat;
    NATIVE_CONCAT as l2_wp5_cohort_arou_concat
} from '../../../AnalysisToolbox/gitatbx/.bin/NATIVE_CONCAT.nf' 

// =========================================================================
// 4. INFRASTRUKTUR-MODULE (DATEN-STAGING & KANAL-INITIALISIERUNG)
// =========================================================================
include {
    NATIVE_CHANNEL as stage_eeg_channel;
    NATIVE_CHANNEL as stage_eda_channel;
    NATIVE_CHANNEL as stage_hrv_channel;
    NATIVE_CHANNEL as stage_sam_channel
} from '../../../AnalysisToolbox/gitatbx/.bin/NATIVE_CHANNEL.nf'

// =========================================================================
// 5. WISSENSCHAFTLICHE CORE-PROCESSORS (REINE FEATURE-EXTRAKTION VIA IOINTERFACE)
// =========================================================================
include {
    NATIVE_MODULE as eeg_spectrum_analyzer;
    NATIVE_MODULE as eeg_psd_analyzer; 
    NATIVE_MODULE as eeg_fai_analyzer; 
    NATIVE_MODULE as eda_amplitude_analyzer;
    NATIVE_MODULE as bvp_peak_detector; 
    NATIVE_MODULE as hrv_interval_analyzer;
    
    NATIVE_MODULE as l1_wp1_anova_combined_analyzer; 
    NATIVE_MODULE as l1_wp2_correl_analyzer; 
    
    NATIVE_MODULE as l1_wp3_correl_analyzer_fai; 
    NATIVE_MODULE as l1_wp3_correl_analyzer_eda; 
    NATIVE_MODULE as l1_wp3_correl_analyzer_hrv; 
    
    NATIVE_MODULE as l1_wp4_correl_analyzer_fai_eda; 
    NATIVE_MODULE as l1_wp4_correl_analyzer_fai_hrv;
    
    NATIVE_MODULE as l1_wp5_regress_analyzer_val;
    NATIVE_MODULE as l1_wp5_regress_analyzer_arousal;
    
    NATIVE_MODULE as l2_wp1_lmm_analyzer;
    NATIVE_MODULE as l2_wp2_lmm_analyzer;
    NATIVE_MODULE as l2_wp3_lmm_analyzer_val_fai; 
    NATIVE_MODULE as l2_wp3_lmm_analyzer_val_eda; 
    NATIVE_MODULE as l2_wp3_lmm_analyzer_val_hrv; 
    NATIVE_MODULE as l2_wp3_lmm_analyzer_arousal_fai; 
    NATIVE_MODULE as l2_wp3_lmm_analyzer_arousal_eda; 
    NATIVE_MODULE as l2_wp3_lmm_analyzer_arousal_hrv; 
    NATIVE_MODULE as l2_wp4_lmm_analyzer_fai_eda;
    NATIVE_MODULE as l2_wp4_lmm_analyzer_fai_hrv;
    NATIVE_MODULE as l2_wp5_lmm_analyzer_val;
    NATIVE_MODULE as l2_wp5_lmm_analyzer_arou
} from '../../../AnalysisToolbox/gitatbx/.bin/NATIVE_MODULE.nf'

workflow {

    // =========================================================================
    // STEP 1: Participant Discovery
    // =========================================================================
    data_discovery(params.input_dir, params.output_dir, params.participant_pattern)
    
    participant_context = data_discovery.out

    // =========================================================================
    // STEP 2: Pure Declarative Data Staging via NATIVE_CHANNEL
    // 100% konform: Reicht das unberührte Entdeckungs-Tuple direkt weiter!
    // =========================================================================
    eeg_epochs_ch = stage_eeg_channel( participant_context, "*_eeg.parquet", "EEG" ).staged_channel
    eda_epochs_ch = stage_eda_channel( participant_context, "*_eda.parquet", "EDA" ).staged_channel
    hrv_epochs_ch = stage_hrv_channel( participant_context, "*_hrv.parquet", "HRV" ).staged_channel
    sam_ch        = stage_sam_channel( participant_context, "*_sam.parquet", "SAM" ).staged_channel

    // =========================================================================
    // STEP 3: Core Feature Extraction (Einspeisung in die wissenschaftlichen Module)
    // 💡 OPTIMIZED: Replaced tuple() with literal arrays [...] to prevent ReferenceQueue leaks
    // =========================================================================
    eeg_spectrum_out = eeg_spectrum_analyzer(params.python_exe, params.spectrum_analyzer_script, ["eeg", "spectrum", "none"], eeg_epochs_ch).signal
    eeg_fai_out      = eeg_fai_analyzer(params.python_exe, params.asymmetry_analyzer_script, ["eeg", "fai", "none"], eeg_epochs_ch).signal
    eda_amplitude_out= eda_amplitude_analyzer(params.python_exe, params.amplitude_analyzer_script, ["eda", "amplitude", "none"], eda_epochs_ch).signal
    hrv_rmssd_out    = hrv_interval_analyzer(params.python_exe, params.interval_analyzer_script, ["hrv", "rmssd", "none"], hrv_epochs_ch).signal

    // =========================================================================
    // LEVEL 1 DOWNSTREAM WORKPACKAGES: Pure Declarative Infinite Intersections
    // 💡 THE PERFECT CASE: Pass an arbitrary amount of signals (2, 3, or 5) cleanly!
    // =========================================================================

    // --- WP1 L1: Combined ANOVA (3-Wege-Feature-Merge — Passes 3 active queues) ---
    wp1_joined = l1_wp1_anova_inputs_join( [eeg_fai_out, eda_amplitude_out, hrv_rmssd_out] )
    wp1_anova  = l1_wp1_anova_combined_analyzer(
        params.python_exe, params.anova_analyzer_script, 
        ["tables", "none", "auto condition false None source terminal"], wp1_joined.merged_matrix
    ).signal
    
    // --- WP2 L1: Multimodal Correlation (2-Wege-Feature-Merge — NO EMPTY [] CLUTTER!) ---
    wp2_joined = l1_wp2_correl_inputs_join( [eeg_spectrum_out, hrv_rmssd_out] )
    wp2_correl = l1_wp2_correl_analyzer(
        params.python_exe, params.correl_analyzer_script, 
        ["tables", "none", "None terminal"], wp2_joined.merged_matrix
    ).signal

    // --- WP3 L1: Modality-Dimension Associations (Signal vs SAM — 2-Wege-Merges) ---
    wp3_fai_sam_joined = l1_wp3_fai_sam_inputs_join( [eeg_fai_out, sam_ch] )
    wp3_fai_sam_correl = l1_wp3_correl_analyzer_fai(
        params.python_exe, params.correlation_analyzer_script, 
        ["results", "bar", "--correlate-ratings valence,arousal None spearman terminal result"], wp3_fai_sam_joined.merged_matrix
    ).signal
    
    wp3_eda_sam_joined = l1_wp3_eda_sam_inputs_join( [eda_amplitude_out, sam_ch] )
    wp3_eda_sam_correl = l1_wp3_correl_analyzer_eda(
        params.python_exe, params.correlation_analyzer_script, 
        ["results", "bar", "--correlate-ratings valence,arousal None spearman terminal result"], wp3_eda_sam_joined.merged_matrix
    ).signal
    
    wp3_hrv_sam_joined = l1_wp3_hrv_sam_inputs_join( [hrv_rmssd_out, sam_ch] )
    wp3_hrv_sam_correl = l1_wp3_correl_analyzer_hrv(
        params.python_exe, params.correlation_analyzer_script, 
        ["results", "bar", "--correlate-ratings valence,arousal None spearman terminal result"], wp3_hrv_sam_joined.merged_matrix
    ).signal

    // --- WP4 L1: Signal-Signal Interactions (Central Asymmetry vs. Autonomic Metrics) ---
    wp4_fai_eda_joined = l1_wp4_fai_eda_inputs_join( [eeg_fai_out, eda_amplitude_out] )
    wp4_fai_eda_correl = l1_wp4_correl_analyzer_fai_eda(
        params.python_exe, params.correlation_analyzer_script, 
        ["results", "scatter", "None terminal result"], wp4_fai_eda_joined.merged_matrix
    ).signal
    
    wp4_fai_hrv_joined = l1_wp4_fai_hrv_inputs_join( [eeg_fai_out, hrv_rmssd_out] )
    wp4_fai_hrv_correl = l1_wp4_correl_analyzer_fai_hrv(
        params.python_exe, params.correlation_analyzer_script, 
        ["results", "scatter", "None terminal result"], wp4_fai_hrv_joined.merged_matrix
    ).signal

    // --- WP5 L1: Tracking Rating Flux (Valence & Arousal Flux Estimations) ---
    wp5_val_joined  = l1_wp5_val_inputs_join( [eeg_fai_out, sam_ch] )
    wp5_regress_val = l1_wp5_regress_analyzer_val(
        params.python_exe, params.ols_processor_script, 
        ["results", "line", "flux_valence FAI group_log result"], wp5_val_joined.merged_matrix
    ).signal
    
    wp5_arou_joined  = l1_wp5_arou_inputs_join( [eeg_fai_out, sam_ch] )
    wp5_regress_arou = l1_wp5_regress_analyzer_arousal(
        params.python_exe, params.ols_processor_script, 
        ["results", "line", "flux_arousal FAI group_log result"], wp5_arou_joined.merged_matrix
    ).signal
    
    // =========================================================================
    // FIRST-LEVEL GATEWAY: JVM Barrier-Management im RAM
    // =========================================================================
    l1_final_modules = [ wp1_anova, wp2_correl, wp3_fai_sam_correl, wp3_eda_sam_correl, wp3_hrv_sam_correl, wp4_fai_eda_correl, wp4_fai_hrv_correl, wp5_regress_val, wp5_regress_arou ]
    finalize_l1( l1_final_modules, 9, participant_context )

    // =========================================================================
    // LEVEL 2: Cohort Aggregation & Second-Level Mixed Linear Models (LMM)
    // 💡 ULTRA-STREAMLINED: Clean channel inputs matching Level 1 style
    // =========================================================================

    // --- WP1 L2: Cohort Aggregation & LMM Analysis ---
    cohort_wp1 = l2_wp1_cohort_anova_concat( wp1_anova, "wp1_anova_binned" )
    l2_wp1     = l2_wp1_lmm_analyzer( params.python_exe, params.lmm_analyzer_script, ["results", "none", "condition FAI group_log result"], cohort_wp1.cohort_matrix )

    // --- WP2 L2: Cohort Aggregation & LMM Analysis ---
    cohort_wp2 = l2_wp2_cohort_correl_concat( wp2_correl, "wp2_correl_binned" )
    l2_wp2     = l2_wp2_lmm_analyzer( params.python_exe, params.lmm_analyzer_script, ["results", "none", "condition correl group_log result"], cohort_wp2.cohort_matrix )
    
    // --- WP3 L2: Cohort Aggregation (Modality-Dimension Profiles) ---
    cohort_wp3_fai = l2_wp3_cohort_fai_sam_concat( wp3_fai_sam_correl, "wp3_fai_binned" )
    cohort_wp3_eda = l2_wp3_cohort_eda_sam_concat( wp3_eda_sam_correl, "wp3_eda_binned" )
    cohort_wp3_hrv = l2_wp3_cohort_hrv_sam_concat( wp3_hrv_sam_correl, "wp3_hrv_binned" )
    
    l2_wp3_v_f = l2_wp3_lmm_analyzer_val_fai( params.python_exe, params.lmm_analyzer_script, ["results", "none", "valence FAI group_log result"], cohort_wp3_fai.cohort_matrix )
    l2_wp3_v_e = l2_wp3_lmm_analyzer_val_eda( params.python_exe, params.lmm_analyzer_script, ["results", "none", "valence eda group_log result"], cohort_wp3_eda.cohort_matrix )
    l2_wp3_v_h = l2_wp3_lmm_analyzer_val_hrv( params.python_exe, params.lmm_analyzer_script, ["results", "none", "valence hrv group_log result"], cohort_wp3_hrv.cohort_matrix )
    
    l2_wp3_a_f = l2_wp3_lmm_analyzer_arousal_fai( params.python_exe, params.lmm_analyzer_script, ["results", "none", "arousal FAI group_log result"], cohort_wp3_fai.cohort_matrix )
    l2_wp3_a_e = l2_wp3_lmm_analyzer_arousal_eda( params.python_exe, params.lmm_analyzer_script, ["results", "none", "arousal eda group_log result"], cohort_wp3_eda.cohort_matrix )
    l2_wp3_a_h = l2_wp3_lmm_analyzer_arousal_hrv( params.python_exe, params.lmm_analyzer_script, ["results", "none", "arousal hrv group_log result"], cohort_wp3_hrv.cohort_matrix )
    
    // --- WP4 L2: Cohort Aggregation & LMM Analysis (Signal vs Signal) ---
    cohort_wp4_eda    = l2_wp4_cohort_fai_eda_concat( wp4_fai_eda_correl, "wp4_eda_binned" )
    l2_wp4_e          = l2_wp4_lmm_analyzer_fai_eda( params.python_exe, params.lmm_analyzer_script, ["results", "none", "FAI eda group_log result"], cohort_wp4_eda.cohort_matrix )

    cohort_wp4_hrv    = l2_wp4_cohort_fai_hrv_concat( wp4_fai_hrv_correl, "wp4_hrv_binned" )
    l2_wp4_h          = l2_wp4_lmm_analyzer_fai_hrv( params.python_exe, params.lmm_analyzer_script, ["results", "none", "FAI hrv group_log result"], cohort_wp4_hrv.cohort_matrix )
    
    // --- WP5 L2: Multimodal Machine Learning Classifier ---
    cohort_wp5_val    = l2_wp5_cohort_val_concat( wp5_regress_val, "wp5_val_binned" )
    l2_wp5_v          = l2_wp5_lmm_analyzer_val( params.python_exe, params.multimodal_classifier_script, ["results", "none", "flux_valence FAI group_log result"], cohort_wp5_val.cohort_matrix )
    
    cohort_wp5_arou    = l2_wp5_cohort_arou_concat( wp5_regress_arou, "wp5_arou_binned" )
    l2_wp5_a           = l2_wp5_lmm_analyzer_arou( params.python_exe, params.multimodal_classifier_script, ["results", "none", "flux_arousal FAI group_log result"], cohort_wp5_arou.cohort_matrix )

    // =========================================================================
    // SECOND-LEVEL GATEWAY
    // =========================================================================
    l2_final_modules = [ l2_wp1.signal, l2_wp2.signal, l2_wp3_v_f.signal, l2_wp3_v_e.signal, l2_wp3_v_h.signal, l2_wp3_a_f.signal, l2_wp3_a_e.signal, l2_wp3_a_h.signal, l2_wp4_e.signal, l2_wp4_h.signal, l2_wp5_v.signal, l2_wp5_a.signal ]
    finalize_l2( l2_final_modules )


}