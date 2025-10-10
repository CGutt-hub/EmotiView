#!/usr/bin/env nextflow

// =========================================
// EmotiView Nextflow Pipeline
// =========================================

// ------- PARAMETERS -----------

// Base parameters
params.participant_pattern = 'EV_*' // Wildcard for participant files
params.input_dir = '/mnt/emotiview/rawData' // Input data directory
params.output_dir = '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_results' // Output results directory
params.python_exe = '/home/gutt/EV_venv/bin/python3' // Python executable for all modules

// Python script paths
params.xdf_reader_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/xdf_reader.py'
params.txt_reader_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/readers/txt_reader.py'
params.eeg_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/eeg_preprocessor.py'
params.ecg_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/ecg_preprocessor.py'
params.eda_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/eda_preprocessor.py'
params.fnirs_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/fnirs_preprocessor.py'
params.event_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/event_preprocessor.py'
params.quest_preprocessor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/preprocessors/quest_preprocessor.py'
params.epoching_processor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/processors/epoching_processor.py'
params.merging_processor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/processors/merging_processor.py'
params.filtering_processor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/processors/filtering_processor.py'
params.normalization_processor_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/processors/normalization_processor.py'
params.ic_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/ic_analyzer.py'
params.hrv_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/hrv_analyzer.py'
params.scr_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/scr_analyzer.py'
params.glm_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/glm_analyzer.py'
params.psd_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/psd_analyzer.py'
params.plv_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/plv_analyzer.py'
params.fai_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/fai_analyzer.py'
params.quest_analyzer_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/analyzers/quest_analyzer.py'
params.plotter_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/plotter.py'
params.git_sync_script = '/mnt/emotiview/repoShaggy/PsyAnalysisToolbox/Python/utils/git_sync.py'
params.stream_extractor_script = '/mnt/emotiview/repoShaggy/EmotiViewPrivate/EV_analysis/EV_modules.py'

// ========== ANALYSIS PARAMETERS (Only Used in Workflow) ==========
// Basic parameters
params.sfreq = 1000 // For HRV and event analysis
params.trigger_condition_map = '1:neutral,2:positive,3:negative' // Event mapping
params.ch_types_map = "{\"fnirs\": \"fnirs\"}" // Channel types mapping for GLM
params.contrasts_config = "{\"pos_vs_neu\": [1, 0, -1], \"neg_vs_neu\": [0, 1, -1]}" // GLM contrasts
params.fai_band_name = "alpha" // Frequency band for FAI analysis
params.electrode_pairs = "[[\"F3\", \"F4\"], [\"F7\", \"F8\"]]" // Electrode pairs for FAI

// Questionnaire analysis patterns (used by questionnaire analyzer)
params.questionnaire_patterns = [
    ea11: [
        question: 'ea11',
        response: 'ea11.Choice1.Value', 
        scale: 'leftScaleWord|rightScaleWord'
    ],
    panas: [
        question: 'panas',
        response: 'panas.Choice1.Value',
        scale: 'Wordp.*'
    ],
    bis_bas: [
        question: 'bis',
        response: 'bisBas.Choice1.Value',
        scale: 'Wordb.*'
    ],
    sam: [
        question: 'SAM',
        response: 'sam.Choice1.Value',
        scale: 'leftScaleWord|rightScaleWord'
    ],
    be7: [
        question: 'be7',
        response: 'be7.Choice1.Value',
        scale: 'leftScaleWord|rightScaleWord'
    ]
]

// Preprocessing filters (used by preprocessor scripts)
params.eeg_l_freq = 0.5 
params.eeg_h_freq = 40.0 
params.eeg_reference = 'average'
params.fnirs_l_freq = 0.01 
params.fnirs_h_freq = 0.1 
params.fnirs_short_reg = 1 
params.eda_l_freq = 0.05 
params.eda_h_freq = 5.0
params.ecg_l_freq = 0.5
params.ecg_h_freq = 50.0

// Analysis parameters (used by analyzer scripts)
params.bands_config = '{"alpha": [8, 13], "beta": [13, 30]}' // For PSD/PLV
params.fai_band_name = 'alpha' // For FAI 
params.electrode_pairs = 'F3-F4' // For FAI

// Column names for stream extraction
params.ecg_col = 'ekg'
params.eda_col = 'eda' 
params.trigger_col = 'triggerStream'

// File encoding parameter for txt files (empty string = auto-detect)
params.txt_encoding = 'utf-16'


// ----------- PROCESS IMPORTS -----------

// Ensure output directory exists for each participant
process make_output_folder {
    input:
        val output_folder
    output:
        path output_folder, emit: output_folder_ready
    script:
        """
        mkdir -p ${output_folder}
        echo "Output folder created: ${output_folder}"
        """
}

// Select specific streams from XDF output
process select_streams {
    input:
        path streams_folder
    output:
        path "*_stream1.parquet", emit: fnirs_stream
        path "*_stream4.parquet", emit: aux_stream
    script:
        """
        cp ${streams_folder}/*_stream1.parquet .
        cp ${streams_folder}/*_stream4.parquet .
        """
}

// Select specific channels from XDF output
process select_channels {
    input:
        path channel_folder
    output:
        path "*_ecgExtr.parquet", emit: ecg_stream
        path "*_eegExtr.parquet", emit: eeg_stream
        path "*_edaExtr.parquet", emit: eda_stream
        path "*_triggerExtr.parquet", emit: trigger_stream
    script:
        """
        cp ${channel_folder}/*_ecgExtr.parquet .
        cp ${channel_folder}/*_eegExtr.parquet .
        cp ${channel_folder}/*_edaExtr.parquet .
        cp ${channel_folder}/*_triggerExtr.parquet .
        """
}

// ---- Dispatcher includes for each module ----
include { python_dispatcher as xdf_reader } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as txt_reader } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as eeg_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as ecg_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as eda_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as fnirs_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as event_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as quest_preprocessor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as fnirs_epoching } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as eda_epoching } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as ecg_epoching } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as eeg_epoching } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as plv_merging_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as psd_merging_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as filtering_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as correlation_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as statistical_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as normalization_processor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as ic_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as hrv_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as scr_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as glm_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as psd_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as plv_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as fai_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as quest_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as git_sync } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as stream_extractor } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'

// Questionnaire analyzer aliases
include { python_dispatcher as ea11_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as panas_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as bis_bas_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as sam_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as be7_analyzer } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'

// Plotter aliases using dispatcher
include { python_dispatcher as ic_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as hrv_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as scr_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as glm_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as psd_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as fai_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as plv_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as ea11_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as panas_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as bis_bas_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as sam_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'
include { python_dispatcher as be7_plotter } from '../../PsyAnalysisToolbox/Python/python_dispatcher.nf'



// ----------- WORKFLOW DEFINITION -----------

workflow {

    // ----------- PARTICIPANT LEVEL ANALYSIS -----------

    // Process existing participants AND watch for new ones (skip already processed)
    existing_participants = Channel.fromPath("${params.input_dir}/${params.participant_pattern}", type: 'dir')
    new_participants = Channel.watchPath("${params.input_dir}").filter { it.name.matches(/EV_.*/) && it.isDirectory() }
    raw_participant = existing_participants.mix(new_participants)
        .filter { participant_dir ->
            def participant_id = participant_dir.getBaseName()
            def result_dir = file("${params.output_dir}/${participant_id}")
            !result_dir.exists() || result_dir.list().size() == 0  // Process if no results folder or empty
        }

    // Create participant_id from watched folders
    participant_id = raw_participant.map { it.getBaseName() }.view { println "[participant_id] ${it}" }

    // Map xdf_file, txt_file, and output_folder from participant_id
    xdf_file = participant_id.map {"${params.input_dir}/${it}/${it}.xdf" }.view { println "[xdf_file] ${it}" }
    txt_file = participant_id.map { "${params.input_dir}/${it}/${it}.txt" }.view { println "[txt_file] ${it}" }
    output_folder = participant_id.map { "${params.output_dir}/${it}" }.view { println "[output_folder] ${it}" }

    // Step 1: Read files
    xdf_read = xdf_reader(params.python_exe, params.xdf_reader_script, xdf_file, ".")
    txt_read = txt_reader(params.python_exe, params.txt_reader_script, txt_file, params.txt_encoding)

    // Step 2: Extract streams
    (fnirs_stream, aux_stream) = select_streams(xdf_read)
    stream_extract = stream_extractor(params.python_exe, params.stream_extractor_script, aux_stream, "${params.ecg_col} ${params.eda_col} ${params.trigger_col}")
    (ecg_stream, eeg_stream, eda_stream, trigger_stream) = select_channels(stream_extract)
    
    // Step 3: Preprocess
    fnirs_preproc = fnirs_preprocessor(params.python_exe, params.fnirs_preprocessor_script, fnirs_stream, "${params.fnirs_l_freq} ${params.fnirs_h_freq} ${params.fnirs_short_reg}")
    eeg_preproc = eeg_preprocessor(params.python_exe, params.eeg_preprocessor_script, eeg_stream, "${params.eeg_l_freq} ${params.eeg_h_freq} ${params.eeg_reference}")
    event_preproc = event_preprocessor(params.python_exe, params.event_preprocessor_script, trigger_stream, "${params.sfreq} ${params.trigger_condition_map}")
    eda_preproc = eda_preprocessor(params.python_exe, params.eda_preprocessor_script, eda_stream, "${params.eda_l_freq} ${params.eda_h_freq}")
    ecg_preproc = ecg_preprocessor(params.python_exe, params.ecg_preprocessor_script, ecg_stream, "${params.ecg_l_freq} ${params.ecg_h_freq}")
    
    // Step 3.2: Questionnaire preprocessing
    quest_preproc = quest_preprocessor(params.python_exe, params.quest_preprocessor_script, txt_read, "${params.txt_encoding} .")
    
    // Step 3.1: In-Between Analyze
    ic_analyze = ic_analyzer(params.python_exe, params.ic_analyzer_script, eeg_preproc, "")

    // Step 4: Processing
    fnirs_epoching = fnirs_epoching(params.python_exe, params.epoching_processor_script, fnirs_preproc, event_preproc)
    eda_epoching = eda_epoching(params.python_exe, params.epoching_processor_script, eda_preproc, event_preproc)
    ecg_epoching = ecg_epoching(params.python_exe, params.epoching_processor_script, ecg_preproc, event_preproc)
    eeg_epoching = eeg_epoching(params.python_exe, params.epoching_processor_script, ic_analyze, event_preproc)
    plv_merge = plv_merging_processor(params.python_exe, params.merging_processor_script, eeg_epoching, "time ${eda_epoching} ${eeg_epoching} ${ecg_epoching}")
    psd_merge = psd_merging_processor(params.python_exe, params.merging_processor_script, eeg_epoching, "time ${fnirs_epoching} ${eeg_epoching}")

    // Step 5: Analysis
    // Questionnaire analyses using parameter-based patterns (cleaner than command-line strings)
    ea11_analyze = ea11_analyzer(params.python_exe, params.quest_analyzer_script, quest_preproc, "${params.questionnaire_patterns.ea11.question} ${params.questionnaire_patterns.ea11.response} ${params.questionnaire_patterns.ea11.scale}")
    panas_analyze = panas_analyzer(params.python_exe, params.quest_analyzer_script, quest_preproc, "${params.questionnaire_patterns.panas.question} ${params.questionnaire_patterns.panas.response} ${params.questionnaire_patterns.panas.scale}")
    bis_bas_analyze = bis_bas_analyzer(params.python_exe, params.quest_analyzer_script, quest_preproc, "${params.questionnaire_patterns.bis_bas.question} ${params.questionnaire_patterns.bis_bas.response} ${params.questionnaire_patterns.bis_bas.scale}")
    sam_analyze = sam_analyzer(params.python_exe, params.quest_analyzer_script, quest_preproc, "${params.questionnaire_patterns.sam.question} ${params.questionnaire_patterns.sam.response} ${params.questionnaire_patterns.sam.scale}")
    be7_analyze = be7_analyzer(params.python_exe, params.quest_analyzer_script, quest_preproc, "${params.questionnaire_patterns.be7.question} ${params.questionnaire_patterns.be7.response} ${params.questionnaire_patterns.be7.scale}")

    // Physiological analyses
    hrv_analyze = hrv_analyzer(params.python_exe, params.hrv_analyzer_script, ecg_epoching, "${params.sfreq}")
    scr_analyze = scr_analyzer(params.python_exe, params.scr_analyzer_script, eda_preproc, "")
    glm_analyze = glm_analyzer(params.python_exe, params.glm_analyzer_script, fnirs_epoching, "${params.sfreq} ${params.ch_types_map} ${params.contrasts_config}")
    psd_analyze = psd_analyzer(params.python_exe, params.psd_analyzer_script, psd_merge, "${params.bands_config}")
    fai_analyze = fai_analyzer(params.python_exe, params.fai_analyzer_script, psd_analyze, "${params.fai_band_name} ${params.electrode_pairs}")
    plv_analyze = plv_analyzer(params.python_exe, params.plv_analyzer_script, plv_merge, "${params.bands_config}")

    // ===== OPTIONAL: PROCESSOR CHAINING EXAMPLES =====
    // Example: Filter processed signals from analyzers for further analysis
    // filtered_hrv = filtering_processor(params.python_exe, params.filtering_processor_script, hrv_analyze, "band 0.04,0.15 1000 rr_interval")
    
    // Example: Normalize physiological data for group analysis
    // norm_psd = normalization_processor(params.python_exe, params.normalization_processor_script, psd_analyze, "zscore power")

    // Step 6: Plotting using corrected plotter (reads plot_type from parquet metadata)
    // Combine analysis results with output folders for plotting
    plot_args_ic = ic_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_hrv = hrv_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_scr = scr_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_glm = glm_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_psd = psd_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_fai = fai_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_plv = plv_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_ea11 = ea11_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_panas = panas_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_bis_bas = bis_bas_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_sam = sam_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }
    plot_args_be7 = be7_analyze.combine(output_folder).map { analysis, outdir -> [analysis, "${outdir}"] }

    // Create plots using the generic plotter (no plot type needed - reads from parquet metadata)
    ic_plotter(params.python_exe, params.plotter_script, plot_args_ic.map{it[0]}, plot_args_ic.map{it[1]}).set { ic_plot_ch }
    hrv_plotter(params.python_exe, params.plotter_script, plot_args_hrv.map{it[0]}, plot_args_hrv.map{it[1]}).set { hrv_plot_ch }
    scr_plotter(params.python_exe, params.plotter_script, plot_args_scr.map{it[0]}, plot_args_scr.map{it[1]}).set { scr_plot_ch }
    glm_plotter(params.python_exe, params.plotter_script, plot_args_glm.map{it[0]}, plot_args_glm.map{it[1]}).set { glm_plot_ch }
    psd_plotter(params.python_exe, params.plotter_script, plot_args_psd.map{it[0]}, plot_args_psd.map{it[1]}).set { psd_plot_ch }
    fai_plotter(params.python_exe, params.plotter_script, plot_args_fai.map{it[0]}, plot_args_fai.map{it[1]}).set { fai_plot_ch }
    plv_plotter(params.python_exe, params.plotter_script, plot_args_plv.map{it[0]}, plot_args_plv.map{it[1]}).set { plv_plot_ch }
    ea11_plotter(params.python_exe, params.plotter_script, plot_args_ea11.map{it[0]}, plot_args_ea11.map{it[1]}).set { ea11_plot_ch }
    panas_plotter(params.python_exe, params.plotter_script, plot_args_panas.map{it[0]}, plot_args_panas.map{it[1]}).set { panas_plot_ch }
    bis_bas_plotter(params.python_exe, params.plotter_script, plot_args_bis_bas.map{it[0]}, plot_args_bis_bas.map{it[1]}).set { bis_bas_plot_ch }
    sam_plotter(params.python_exe, params.plotter_script, plot_args_sam.map{it[0]}, plot_args_sam.map{it[1]}).set { sam_plot_ch }
    be7_plotter(params.python_exe, params.plotter_script, plot_args_be7.map{it[0]}, plot_args_be7.map{it[1]}).set { be7_plot_ch }

    // Step 7: Final git sync (collect all plots)
    all_plots = 
        ic_plot_ch.mix(
        hrv_plot_ch,
        scr_plot_ch,
        glm_plot_ch,
        psd_plot_ch,
        fai_plot_ch,
        plv_plot_ch,
        ea11_plot_ch,
        panas_plot_ch,
        bis_bas_plot_ch,
        sam_plot_ch,
        be7_plot_ch
    )
    
    // Sync all results to git repository
    git_sync(params.python_exe, params.git_sync_script, all_plots, "")

    // ----------- GROUP LEVEL ANALYSIS -----------

    // Watch EV_results directory for group level analysis
    // analyzed_participant = Channel.watchPath( "${params.output_dir}/${params.participant_pattern}" )


}