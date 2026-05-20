// EV2 Module Aliases
// All processes are aliases of IOInterface from the shared toolbox workflow wrapper.

// Import wrapper finalization process
include { finalize_participant } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { finalize_l2 } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EV2-specific
include { IOInterface as deap_bootstrap } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as deap_ingestor  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as study_l2_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as lgcrct_l2_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as labels_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as plv_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as baseline_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_label_exporter } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// File extraction (signal pointer → specific parquet)
include { IOInterface as eeg_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as bvp_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as bvp_baseline_file_finder } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as labels_file_finder      } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EEG analysis
include { IOInterface as eeg_psd_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_fai_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EDA analysis
include { IOInterface as eda_amplitude_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// HRV analysis (BVP peaks → RMSSD)
include { IOInterface as bvp_peak_detector    } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_interval_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// Correlation (physiology × continuous ratings)
include { IOInterface as fai_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as plv_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as plv_fai_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EEG-HRV Phase-Locking Value
include { IOInterface as plv_analyzer             } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
