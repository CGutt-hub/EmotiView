// EV2 Module Aliases
// All processes are aliases of IOInterface from the shared toolbox workflow wrapper.

// Import wrapper finalization process
include { finalize_participant } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { finalize_l2 } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EV2-specific
include { IOInterface as deap_bootstrap } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as deap_ingestor  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as lgcrct_l2_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as labels_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// File extraction (signal pointer → specific parquet)
include { IOInterface as eeg_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as bvp_file_finder         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as labels_file_finder      } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EEG analysis
include { IOInterface as eeg_spectrum_analyzer        } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_spectrum_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_fai_psd_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_fai_analyzer     } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_windowing_processor  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
// EEG ROI band-power (Frontal + Parietal, alpha/beta/theta)
include { IOInterface as eeg_roi_psd_analyzer         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd_all_file_finder  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_row_filter       } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_row_filter  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_label_binner } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_label_binner } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_bootstrap_analyzer  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_bootstrap_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_bs_file_finder  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_bs_file_finder } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_bs_concatenating_processor  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_bs_concatenating_processor } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_bootstrap_result_collector  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_bootstrap_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EDA analysis
include { IOInterface as eda_amplitude_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// HRV analysis (BVP peaks → RMSSD)
include { IOInterface as bvp_peak_detector          } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_interval_analyzer      } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// Correlation (physiology × continuous ratings)
include { IOInterface as fai_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_correlation_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
// Multimodal EEG–HRV correlation (Spearman r, N=40 trials)
include { IOInterface as multimodal_correl_analyzer        } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as multimodal_correl_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// Per-condition bootstrap analysis (physiology binned by valence / arousal)
// label_binner merges per-trial physiology with DEAP labels and bins into
// valence_high / valence_low / arousal_high / arousal_low conditions.
// bootstrap_analyzer then computes CIs within each bin.
include { IOInterface as eda_label_binner  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_label_binner  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_label_binner  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_bootstrap_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_bootstrap_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_bootstrap_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
// Single file finders: collect all available bs*.parquet per participant in one pass
include { IOInterface as eda_bs_file_finder } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_bs_file_finder } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_bs_file_finder } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
// Concatenating processors: merge per-condition bs*.parquet into one multi-row bar chart parquet
include { IOInterface as eda_bs_concatenating_processor } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_bs_concatenating_processor } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_bs_concatenating_processor } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eda_bootstrap_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_bootstrap_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_bootstrap_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// L2 group-level ANOVA (toolbox anova_analyzer, multi-participant collected input)
include { IOInterface as eda_l2_anova_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as hrv_l2_anova_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as fai_l2_anova_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// Within-participant ANOVA (L1: compares valence/arousal bins across modalities)
include { IOInterface as anova_merger                   } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as anova_combined_analyzer        } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as anova_combined_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// EEG OLS contrasts (within-participant pairwise bin comparisons, e.g. arousal_high vs arousal_low)
include { IOInterface as eeg_frontal_ols_processor              } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_ols_processor             } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_contrast_processor         } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_contrast_processor        } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_frontal_contrast_result_collector  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_contrast_result_collector } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'

// L2 EEG band-power group ANOVA (mirrors EDA/HRV/FAI L2 analyses)
include { IOInterface as eeg_frontal_l2_anova_analyzer  } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as eeg_parietal_l2_anova_analyzer } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
include { IOInterface as l2_anova_consolidator } from '../../../AnalysisToolbox/gitatbx/bin/workflow_wrapper.nf'
