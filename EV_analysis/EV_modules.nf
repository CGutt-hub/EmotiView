
// Standard Modules - all from AnalysisToolbox
include { participant_discovery; finalize_participant; finalize_l2 } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'

// EmotiView pipeline Modules - all using IOInterface from AnalysisToolbox
// readers
include { IOInterface as txt_reader } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as xdf_reader } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// extractor
include { IOInterface as extracting_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// rejection (artifact removal)
include { IOInterface as eda_rejection_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv_rejection_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// filterers
include { IOInterface as eda_filtering_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ecg_filtering_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_filtering_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_filtering_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// peak detector
include { IOInterface as peak_detection_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// referencer
include { IOInterface as referencing_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// transforms
include { IOInterface as log_transform_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as tddr_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as linear_transform_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as regression_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// event processor
include { IOInterface as events_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// tree builder
include { IOInterface as tree_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// epochers
include { IOInterface as eda_epoching_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ecg_epoching_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_epoching_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_epoching_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// analyzers
include { IOInterface as ic_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as interval_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as amplitude_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_hbc1_amplitude_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_hbc2_amplitude_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_hbc3_amplitude_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// EEG PSD analysis — explicit processing chain
// Step 2: spatial ROI averaging (average electrodes within Frontal/Parietal patches)
include { IOInterface as eeg_roi_group_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Step 3: condition separation (file finders for per-condition ROI time-series)
include { IOInterface as eeg_roi1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Step 4: frequency band averaging (psd_analyzer per condition)
include { IOInterface as eeg_roi_psd1_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd2_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd3_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd1_epoch_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd2_epoch_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_roi_psd3_epoch_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Step 5: outlier detection per condition
include { IOInterface as eeg_psd1_outlier_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_psd2_outlier_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_psd3_outlier_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Step 6: concatenate clean conditions + bootstrap CIs (alpha, beta, theta)
include { IOInterface as eeg_psd_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_alpha_bootstrap_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_beta_bootstrap_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_theta_bootstrap_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// PSD per-band file finders (extract per-condition bootstrap files)
include { IOInterface as eeg_alpha_bs1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_alpha_bs2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_alpha_bs3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_beta_bs1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_beta_bs2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_beta_bs3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_theta_bs1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_theta_bs2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_theta_bs3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// PSD per-band + cross-band concatenation
include { IOInterface as eeg_alpha_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_beta_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_theta_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_psd_result_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// FAI (F3–F4 asymmetry): channel-mode PSD, separate from ROI chain
include { IOInterface as psd_fai_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai1_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai2_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai3_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as group_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as bootstrap_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as panas_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as bisbas_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as sam_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be7_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea11_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// concatenators
include { IOInterface as sam_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be7_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea11_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// plot data merger (generic - combines plot-ready data from multiple sources)
include { IOInterface as fai_merger } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// relative normalizers (baseline normalization to NEU condition)
include { IOInterface as hrv_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fai_combined_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// file finders
include { IOInterface as aux_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ecg_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as trigger_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_cleaned_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as sam1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as sam2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as sam3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be71_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be72_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be73_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea111_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea112_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea113_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd1_raw_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd2_raw_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as psd3_raw_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// fnirs agg file finders: extract the actual data file from amplitude_analyzer signal output
include { IOInterface as fnirs_hbc1_agg_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_hbc2_agg_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_hbc3_agg_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// fnirs asymmetry input file finders (separate from fnirs concat file finders)
include { IOInterface as fnirs_asym_input1_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym_input2_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym_input3_file_finder } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// fnirs asymmetry (hemispheric R-L)
include { IOInterface as fnirs_asym1_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym2_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym3_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_asym_relative_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// physio combined concatenation (HRV + EDA → single plot)
include { IOInterface as physio_concatenating_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// result collectors (clean-named copies into results/ folder)
include { IOInterface as panas_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as bisbas_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as sam_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as be7_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as ea11_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eda_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_psd_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_contrast_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_fai_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as fnirs_fai_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as physio_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// L2 result collectors
include { IOInterface as condition_profile_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as cross_modal_correl_result_collector } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// EEG within-participant statistics: OLS condition betas + pairwise contrasts (t-values)
include { IOInterface as eeg_psd_ols_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as eeg_psd_contrast_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Cross-modal consistency: OLS betas for EDA and HRV (pre-bootstrap epoch level)
include { IOInterface as eda_ols_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as hrv_ols_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
// Pivot all three OLS outputs wide and correlate pairwise (Pearson r per signal pair, per participant)
include { IOInterface as condition_profile_processor } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'
include { IOInterface as condition_profile_correl_analyzer } from '../../AnalysisToolbox/Python/utils/workflow_wrapper.nf'


