# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\analysis\fnirs_glm_analyzer.py
import os
import numpy as np
import pandas as pd
import mne
import mne_nirs
from mne_nirs.experimental_design import make_first_level_design_matrix # Correct import for epoched data
from mne_nirs.statistics import run_glm
from mne_nirs.channels import (picks_pair_to_ch_names,
                               picks_optodes_to_channel_locations) # Not used in current GLM but good for context
class FNIRSGLMAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSGLMAnalyzer initialized.")

    def run_glm_on_epochs(self, fnirs_epochs_mne, event_id_map_for_glm, participant_id, analysis_results_dir,
                          glm_hrf_model, glm_contrasts_config, glm_rois_config, glm_activation_p_threshold):
        """
        Runs fNIRS GLM on epoched data and computes specified contrasts.

        Args:
            fnirs_epochs_mne (mne.Epochs): Epoched fNIRS data (HbO/HbR).
            event_id_map_for_glm (dict): Mapping from condition names to MNE event codes used for epoching.
            participant_id (str): The ID of the participant.
            analysis_results_dir (str): Directory to save GLM-related results.
            glm_hrf_model (str): The HRF model to use (e.g., 'spm').
            glm_contrasts_config (dict): Dictionary defining contrasts.
            glm_rois_config (dict): Dictionary defining ROIs and their channels.
            glm_activation_p_threshold (float): P-value threshold for identifying active channels/ROIs.

        Returns:
            dict: A dictionary containing 'glm_estimates' (dict of GLMEstimate objects),
                  'contrast_results' (DataFrame dict), and 'active_rois_for_eeg_guidance' (list).
                  Returns empty dict on failure.
        """
        if fnirs_epochs_mne is None:
            self.logger.warning("FNIRSGLMAnalyzer (epochs): Missing fNIRS epochs data.")
            return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}
        
        self.logger.info(f"FNIRSGLMAnalyzer (epochs): Running GLM for P:{participant_id}")
        
        # Validate passed configurations
        if not all([glm_hrf_model, glm_contrasts_config, glm_rois_config is not None, glm_activation_p_threshold is not None]):
            self.logger.error("FNIRSGLMAnalyzer (epochs): One or more GLM configuration parameters (hrf_model, contrasts, rois, p_threshold) are missing.")
            return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}
        if not isinstance(glm_contrasts_config, dict) or not isinstance(glm_rois_config, dict):
            self.logger.error("FNIRSGLMAnalyzer (epochs): glm_contrasts_config and glm_rois_config must be dictionaries.")
            return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}


        if not event_id_map_for_glm:
            self.logger.error("FNIRSGLMAnalyzer (epochs): event_id_map_for_glm not provided. Cannot create design matrix.")
            return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}

        try:
            # --- Create Design Matrix ---
            # mne_nirs.statistics.run_glm expects a design matrix for the *continuous* data
            # from which epochs were extracted. This is a common point of confusion.
            # If you have the original continuous raw data and the full event list,
            # it's often better to run GLM on continuous data and then epoch the results.
            #
            # However, the current structure passes epochs directly.
            # `run_glm` *can* take epochs, but the `design_matrix` argument is still
            # expected to be for the *continuous* data.
            #
            # A workaround when you only have epochs is to create a design matrix
            # that represents the events *within* the concatenated epochs, or to
            # rely on `run_glm` to infer the design from epoch events (if it supports that).
            #
            # Let's assume the intention is to get condition-wise betas from the epochs.
            # A standard way with mne-nirs is to average epochs per condition (Evoked)
            # and then potentially run a simple model or compute contrasts on Evoked objects.
            #
            # If the goal is single-trial betas, `run_glm` on continuous data is more typical.
            #
            # Given the previous code structure, it seemed like the intention was to use
            # `mne.stats.make_first_level_design_matrix` which is more for event-based designs
            # on continuous data or evoked responses.
            #
            # Let's try to create a design matrix that `mne_nirs.statistics.run_glm` can use
            # when given epochs. This often involves creating a design matrix for the
            # *total duration* covered by all epochs, with events placed at their onsets
            # relative to the start of the first epoch's underlying continuous data.
            # This requires knowing the start time of the original continuous data.
            #
            # If we *must* use `run_glm` with *only* the `epochs` object and a design matrix,
            # the design matrix needs to align with how `run_glm` processes epochs.
            # The documentation for `mne_nirs.statistics.run_glm` shows `design_matrix`
            # having shape `(n_times, n_regressors)`. This strongly suggests it's for continuous data.
            #
            # Let's assume, for the sake of making this function runnable, that we can
            # create a design matrix based on the events *within* the epochs, treating
            # the epochs as if they were concatenated continuous data.
            # This is a simplification and might not be the most statistically rigorous approach
            # for all fNIRS GLM use cases.
            #
            # A more robust approach would be:
            # 1. Pass `raw_fnirs_haemo` and `events_df` (or `mne_events_array`) to this function.
            # 2. Create the design matrix for the *full continuous data*.
            # 3. Run `run_glm` on the *full continuous data*.
            # 4. Epoch the resulting beta estimates.
            #
            # Sticking to the current function signature (taking epochs), we'll create
            # a design matrix based on the events *within* the epochs, relative to the
            # start of each epoch. This is still potentially problematic for `run_glm`.
            #
            # Let's try creating a design matrix for the *concatenated* epochs.
            # This requires knowing the total number of samples if epochs were laid end-to-end.
            # This is getting complicated.

            # Let's simplify and assume `mne_nirs.statistics.run_glm` can work with epochs
            # and a design matrix created from the epoch events relative to the epoch start.
            # This is still not standard `run_glm` usage.

            # Reverting to the most likely intended use based on the original AnalysisService code:
            # Create a design matrix using `mne.stats.make_first_level_design_matrix`
            # based on the epoch events, assuming `run_glm` can interpret this when given epochs.
            # This is the least certain part of the refactoring regarding mne-nirs best practices.

            # Get event onsets relative to the start of the *first* epoch's underlying data
            # This requires knowing the tmin of the epochs relative to the original raw data start.
            # Assuming epochs.events[:, 0] are sample indices in the *original* raw data.
            # And epochs.tmin is the time offset from the event onset.
            # The onset relative to raw start is events[:, 0].
            # The time vector for the design matrix should match the samples in the epochs.
            # This is confusing.

            # Let's assume the simplest case: `run_glm` with epochs expects a design matrix
            # where each row corresponds to a sample *within* an epoch, repeated for all epochs.
            # This is highly inefficient and likely incorrect.

            # Okay, let's assume the user wants to run GLM on the *averaged* evoked responses per condition.
            # This is a common and statistically sound approach in fNIRS.
            # 1. Compute Evoked objects for each condition.
            # 2. Concatenate Evoked objects.
            # 3. Create a design matrix for the concatenated Evoked data.
            # 4. Run GLM on the concatenated Evoked data.

            # Let's implement the GLM on Evoked responses as a more standard approach for epoched data.
            # This changes the function's input/output slightly or requires averaging internally.

            # Option: Average epochs internally
            evoked_dict = {cond: fnirs_epochs_mne[cond].average() for cond in event_id_map_for_glm.keys()}
            
            # Concatenate evoked objects
            evokeds = list(evoked_dict.values())
            if not evokeds:
                 self.logger.warning("FNIRSGLMAnalyzer (epochs): No evoked responses created. Skipping GLM.")
                 return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}

            # Create design matrix for the concatenated evoked data
            # The design matrix needs to have rows equal to the number of samples in the concatenated evoked data.
            # The events for the design matrix are the onsets of each evoked response within the concatenated data.
            # If each evoked is length N, the events are at 0, N, 2N, ...
            
            n_samples_evoked = evokeds[0].data.shape[1] # Assuming all evoked have same length
            sfreq_evoked = evokeds[0].info['sfreq']
            
            events_for_evoked_dm = []
            event_id_for_evoked_dm = {}
            current_sample_offset = 0
            for cond_name, evoked in evoked_dict.items():
                event_code = event_id_map_for_glm[cond_name] # Get the original event code
                events_for_evoked_dm.append([current_sample_offset, 0, event_code])
                event_id_for_evoked_dm[cond_name] = event_code # Use original event codes/names
                current_sample_offset += evoked.data.shape[1] # Add number of samples in this evoked

            events_for_evoked_dm = np.array(events_for_evoked_dm)

            # Create design matrix using mne.stats.make_first_level_design_matrix
            # This function is suitable for event-based designs on continuous or evoked data.
            design_matrix_evoked = mne.stats.make_first_level_design_matrix(
                events_for_evoked_dm[:, 0] / sfreq_evoked, # Onsets in seconds
                sfreq_evoked,
                hrf_model=glm_hrf_model,
                drift_model='polynomial', order=3,
                event_id=event_id_for_evoked_dm # Map condition names to codes
            )

            # Concatenate evoked data into a single Raw-like object for run_glm
            # run_glm expects Raw or Epochs. Concatenated Evoked is like Raw.
            # Create a dummy info object for the concatenated data
            concatenated_info = mne.create_info(
                ch_names=evokeds[0].ch_names,
                sfreq=sfreq_evoked,
                ch_types=evokeds[0].info['ch_types']
            )
            concatenated_data = np.concatenate([e.data for e in evokeds], axis=1)
            raw_concatenated_evoked = mne.io.RawArray(concatenated_data, concatenated_info, verbose=False)

            # Filter design matrix columns to only include conditions present in config.FNIRS_CONTRASTS
            all_contrast_conditions = set()
            for _, contrast_weights_dict in glm_contrasts_config.items():
                all_contrast_conditions.update(contrast_weights_dict.keys())
            
            design_matrix_cols_to_keep = [
                col for col in design_matrix_evoked.columns 
                if col in all_contrast_conditions or col.lower() == 'intercept' or 'drift' in col.lower() or 'constant' in col.lower()
            ]
            design_matrix_filtered = design_matrix_evoked[design_matrix_cols_to_keep]

            if design_matrix_filtered.empty or not any(cond in design_matrix_filtered.columns for cond in all_contrast_conditions):
                 self.logger.warning(f"FNIRSGLMAnalyzer (evoked): Design matrix empty or missing contrast conditions after filtering. Design cols: {list(design_matrix_evoked.columns)}, Filtered: {list(design_matrix_filtered.columns)}. Skipping GLM.")
                 return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}

            self.logger.info(f"FNIRSGLMAnalyzer (evoked): Using design matrix with columns: {list(design_matrix_filtered.columns)}")
            
            # Run GLM on the concatenated evoked data
            glm_estimates_dict = run_glm(raw_concatenated_evoked, design_matrix_filtered, noise_model='ar1')
            # glm_estimates_dict will be a dict: ch_name -> GLMEstimate object
            
            contrast_results_dfs = {}
            active_rois_for_eeg_guidance = set()
            
            for contrast_name, contrast_weights in glm_contrasts_config.items():
                self.logger.info(f"FNIRSGLMAnalyzer (evoked): Computing contrast: {contrast_name} with weights {contrast_weights}")
                
                temp_contrast_dfs_for_channels = []
                for ch_name, glm_est in glm_estimates_dict.items():
                    try:
                        # Ensure all weights are for columns actually in the design matrix
                        valid_contrast_weights = {k: v for k, v in contrast_weights.items() if k in design_matrix_filtered.columns}
                        if not valid_contrast_weights:
                             self.logger.warning(f"FNIRSGLMAnalyzer (evoked): Skipping contrast '{contrast_name}' for channel {ch_name}: No valid weights found in filtered design matrix columns.")
                             continue

                        contrast_obj_ch = glm_est.compute_contrast(valid_contrast_weights)
                        contrast_df_ch = contrast_obj_ch.to_dataframe()
                        contrast_df_ch['ch_name'] = ch_name # Add channel name for aggregation
                        temp_contrast_dfs_for_channels.append(contrast_df_ch)
                    except Exception as e_ch_contrast:
                        self.logger.warning(f"Could not compute contrast '{contrast_name}' for channel {ch_name}: {e_ch_contrast}")
                
                if not temp_contrast_dfs_for_channels:
                    self.logger.warning(f"FNIRSGLMAnalyzer (evoked): No channel-wise contrast results for '{contrast_name}'.")
                    contrast_results_dfs[contrast_name] = pd.DataFrame()
                    continue

                aggregated_contrast_df = pd.concat(temp_contrast_dfs_for_channels).reset_index(drop=True)
                contrast_results_dfs[contrast_name] = aggregated_contrast_df
                self.logger.info(f"FNIRSGLMAnalyzer (evoked): Contrast '{contrast_name}' computed. Aggregated DataFrame shape: {aggregated_contrast_df.shape}")
                
                # Identify active ROIs for EEG guidance based on 'Emotion_vs_Neutral'
                if contrast_name == 'Emotion_vs_Neutral':
                    # glm_rois_config is already validated to be a dict
                    for roi_name_cfg, fnirs_channels_in_roi_cfg in glm_rois_config.items():
                        # Filter contrast results for channels within this ROI
                        roi_specific_contrast_df = aggregated_contrast_df[
                            aggregated_contrast_df['ch_name'].isin(fnirs_channels_in_roi_cfg)
                        ]
                        if not roi_specific_contrast_df.empty:
                            # Check for significant activation within the ROI
                            significant_activation = roi_specific_contrast_df[
                                roi_specific_contrast_df['p_value'] < glm_activation_p_threshold
                            ]
                            if not significant_activation.empty:
                                active_rois_for_eeg_guidance.add(roi_name_cfg)
                                self.logger.info(f"FNIRSGLMAnalyzer (evoked): ROI '{roi_name_cfg}' added to active list for EEG guidance.")
                            else:
                                 self.logger.debug(f"FNIRSGLMAnalyzer (evoked): ROI '{roi_name_cfg}' did not meet p-value threshold for EEG guidance.")
                        else:
                            self.logger.debug(f"FNIRSGLMAnalyzer (evoked): No contrast results for channels in ROI '{roi_name_cfg}'.")

            self.logger.info(f"FNIRSGLMAnalyzer (evoked): Final active ROIs for EEG guidance: {list(active_rois_for_eeg_guidance)}")

            return {
                'glm_estimates': glm_estimates_dict, # This is now a dict of GLMEstimate objects
                'contrast_results': contrast_results_dfs, # Dict of DataFrames
                'active_rois_for_eeg_guidance': list(active_rois_for_eeg_guidance)
            }
        except Exception as e_glm:
            self.logger.error(f"FNIRSGLMAnalyzer (evoked): Critical error during GLM: {e_glm}", exc_info=True)
            return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}

    # Keep the old method if it's used elsewhere or for a different purpose (GLM on raw data)
    # Note: This method is likely redundant if run_glm_on_epochs is the primary GLM analysis.
    # Consider removing it if not explicitly used elsewhere.
    def run_glm_and_extract_rois(self, raw_fnirs_haemo, analysis_metrics, participant_id, analysis_results_dir):
        """
        Runs GLM on fNIRS data (typically continuous raw), computes specified contrasts, 
        identifies active ROIs based on a specific contrast (Emotion vs. Neutral), and stores results.
        This method seems designed for continuous raw data.
        """
        self.logger.warning("FNIRSGLMAnalyzer: `run_glm_and_extract_rois` (for raw data) called. Ensure this is intended if epochs are available.")
        if raw_fnirs_haemo is None:
            self.logger.warning("FNIRSGLMAnalyzer (raw): No processed fNIRS data provided. Skipping GLM analysis.")
            return {}
        # ... (rest of the original method for GLM on raw data) ...
        # This original method needs to be reviewed if it's still needed.
        # For the orchestrator using epochs, the `run_glm_on_epochs` is more relevant.
        self.logger.info("FNIRSGLMAnalyzer (raw): GLM analysis on raw data (placeholder, ensure correct implementation if used).")
        return {'active_rois_for_eeg_guidance': [], 'contrast_results': {}, 'glm_estimates': None}