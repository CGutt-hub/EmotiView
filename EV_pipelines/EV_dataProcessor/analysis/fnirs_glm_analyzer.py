import os
import numpy as np
import pandas as pd
import mne
import mne_nirs
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import (picks_pair_to_ch_names,
                               picks_optodes_to_channel_locations)
from .. import config # Relative import

class FNIRSGLMAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSGLMAnalyzer initialized.")

    def run_glm_and_extract_rois(self, raw_fnirs_haemo, analysis_metrics, participant_id, analysis_results_dir):
        """
        Runs GLM on fNIRS data, computes specified contrasts, identifies active ROIs
        based on a specific contrast (Emotion vs. Neutral), and stores results.

        Args:
            raw_fnirs_haemo (mne.io.Raw): Processed MNE Raw object for fNIRS (HbO/HbR).
            analysis_metrics (dict): Dictionary to store results.
            participant_id (str): The ID of the participant.
            analysis_results_dir (str): Directory to save results and plots.

        Returns:
            dict: A dictionary containing information about active ROIs and contrast results,
                  or an empty dict on failure.
                  Includes 'active_rois_per_condition' (current logic) and
                  'active_rois_for_eeg_guidance' (based on specific contrast).
        """
        if raw_fnirs_haemo is None:
            self.logger.warning("FNIRSGLMAnalyzer - No processed fNIRS data provided. Skipping GLM analysis.")
            return {}

        self.logger.info("FNIRSGLMAnalyzer - Running GLM analysis.")
        results_info = {} # Dictionary to return information

        try:
            # 1. Create Design Matrix
            # Need events from the raw object
            events, event_id = mne.events_from_annotations(raw_fnirs_haemo, verbose=False)
            if events.size == 0 or not event_id:
                self.logger.warning("FNIRSGLMAnalyzer - No events found in fNIRS data. Cannot create design matrix. Skipping GLM.")
                return {}

            # Use only stimulus conditions for the design matrix
            stimulus_event_id = {k: v for k, v in event_id.items() if k in config.EVENT_ID_TO_CONDITION.values()}
            if not stimulus_event_id:
                 self.logger.warning("FNIRSGLMAnalyzer - No stimulus events found in fNIRS data based on config.EVENT_ID_TO_CONDITION. Skipping GLM.")
                 return {}

            self.logger.info(f"FNIRSGLMAnalyzer - Creating design matrix with events: {list(stimulus_event_id.keys())}")

            # Design matrix parameters (adjust duration and other params as needed)
            # Duration should match the stimulus duration
            design_matrix = make_first_level_design_matrix(
                raw_fnirs_haemo,
                events=events,
                event_id=stimulus_event_id,
                tmin=0, # Start of epoch relative to event
                tmax=config.STIMULUS_DURATION_SECONDS, # End of epoch relative to event
                sfreq=raw_fnirs_haemo.info['sfreq'],
                hrf_model='spm', # Hemodynamic Response Function model
                add_intercept=True # Add a constant term
            )

            # Save design matrix for inspection
            design_matrix_path = os.path.join(analysis_results_dir, f'{participant_id}_fnirs_design_matrix.csv')
            design_matrix.to_csv(design_matrix_path)
            self.logger.info(f"FNIRSGLMAnalyzer - Design matrix saved to {design_matrix_path}")

            # 2. Run GLM
            self.logger.info("FNIRSGLMAnalyzer - Running GLM...")
            glm_results = run_glm(raw_fnirs_haemo, design_matrix)
            self.logger.info("FNIRSGLMAnalyzer - GLM completed.")

            # glm_results is a dictionary where keys are channel names and values are GLMEstimate objects

            # 3. Extract Beta Coefficients (theta) per condition and ROI
            # Store mean beta per ROI for each condition
            active_rois_per_condition = {} # For current PLV guidance logic (WP1 part 1)
            for roi_name, channel_names in config.FNIRS_ROIS.items():
                roi_channels = [ch for ch in channel_names if ch in raw_fnirs_haemo.ch_names]
                if not roi_channels:
                    self.logger.warning(f"FNIRSGLMAnalyzer - None of the channels for ROI '{roi_name}' found in fNIRS data. Skipping.")
                    continue

                # Get GLM results for channels within this ROI
                roi_glm_results = {ch: glm_results[ch] for ch in roi_channels if ch in glm_results}

                if not roi_glm_results:
                     self.logger.warning(f"FNIRSGLMAnalyzer - No GLM results available for channels in ROI '{roi_name}'. Skipping.")
                     continue

                for condition_name in stimulus_event_id.keys():
                    # Extract beta values for this condition across channels in the ROI
                    beta_values = [
                        res.theta[condition_name] for ch, res in roi_glm_results.items()
                        if condition_name in res.theta # Ensure condition is in the design matrix
                    ]

                    if beta_values:
                        mean_beta = np.mean(beta_values)
                        # Store mean beta for this condition and ROI
                        analysis_metrics[f'fnirs_hbo_{condition_name}_{roi_name}_theta_mean'] = mean_beta
                        self.logger.debug(f"FNIRSGLMAnalyzer - Mean HbO beta for {condition_name} in {roi_name}: {mean_beta:.4f}")

                        # Keep track of ROIs with non-NaN betas for current PLV guidance logic
                        if not np.isnan(mean_beta):
                            active_rois_per_condition.setdefault(condition_name, []).append(roi_name)
                    else:
                        analysis_metrics[f'fnirs_hbo_{condition_name}_{roi_name}_theta_mean'] = np.nan
                        self.logger.debug(f"FNIRSGLMAnalyzer - No valid HbO beta values for {condition_name} in {roi_name}. Storing NaN.")

            results_info['active_rois_per_condition'] = active_rois_per_condition
            self.logger.info("FNIRSGLMAnalyzer - Extracted mean HbO betas per condition and ROI.")


            # 4. Compute Contrasts (for WP1 fNIRS guidance)
            self.logger.info("FNIRSGLMAnalyzer - Computing specified contrasts.")
            contrast_results = {}
            contrast_objects_or_paths = {} # Store objects or paths for plotting

            # Identify active ROIs based on the specific 'Emotion_vs_Neutral' contrast (WP1 part 1)
            active_rois_for_eeg_guidance = set()
            emotion_vs_neutral_contrast_name = 'Emotion_vs_Neutral' # Hardcoded name from proposal/config

            for contrast_name, contrast_weights in config.FNIRS_GLM_CONTRASTS.items():
                self.logger.info(f"FNIRSGLMAnalyzer - Computing contrast: {contrast_name}")
                try:
                    # Ensure all conditions in the contrast are in the design matrix
                    if not all(cond in design_matrix.columns for cond in contrast_weights.keys()):
                        self.logger.warning(f"FNIRSGLMAnalyzer - Not all conditions for contrast '{contrast_name}' found in design matrix. Skipping contrast.")
                        continue

                    # Compute the contrast for each channel
                    contrast = glm_results.compute_contrast(contrast_weights)
                    contrast_results[contrast_name] = contrast

                    # Save contrast results (e.g., t-values)
                    contrast_df = contrast.to_dataframe()
                    contrast_path = os.path.join(analysis_results_dir, f'{participant_id}_fnirs_contrast_{contrast_name}.csv')
                    contrast_df.to_csv(contrast_path)
                    self.logger.info(f"FNIRSGLMAnalyzer - Contrast '{contrast_name}' results saved to {contrast_path}")
                    contrast_objects_or_paths[contrast_name] = contrast_path # Store path for plotting

                    # Identify active ROIs based on the specific contrast for EEG guidance
                    if contrast_name == emotion_vs_neutral_contrast_name:
                        self.logger.info(f"FNIRSGLMAnalyzer - Identifying active ROIs for EEG guidance based on '{contrast_name}' contrast.")
                        for roi_name, channel_names in config.FNIRS_ROIS.items():
                            roi_channels = [ch for ch in channel_names if ch in contrast.ch_names]
                            if not roi_channels:
                                continue

                            # Check if any channel in the ROI meets the activation threshold for this contrast
                            roi_contrast_df = contrast_df[contrast_df['ch_name'].isin(roi_channels)]
                            if not roi_contrast_df.empty:
                                # Check if any channel's t-value exceeds threshold AND p-value is below threshold
                                # Using abs(t) for threshold check as activation can be positive or negative
                                significant_channels_in_roi = roi_contrast_df[
                                    (np.abs(roi_contrast_df['t']) >= config.FNIRS_ACTIVATION_T_THRESHOLD) &
                                    (roi_contrast_df['p_value'] < config.FNIRS_ACTIVATION_P_THRESHOLD)
                                ]
                                if not significant_channels_in_roi.empty:
                                    active_rois_for_eeg_guidance.add(roi_name)
                                    self.logger.info(f"FNIRSGLMAnalyzer - ROI '{roi_name}' identified as active for EEG guidance (Emotion vs Neutral contrast).")
                                else:
                                     self.logger.debug(f"FNIRSGLMAnalyzer - ROI '{roi_name}' did not meet activation thresholds for EEG guidance.")
                            else:
                                self.logger.debug(f"FNIRSGLMAnalyzer - No contrast results for channels in ROI '{roi_name}'.")

                except Exception as e:
                    self.logger.error(f"FNIRSGLMAnalyzer - Error computing contrast '{contrast_name}': {e}", exc_info=True)
                    contrast_results[contrast_name] = None # Store None on error

            results_info['contrast_results'] = contrast_results
            results_info['contrast_objects_or_paths'] = contrast_objects_or_paths # For plotting
            results_info['active_rois_for_eeg_guidance'] = list(active_rois_for_eeg_guidance) # Convert set to list

            self.logger.info("FNIRSGLMAnalyzer - Contrast computation completed.")
            self.logger.info(f"FNIRSGLMAnalyzer - Active ROIs identified for EEG guidance: {results_info['active_rois_for_eeg_guidance']}")


            self.logger.info("FNIRSGLMAnalyzer - GLM analysis and extraction completed.")
            return results_info

        except Exception as e:
            self.logger.error(f"FNIRSGLMAnalyzer - Critical error during GLM analysis: {e}", exc_info=True)
            return {} # Return empty dict on critical failure