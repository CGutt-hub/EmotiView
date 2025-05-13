import mne
import mne_nirs
import numpy as np
import os # For saving contrast objects if needed
from ... import config # Relative import

class FNIRSGLMAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSGLMAnalyzer initialized.")

    def run_glm_and_extract_rois(self, fnirs_haemo_processed, analysis_metrics, participant_id, analysis_results_dir):
        """
        Runs a first-level GLM on fNIRS data and extracts mean theta for defined ROIs.
        Args:
            fnirs_haemo_processed (mne.io.Raw): Preprocessed fNIRS haemoglobin data.
            analysis_metrics (dict): Dictionary to store/update results.
            participant_id (str): Participant ID for saving files.
            analysis_results_dir (str): Directory to save any GLM output like contrasts.
        """
        if fnirs_haemo_processed is None:
            self.logger.warning("FNIRSGLMAnalyzer - No processed fNIRS data provided. Skipping GLM.")
            return

        self.logger.info("FNIRSGLMAnalyzer - Starting fNIRS GLM analysis.")
        try:
            events_fnirs, event_id_fnirs = mne.events_from_annotations(fnirs_haemo_processed, verbose=False)
            if not events_fnirs.size: # Check if events_fnirs is empty
                self.logger.warning("FNIRSGLMAnalyzer - No events found from annotations. Skipping GLM.")
                return
            self.logger.info(f"FNIRSGLMAnalyzer - Events found: {len(events_fnirs)}, Event IDs: {event_id_fnirs}")
            
            event_id_fnirs_str_keys = {str(k): int(v) for k, v in event_id_fnirs.items()} # Ensure values are int

            design_matrix = mne_nirs.statistics.make_first_level_design_matrix(
                fnirs_haemo_processed, events_fnirs,
                hrf_model='spm', drift_model='polynomial', drift_order=1,
                stim_dur=config.STIMULUS_DURATION_SECONDS,
                event_id=event_id_fnirs_str_keys, # MNE expects integer event IDs in the event_id dict values
                verbose=False
            )
            self.logger.info(f"FNIRSGLMAnalyzer - Design matrix created (Shape: {design_matrix.shape}).")

            glm_estimates = mne_nirs.statistics.run_glm(fnirs_haemo_processed, design_matrix, noise_model='ar1', verbose=False)
            self.logger.info("FNIRSGLMAnalyzer - GLM run completed.")

            conditions_in_design = [col for col in design_matrix.columns if col not in ['drift_1', 'constant'] and not col.startswith('cosine')]
            
            for condition_name in conditions_in_design:
                # contrast_obj = glm_estimates.compute_contrast(condition_name) # This is a Contrast object
                # self.logger.info(f"FNIRSGLMAnalyzer - Contrast computed for: {condition_name} vs Baseline.")
                # contrast_path = os.path.join(analysis_results_dir, f"{participant_id}_fnirs_contrast_{condition_name}_hbo.fif")
                # try: contrast_obj.save(contrast_path, overwrite=True) # Might need specific handling to save
                # except: self.logger.warning(f"Could not save contrast object for {condition_name}")

                for roi_name, roi_channels_config in config.FNIRS_ROIS.items():
                    valid_roi_ch_names = [ch for ch in roi_channels_config if ch in fnirs_haemo_processed.ch_names]
                    if not valid_roi_ch_names: continue

                    hbo_channels_indices_in_roi = mne.pick_channels_regexp(fnirs_haemo_processed.ch_names, f"({'|'.join(valid_roi_ch_names)}).*hbo")
                    if len(hbo_channels_indices_in_roi) > 0:
                        # Theta for the condition (beta coefficient)
                        theta_condition_all_chans = glm_estimates.theta[design_matrix.columns.get_loc(condition_name)]
                        mean_theta_roi_hbo = np.mean(theta_condition_all_chans[hbo_channels_indices_in_roi])
                        analysis_metrics[f'fnirs_hbo_{condition_name}_{roi_name}_theta_mean'] = mean_theta_roi_hbo
                        self.logger.info(f"FNIRSGLMAnalyzer - Metric: fnirs_hbo_{condition_name}_{roi_name}_theta_mean = {mean_theta_roi_hbo:.4f}")
            self.logger.info("FNIRSGLMAnalyzer - ROI metrics extracted.")
        except Exception as e:
            self.logger.error(f"FNIRSGLMAnalyzer - Error during fNIRS GLM analysis: {e}", exc_info=True)