import os
import pandas as pd
import numpy as np
import json
import pingouin as pg # For post-hoc in EDA ANOVA

from .analysis_service import AnalysisService
from ..reporting.plotting_service import PlottingService
from ..utils.helpers import apply_fdr_correction

# Helper class for JSON encoding numpy types, if not already globally available
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return super(NpEncoder, self).default(obj)

class GroupAnalyzer:
    def __init__(self, main_logger, output_base_dir, 
                 emotional_conditions_config, plv_eeg_bands_config, plv_primary_eeg_band_for_wp3_config,
                 figure_format_config, figure_dpi_config): # Added plotting configs
        self.logger = main_logger
        self.output_base_dir = output_base_dir
        self.group_results_dir = os.path.join(output_base_dir, "_GROUP_RESULTS")
        os.makedirs(self.group_results_dir, exist_ok=True)
        self.logger.info(f"Group results will be saved in: {self.group_results_dir}")

        # Store passed configurations
        self.EMOTIONAL_CONDITIONS = emotional_conditions_config
        self.PLV_EEG_BANDS = plv_eeg_bands_config
        # Ensure PLV_EEG_BANDS is a dict, provide a default if not (though it should be from EV_config)
        if not isinstance(self.PLV_EEG_BANDS, dict):
            self.logger.warning("PLV_EEG_BANDS configuration is not a dictionary. Using a default empty dict.")
            self.PLV_EEG_BANDS = {}
        self.PLV_PRIMARY_EEG_BAND_FOR_WP3 = plv_primary_eeg_band_for_wp3_config
        if not all([self.EMOTIONAL_CONDITIONS, self.PLV_EEG_BANDS, self.PLV_PRIMARY_EEG_BAND_FOR_WP3]):
            self.logger.warning("GroupAnalyzer initialized with one or more missing critical configurations (emotional_conditions, plv_eeg_bands, plv_primary_eeg_band_for_wp3). This may lead to errors.")
        
        self.group_plots_dir = os.path.join(output_base_dir, "_GROUP_PLOTS")
        os.makedirs(self.group_plots_dir, exist_ok=True)
        self.logger.info(f"Group plots will be saved in: {self.group_plots_dir}")

        # Instantiate services needed for group analysis
        # AnalysisService is needed for ANOVA, Correlation
        # PlottingService is needed for visualizing group results
        self.analysis_service = AnalysisService(self.logger) # Assuming AnalysisService doesn't need main_config for group stats
        self.plotting_service = PlottingService(self.logger, self.group_plots_dir,
                                                reporting_figure_format_config=figure_format_config,
                                                reporting_dpi_config=figure_dpi_config)
        
        self.all_group_level_results = {} # To store all statistical outputs

    def _aggregate_data_for_wp1(self, all_participant_artifacts):
        """Aggregates average PLV data for WP1 ANOVA."""
        all_avg_plv_dfs = []
        for artifact in all_participant_artifacts:
            # Assuming 'avg_plv_wp1' is the DataFrame with participant_id, condition, modality_pair, eeg_band, plv
            avg_plv_df = artifact.get('analysis_outputs', {}).get('dataframes', {}).get('avg_plv_wp1', pd.DataFrame())
            if not avg_plv_df.empty:
                all_avg_plv_dfs.append(avg_plv_df)
        
        combined_avg_plv_df = pd.concat(all_avg_plv_dfs, ignore_index=True) if all_avg_plv_dfs else pd.DataFrame()
        if combined_avg_plv_df.empty:
            self.logger.warning("WP1: No average PLV data found across participants for aggregation.")
        return combined_avg_plv_df

    def _aggregate_data_for_wp2(self, all_participant_artifacts):
        """Aggregates trial-wise PLV and survey data (SAM arousal) for WP2 correlation."""
        wp2_data_list = []
        for artifact in all_participant_artifacts:
            p_id = artifact.get('participant_id')
            trial_plv_df = artifact.get('analysis_outputs', {}).get('dataframes', {}).get('trial_plv_wp1', pd.DataFrame())
            survey_df = artifact.get('analysis_outputs', {}).get('dataframes', {}).get('survey_data_per_trial', pd.DataFrame())

            if not trial_plv_df.empty and not survey_df.empty and \
               'trial_identifier_eprime' in trial_plv_df.columns and \
               'trial_identifier_eprime' in survey_df.columns and 'sam_arousal' in survey_df.columns:
                
                # Merge PLV with survey data on trial_identifier_eprime
                # We might need to average PLV per trial if there are multiple bands/modalities
                # For simplicity, let's assume we focus on a primary band/modality or average across them per trial
                # Example: Focus on a specific band and modality, or average PLV across bands/modalities per trial
                
                # For now, let's assume trial_plv_df has one PLV value per trial_identifier_eprime relevant for WP2
                # Or we average it:
                avg_trial_plv_for_wp2 = trial_plv_df.groupby('trial_identifier_eprime')['plv'].mean().reset_index()
                
                merged_df = pd.merge(avg_trial_plv_for_wp2, survey_df[['trial_identifier_eprime', 'sam_arousal']], on='trial_identifier_eprime', how='inner')
                if not merged_df.empty:
                    merged_df['participant_id'] = p_id
                    wp2_data_list.append(merged_df)
        
        return pd.concat(wp2_data_list, ignore_index=True) if wp2_data_list else pd.DataFrame()

    def _aggregate_eda_metrics(self, all_participant_artifacts):
        eda_metrics_list = []
        for res_idx, res_val in enumerate(all_participant_artifacts):
            if isinstance(res_val, dict) and res_val.get('status') == 'success':
                p_id_eda = res_val.get('participant_id', f'unknown_participant_{res_idx}')
                metrics_eda = res_val.get('analysis_outputs', {}).get('metrics', {})
                for cond_name in self.EMOTIONAL_CONDITIONS: # Use instance variable
                    phasic_mean_key = f'eda_phasic_mean_{cond_name}'
                    scr_count_key = f'eda_scr_count_{cond_name}'
                    if phasic_mean_key in metrics_eda:
                        eda_metrics_list.append({'participant_id': p_id_eda, 'condition': cond_name, 'metric_type': 'phasic_mean', 'value': metrics_eda[phasic_mean_key]})
                    if scr_count_key in metrics_eda:
                         eda_metrics_list.append({'participant_id': p_id_eda, 'condition': cond_name, 'metric_type': 'scr_count', 'value': metrics_eda[scr_count_key]})
        return pd.DataFrame(eda_metrics_list)

    def _analyze_wp1_plv_anova(self, combined_plv_data_wp1):
        """Performs Repeated Measures ANOVA for WP1 average PLV data."""
        if combined_plv_data_wp1.empty:
            self.logger.warning("WP1: No combined PLV data to analyze for ANOVA.")
            return

        # Ensure 'condition' is categorical and ordered if necessary
        combined_plv_data_wp1['condition'] = pd.Categorical(combined_plv_data_wp1['condition'], categories=self.EMOTIONAL_CONDITIONS, ordered=True)
        
        # Perform ANOVA for each EEG band and modality pair
        if not combined_plv_data_wp1.empty:
            self.logger.info(f"WP1: Aggregated PLV data from {combined_plv_data_wp1['participant_id'].nunique()} participants for group ANOVA.")
            for eeg_band in self.PLV_EEG_BANDS.keys(): # Use instance variable
                for modality in ['EEG-HRV', 'EEG-EDA']:
                    self.logger.info(f"--- WP1 ANOVA for: {eeg_band} & {modality} ---")
                    df_subset = combined_plv_data_wp1[(combined_plv_data_wp1['eeg_band'] == eeg_band) & (combined_plv_data_wp1['modality_pair'] == modality)].copy()
                    
                    if not df_subset.empty and df_subset['participant_id'].nunique() > 1 and df_subset['condition'].nunique() > 1:
                        anova_results = self.analysis_service.run_repeated_measures_anova(data_df=df_subset, dv='plv', within='condition', subject='participant_id')
                        self.logger.info(f"WP1 ANOVA Results ({eeg_band}, {modality}):\n{anova_results}")
                        if anova_results is not None and not anova_results.empty:
                            self.all_group_level_results[f'wp1_anova_{eeg_band}_{modality}'] = anova_results.to_dict()
                            anova_results.to_csv(os.path.join(self.group_results_dir, f"group_anova_wp1_{eeg_band}_{modality}.csv"))
                            self.plotting_service.plot_anova_results("GROUP", df_subset, 'plv', 'condition', None, f"WP1 PLV: {eeg_band} {modality}", f"wp1_anova_{eeg_band}_{modality}")
                    else:
                        self.logger.warning(f"WP1: Not enough data for ANOVA for {eeg_band} & {modality} (participants: {df_subset['participant_id'].nunique()}, conditions: {df_subset['condition'].nunique()}).")


    def _analyze_wp2_correlation_plv_sam(self, aggregated_wp2_data):
        """Performs correlation analysis for WP2 (PLV vs SAM arousal)."""
        if aggregated_wp2_data.empty:
            self.logger.warning("WP2: No aggregated data for PLV vs SAM arousal correlation.")
            return

        if not aggregated_wp2_data.empty and 'participant_id' in aggregated_wp2_data.columns and \
           'plv' in aggregated_wp2_data.columns and 'sam_arousal' in aggregated_wp2_data.columns:
            
            participant_avg_wp2 = aggregated_wp2_data.groupby('participant_id')[['plv', 'sam_arousal']].mean().reset_index()
            
            if len(participant_avg_wp2) < 3: # Need at least 3 data points for meaningful correlation
                self.logger.warning(f"WP2: Not enough participants ({len(participant_avg_wp2)}) with averaged PLV and SAM arousal for correlation.")
                return

            corr_wp2 = self.analysis_service.run_correlation_analysis(participant_avg_wp2['plv'], participant_avg_wp2['sam_arousal'], name1='Avg_Participant_PLV', name2='Avg_Participant_SAM_Arousal')
            self.logger.info(f"WP2 Correlation Results (PLV vs SAM Arousal):\n{corr_wp2}")
            if corr_wp2 is not None and not corr_wp2.empty:
                self.all_group_level_results['wp2_correlation_plv_sam'] = corr_wp2.to_dict(orient='records')
                corr_wp2.to_csv(os.path.join(self.group_results_dir, "group_corr_wp2_plv_vs_sam.csv"))
                self.plotting_service.plot_correlation_results("GROUP", participant_avg_wp2['plv'], participant_avg_wp2['sam_arousal'], "Average Participant PLV", "Average Participant SAM Arousal", "WP2: PLV vs SAM Arousal", "wp2_plv_sam_corr", corr_wp2)
        else:
            self.logger.warning("WP2: 'plv' or 'sam_arousal' columns not found in aggregated data for correlation.")

    def _analyze_wp3_correlation_rmssd_plv(self, all_participant_artifacts):
        """Performs correlation analysis for WP3 (Baseline RMSSD vs. task-related PLV)."""
        self.logger.info("--- Group Analysis: WP3 (RMSSD vs PLV Correlation) ---")
        wp3_data_list = []
        for artifact in all_participant_artifacts:
            p_id = artifact.get('participant_id')
            metrics = artifact.get('analysis_outputs', {}).get('metrics', {})
            baseline_rmssd = metrics.get('baseline_rmssd')
            plv_negative_specific = metrics.get('wp3_avg_plv_negative_specific') # This was calculated in EV_analyzer

            if baseline_rmssd is not None and not np.isnan(baseline_rmssd) and \
               plv_negative_specific is not None and not np.isnan(plv_negative_specific):
                wp3_data_list.append({'participant_id': p_id, 'baseline_rmssd': baseline_rmssd, 'plv_negative_specific': plv_negative_specific})

        if wp3_data_list:
            wp3_group_df = pd.DataFrame(wp3_data_list).dropna()
            if not wp3_group_df.empty and len(wp3_group_df) >= 3: # Need at least 3 data points
                corr_wp3 = self.analysis_service.run_correlation_analysis(wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], name1='Baseline_RMSSD', name2=f'Avg_PLV_{self.PLV_PRIMARY_EEG_BAND_FOR_WP3}_HRV_Negative')
                self.logger.info(f"WP3 Correlation Results:\n{corr_wp3}")
                if corr_wp3 is not None and not corr_wp3.empty:
                    self.all_group_level_results['wp3_correlation_rmssd_plv'] = corr_wp3.to_dict(orient='records')
                    corr_wp3.to_csv(os.path.join(self.group_results_dir, f"group_corr_wp3_rmssd_vs_plv_neg_{self.PLV_PRIMARY_EEG_BAND_FOR_WP3}.csv"))
                    self.plotting_service.plot_correlation_results("GROUP", wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], "Baseline RMSSD (ms)", f"Avg PLV ({self.PLV_PRIMARY_EEG_BAND_FOR_WP3}-HRV-Negative)", f"WP3: RMSSD vs Negative PLV ({self.PLV_PRIMARY_EEG_BAND_FOR_WP3})", f"wp3_rmssd_plv_neg_{self.PLV_PRIMARY_EEG_BAND_FOR_WP3}", corr_wp3)
            else:
                self.logger.warning(f"WP3: Not enough valid data points ({len(wp3_group_df)}) for RMSSD vs PLV correlation.")

    def _analyze_wp4_correlation_fai_plv(self, all_participant_artifacts):
        """Performs correlation analysis for WP4 (FAI vs. branch-specific PLV)."""
        self.logger.info("--- Group Analysis: WP4 (FAI vs PLV Correlation) ---")
        wp4_data_list = []
        for artifact in all_participant_artifacts:
            p_id = artifact.get('participant_id')
            metrics = artifact.get('analysis_outputs', {}).get('metrics', {})
            avg_fai_emotional = metrics.get('wp4_avg_fai_f4f3_emotional') # Calculated in EV_analyzer

            avg_plv_df = artifact.get('analysis_outputs', {}).get('dataframes', {}).get('avg_plv_wp1', pd.DataFrame())
            plv_for_fai_correlation = np.nan
            if not avg_plv_df.empty:
                relevant_plv = avg_plv_df[
                    (avg_plv_df['condition'].isin(['Positive', 'Negative'])) &
                    (avg_plv_df['eeg_band'] == self.PLV_PRIMARY_EEG_BAND_FOR_WP3) & 
                    (avg_plv_df['modality_pair'] == 'EEG-HRV') 
                ]['plv']
                if not relevant_plv.empty:
                    plv_for_fai_correlation = relevant_plv.mean()

            if avg_fai_emotional is not None and not np.isnan(avg_fai_emotional) and \
               plv_for_fai_correlation is not None and not np.isnan(plv_for_fai_correlation):
                wp4_data_list.append({'participant_id': p_id, 'avg_fai_emotional': avg_fai_emotional, 'plv_for_fai_correlation': plv_for_fai_correlation})

        if wp4_data_list:
            wp4_group_df = pd.DataFrame(wp4_data_list).dropna()
            if not wp4_group_df.empty and len(wp4_group_df) >= 3: # Need at least 3 data points
                corr_wp4 = self.analysis_service.run_correlation_analysis(wp4_group_df['avg_fai_emotional'], wp4_group_df['plv_for_fai_correlation'], name1='Avg_FAI_Emotional', name2='PLV_Emotional')
                self.logger.info(f"WP4 Correlation Results (FAI vs PLV):\n{corr_wp4}")
                if corr_wp4 is not None and not corr_wp4.empty:
                    self.all_group_level_results['wp4_correlation_fai_plv'] = corr_wp4.to_dict(orient='records')
                    corr_wp4.to_csv(os.path.join(self.group_results_dir, "group_corr_wp4_fai_vs_plv.csv"))
                    self.plotting_service.plot_correlation_results("GROUP", wp4_group_df['avg_fai_emotional'], wp4_group_df['plv_for_fai_correlation'], "Average FAI (Emotional Conditions)", "PLV (Emotional Conditions)", "WP4: FAI vs. PLV", "wp4_fai_plv_corr", corr_wp4)
            else:
                self.logger.warning(f"WP4: Not enough valid data points ({len(wp4_group_df)}) for FAI vs PLV correlation.")

    def run_group_analysis(self, all_participant_artifacts):
        """
        Main orchestrator for group-level analyses.
        """
        if not all_participant_artifacts:
            self.logger.warning("No participant artifacts provided for group analysis. Skipping.")
            return

        self.logger.info(f"--- Starting Group-Level Analysis with {len(all_participant_artifacts)} participant artifacts ---")

        # --- WP1: Emotional Modulation of Synchrony (ANOVA on PLV) ---
        combined_plv_data_wp1 = self._aggregate_data_for_wp1(all_participant_artifacts)
        self._analyze_wp1_plv_anova(combined_plv_data_wp1)

        # --- WP2: Synchrony and Subjective Arousal (Correlation PLV vs SAM) ---
        aggregated_wp2_data = self._aggregate_data_for_wp2(all_participant_artifacts)
        if not aggregated_wp2_data.empty:
            self._analyze_wp2_correlation_plv_sam(aggregated_wp2_data)
        else:
            self.logger.warning("WP2: No data aggregated for PLV vs SAM correlation analysis.")

        # --- WP3: Baseline Vagal Tone and Task-Related Synchrony (Correlation RMSSD vs PLV) ---
        self._analyze_wp3_correlation_rmssd_plv(all_participant_artifacts)

        # --- WP4: Frontal Asymmetry and Branch-Specific Synchrony (Correlation FAI vs PLV) ---
        self._analyze_wp4_correlation_fai_plv(all_participant_artifacts)

        # --- (Optional) EDA Metrics ANOVA/Comparison ---
        # self.logger.info("--- Group Analysis: EDA Metrics ---")
        # aggregated_eda_metrics = self._aggregate_eda_metrics(all_participant_artifacts)
        # if not aggregated_eda_metrics.empty:
        #     # Example: Perform ANOVA on phasic_mean or scr_count if desired
        #     # self._analyze_eda_anova(aggregated_eda_metrics) # You'd need to implement this
        #     aggregated_eda_metrics.to_csv(os.path.join(self.group_results_dir, "group_eda_metrics_summary.csv"), index=False)
        #     self.logger.info(f"Aggregated EDA metrics saved to group_eda_metrics_summary.csv")

        # Save all collected group-level statistical results to a JSON file
        summary_json_path = os.path.join(self.group_results_dir, "group_analysis_summary_stats.json")
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(self.all_group_level_results, f, indent=4, cls=NpEncoder)
            self.logger.info(f"All group analysis results saved to {summary_json_path}")
        except Exception as e_save:
            self.logger.error(f"Failed to save all group analysis results: {e_save}", exc_info=True)

        self.logger.info("--- Group-Level Analysis Completed ---")