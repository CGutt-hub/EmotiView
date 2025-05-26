import os
import pandas as pd
import numpy as np
import json
import pingouin as pg # For post-hoc in EDA ANOVA

from .analysis_service import AnalysisService
from ..reporting.plotting_service import PlottingService
from ..utils.helpers import apply_fdr_correction
from ..orchestrators import config # Assuming config is accessible

EMOTIONAL_CONDITIONS = config.EMOTIONAL_CONDITIONS # Get from config

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
    def __init__(self, main_logger, output_base_dir):
        self.logger = main_logger
        self.output_base_dir = output_base_dir
        self.group_results_dir = os.path.join(output_base_dir, "_GROUP_RESULTS")
        os.makedirs(self.group_results_dir, exist_ok=True)
        
        self.group_plots_dir = os.path.join(output_base_dir, "_GROUP_PLOTS")
        os.makedirs(self.group_plots_dir, exist_ok=True)

        self.analysis_service = AnalysisService(self.logger)
        self.plotting_service = PlottingService(self.logger, self.group_plots_dir)
        self.all_group_level_results = {} # To store results for JSON dump

    def run_group_analysis(self, all_participants_summary_artifacts):
        """
        Aggregates data from all participants and performs group-level analyses.
        """
        self.logger.info("--- Starting Group-Level Analysis ---")

        # --- Data Aggregation (moved from pilot_orchestrator.py) ---
        # WP1: ANOVA on PLV
        all_avg_plv_wp1_dfs = [
            res['analysis_outputs']['dataframes']['avg_plv_wp1'] 
            for res in all_participants_summary_artifacts
            if isinstance(res, dict) and res.get('status') == 'success' and 
               res.get('analysis_outputs', {}).get('dataframes', {}).get('avg_plv_wp1') is not None
        ]
        all_baseline_plv_dfs = [
            res['analysis_outputs']['dataframes']['baseline_plv']
            for res in all_participants_summary_artifacts
            if isinstance(res, dict) and res.get('status') == 'success' and
               res.get('analysis_outputs', {}).get('dataframes', {}).get('baseline_plv') is not None
        ]

        # WP2: Correlation SAM Arousal vs. PLV
        all_trial_plv_wp1_dfs_for_wp2 = [ 
            res['analysis_outputs']['dataframes']['trial_plv_wp1']
            for res in all_participants_summary_artifacts
            if isinstance(res, dict) and res.get('status') == 'success' and
               res.get('analysis_outputs', {}).get('dataframes', {}).get('trial_plv_wp1') is not None
        ]
        all_survey_dfs_for_wp2 = [ 
            res['analysis_outputs']['dataframes']['survey_data_per_trial'] 
            for res in all_participants_summary_artifacts
            if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('survey_data_per_trial') is not None
        ]

        # WP3: Correlation RMSSD vs. PLV
        wp3_data_list = [{'participant_id': r['participant_id'],
                          'baseline_rmssd': r['analysis_outputs']['metrics'].get('baseline_rmssd'),
                          'plv_negative_specific': r['analysis_outputs']['metrics'].get('wp3_avg_plv_negative_specific')}
                         for r in all_participants_summary_artifacts 
                         if isinstance(r, dict) and r.get('status') == 'success' and
                         r.get('analysis_outputs', {}).get('metrics', {}).get('baseline_rmssd') is not None and
                         not np.isnan(r['analysis_outputs']['metrics'].get('baseline_rmssd'))] 

        # WP4: Correlation FAI vs. PLV
        wp4_fai_list = []
        for res in all_participants_summary_artifacts:
            if isinstance(res, dict) and res.get('status') == 'success' and \
               res.get('analysis_outputs', {}).get('metrics', {}).get('wp4_avg_fai_f4f3_emotional') is not None:
                avg_fai_val = res['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional']
                if not np.isnan(avg_fai_val): 
                     wp4_fai_list.append({'participant_id': res['participant_id'], 
                                          'avg_fai_f4f3_emotional': avg_fai_val})

        # EDA Features ANOVA
        eda_features_data_list = []
        for res_idx, res_val in enumerate(all_participants_summary_artifacts):
            if isinstance(res_val, dict) and res_val.get('status') == 'success':
                p_id_eda = res_val.get('participant_id', f'unknown_participant_{res_idx}')
                metrics_eda = res_val.get('analysis_outputs', {}).get('metrics', {})
                for cond_name in EMOTIONAL_CONDITIONS:
                    phasic_mean_key = f'eda_phasic_mean_{cond_name}'
                    scr_count_key = f'eda_scr_count_{cond_name}'
                    if phasic_mean_key in metrics_eda:
                        eda_features_data_list.append({'participant_id': p_id_eda, 'condition': cond_name, 'metric_type': 'phasic_mean', 'value': metrics_eda[phasic_mean_key]})
                    if scr_count_key in metrics_eda:
                         eda_features_data_list.append({'participant_id': p_id_eda, 'condition': cond_name, 'metric_type': 'scr_count', 'value': metrics_eda[scr_count_key]})

        # --- Perform Group-Level Statistical Analyses (moved from pilot_orchestrator.py) ---
        
        # WP1 ANOVA
        if all_avg_plv_wp1_dfs:
            group_stim_plv_data_wp1 = pd.concat(all_avg_plv_wp1_dfs, ignore_index=True)
            group_baseline_plv_data_wp1 = pd.DataFrame()
            if all_baseline_plv_dfs:
                temp_baseline_df = pd.concat(all_baseline_plv_dfs, ignore_index=True)
                if not temp_baseline_df.empty:
                    group_baseline_plv_data_wp1 = temp_baseline_df.groupby(['participant_id', 'eeg_band', 'modality_pair'])['plv'].mean().reset_index()
                    group_baseline_plv_data_wp1['condition'] = 'Baseline'
                    group_baseline_plv_data_wp1['modality_pair'] = group_baseline_plv_data_wp1['modality_pair'].str.replace('_Baseline', '', regex=False)
            combined_plv_data_wp1 = pd.concat([group_stim_plv_data_wp1, group_baseline_plv_data_wp1], ignore_index=True)
            if not combined_plv_data_wp1.empty:
                self.logger.info(f"WP1: Aggregated PLV data from {combined_plv_data_wp1['participant_id'].nunique()} participants for group ANOVA.")
                for eeg_band in config.PLV_EEG_BANDS.keys():
                    for modality in ['EEG-HRV', 'EEG-EDA']:
                        self.logger.info(f"--- WP1 ANOVA for: {eeg_band} & {modality} ---")
                        df_subset = combined_plv_data_wp1[(combined_plv_data_wp1['eeg_band'] == eeg_band) & (combined_plv_data_wp1['modality_pair'] == modality)].copy()
                        if not df_subset.empty and df_subset['participant_id'].nunique() > 1:
                            anova_res = self.analysis_service.run_repeated_measures_anova(data_df=df_subset, dv='plv', within='condition', subject='participant_id')
                            self.logger.info(f"WP1 ANOVA Results ({eeg_band}, {modality}):\n{anova_res}")
                            if anova_res is not None and not anova_res.empty:
                                p_vals = anova_res['p-unc'].dropna().tolist()
                                if p_vals:
                                    reject, p_corr = apply_fdr_correction(p_vals, alpha=0.05)
                                    anova_res.loc[anova_res['p-unc'].notna(), 'p-corr-fdr'] = p_corr
                                    anova_res.loc[anova_res['p-unc'].notna(), 'reject_fdr'] = reject
                                    self.logger.info(f"WP1 ANOVA ({eeg_band}, {modality}) with FDR:\n{anova_res}")
                                cond_eff = anova_res[anova_res['Source'] == 'condition']
                                if not cond_eff.empty and cond_eff['p-corr-fdr'].iloc[0] < 0.05:
                                    self.logger.info(f"WP1 ANOVA ({eeg_band}, {modality}) 'condition' effect SIGNIFICANT. Post-hocs...")
                                    posthoc_res = pg.pairwise_ttests(data=df_subset, dv='plv', within='condition', subject='participant_id', padjust='fdr_bh')
                                    self.logger.info(f"WP1 Post-hoc ({eeg_band}, {modality}):\n{posthoc_res}")
                                    posthoc_res.to_csv(os.path.join(self.group_results_dir, f"group_posthoc_wp1_plv_{eeg_band}_{modality.replace('EEG-','')}.csv"))
                                anova_res.to_csv(os.path.join(self.group_results_dir, f"group_anova_wp1_plv_{eeg_band}_{modality.replace('EEG-','')}.csv"))
                                self.plotting_service.plot_anova_results("GROUP", anova_res, df_subset, 'plv', 'condition', f"WP1: PLV ({eeg_band}, {modality})", f"wp1_plv_{eeg_band}_{modality.replace('EEG-','')}")
                        else:
                            self.logger.warning(f"WP1: Not enough data for ANOVA ({eeg_band}, {modality}).")
            else:
                self.logger.warning("WP1: No combined PLV data for group ANOVA.")

        # WP2 Correlations
        if all_trial_plv_wp1_dfs_for_wp2 and all_survey_dfs_for_wp2:
            # ... (WP2 aggregation and correlation logic as in pilot_orchestrator.py) ...
            # Ensure results are stored in self.all_group_level_results
            # Example: self.all_group_level_results[f"wp2_corr_arousal_vs_hrv"] = corr_wp2_hrv.to_dict(orient='records')
            # Save CSVs and plots using self.group_results_dir and self.plotting_service
            pass # Placeholder for full WP2 logic

        # WP3 Correlation
        if wp3_data_list:
            wp3_group_df = pd.DataFrame(wp3_data_list).dropna()
            if not wp3_group_df.empty and len(wp3_group_df) >= 3:
                corr_wp3 = self.analysis_service.run_correlation_analysis(wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], name1='Baseline_RMSSD', name2=f'Avg_PLV_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}_HRV_Negative')
                self.logger.info(f"WP3 Correlation Results:\n{corr_wp3}")
                if corr_wp3 is not None and not corr_wp3.empty:
                    self.all_group_level_results['wp3_correlation_rmssd_plv'] = corr_wp3.to_dict(orient='records')
                    corr_wp3.to_csv(os.path.join(self.group_results_dir, f"group_corr_wp3_rmssd_vs_plv_neg_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}.csv"))
                    self.plotting_service.plot_correlation("GROUP", wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], "Baseline RMSSD (ms)", f"Avg PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP3}-HRV-Negative)", f"WP3: RMSSD vs Negative PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP3})", f"wp3_rmssd_plv_neg_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}", corr_wp3)
            else:
                self.logger.warning("WP3: Not enough valid data for RMSSD vs PLV correlation.")

        # WP4 Correlations
        if wp4_fai_list and all_avg_plv_wp1_dfs:
            # ... (WP4 aggregation and correlation logic as in pilot_orchestrator.py) ...
            # Ensure results are stored in self.all_group_level_results
            # Save CSVs and plots using self.group_results_dir and self.plotting_service
            pass # Placeholder for full WP4 logic

        # EDA Features ANOVA
        if eda_features_data_list:
            df_eda_features_group = pd.DataFrame(eda_features_data_list).dropna(subset=['value'])
            if not df_eda_features_group.empty:
                self.logger.info(f"Aggregated EDA features from {df_eda_features_group['participant_id'].nunique()} participants for group ANOVA.")
                for metric_type in ['phasic_mean', 'scr_count']:
                    df_eda_subset = df_eda_features_group[df_eda_features_group['metric_type'] == metric_type]
                    if not df_eda_subset.empty and df_eda_subset['participant_id'].nunique() > 1:
                        self.logger.info(f"--- EDA ANOVA for: {metric_type} ---")
                        anova_res_eda = self.analysis_service.run_repeated_measures_anova(data_df=df_eda_subset, dv='value', within='condition', subject='participant_id')
                        self.logger.info(f"EDA ANOVA Results ({metric_type}):\n{anova_res_eda}")
                        if anova_res_eda is not None and not anova_res_eda.empty:
                            self.all_group_level_results[f'eda_anova_{metric_type}'] = anova_res_eda.to_dict(orient='list')
                            cond_eff_eda = anova_res_eda[anova_res_eda['Source'] == 'condition']
                            if not cond_eff_eda.empty and cond_eff_eda['p-unc'].iloc[0] < 0.05:
                                self.logger.info(f"EDA ANOVA ({metric_type}) 'condition' effect SIGNIFICANT. Post-hocs...")
                                posthoc_eda_res = pg.pairwise_ttests(data=df_eda_subset, dv='value', within='condition', subject='participant_id', padjust='fdr_bh')
                                self.logger.info(f"EDA Post-hoc ({metric_type}):\n{posthoc_eda_res}")
                                posthoc_eda_res.to_csv(os.path.join(self.group_results_dir, f"group_posthoc_eda_{metric_type}.csv"))
                            anova_res_eda.to_csv(os.path.join(self.group_results_dir, f"group_anova_eda_{metric_type}.csv"))
                            self.plotting_service.plot_anova_results("GROUP", anova_res_eda, df_eda_subset, 'value', 'condition', f"EDA {metric_type.replace('_', ' ').title()}", f"eda_{metric_type}_by_condition")
        
        # --- Save All Group Analysis Results to JSON ---
        group_results_summary_path = os.path.join(self.output_base_dir, "group_analysis_results.json")
        try:
            with open(group_results_summary_path, 'w') as f:
                serializable_results = {}
                for key, value in self.all_group_level_results.items():
                    if isinstance(value, pd.DataFrame):
                        serializable_results[key] = value.to_dict(orient='list')
                    elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                        serializable_results[key] = value
                    elif isinstance(value, dict):
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = str(value)
                json.dump(serializable_results, f, indent=4, cls=NpEncoder)
            self.logger.info(f"All group analysis results saved to {group_results_summary_path}")
        except Exception as e_save:
            self.logger.error(f"Failed to save all group analysis results: {e_save}", exc_info=True)

        self.logger.info("--- Group-Level Analysis Completed ---")

    # Placeholder methods for WP2 and WP4 to be filled with their specific aggregation and analysis logic
    def _run_wp2_analysis(self, all_participants_summary_artifacts):
        self.logger.info("--- Running Group Analysis: WP2 - Arousal vs. PLV Correlation ---")
        # ... (Full WP2 aggregation, correlation, FDR, saving, plotting logic here) ...
        pass

    def _run_wp4_analysis(self, all_participants_summary_artifacts):
        self.logger.info("--- Running Group Analysis: WP4 - FAI vs. PLV Correlation ---")
        # ... (Full WP4 aggregation, correlation, FDR, saving, plotting logic here) ...
        pass