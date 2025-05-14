import os
import matplotlib.pyplot as plt
import mne
from .. import config # Relative import
import pandas as pd
import seaborn as sns

class PlottingService:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("PlottingService initialized.")

    def plot_eeg_psd(self, raw_eeg, participant_id, output_dir):
        """Plots and saves EEG Power Spectral Density."""
        if raw_eeg is None:
            self.logger.warning("PlottingService - No EEG data to plot PSD.")
            return
        try:
            self.logger.info(f"PlottingService - Generating EEG PSD plot for {participant_id}.")
            fig = raw_eeg.compute_psd(picks='eeg', fmax=config.EEG_PSD_FMAX, n_fft=2048).plot(show=False, average=True)
            plot_path = os.path.join(output_dir, f"{participant_id}_eeg_psd.png")
            fig.savefig(plot_path)
            plt.close(fig)
            self.logger.info(f"PlottingService - EEG PSD plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"PlottingService - Error plotting EEG PSD for {participant_id}: {e}", exc_info=True)

    def plot_fnirs_glm_contrast(self, contrast_evoked, participant_id, contrast_name, output_dir):
        """Plots and saves fNIRS GLM contrast (e.g., topomap). Conceptual."""
        if contrast_evoked is None:
            self.logger.warning(f"PlottingService - No fNIRS contrast data for {contrast_name} to plot.")
            return
        try:
            self.logger.info(f"PlottingService - Generating fNIRS GLM contrast plot for {participant_id} - {contrast_name}.")
            # Example: This assumes contrast_evoked is an MNE Evoked object
            # fig = contrast_evoked.plot_topomap(ch_type='hbo', show=False, times='peaks') # Adjust as needed
            # plot_path = os.path.join(output_dir, f"{participant_id}_fnirs_contrast_{contrast_name}_hbo_topo.png")
            # fig.savefig(plot_path)
            # plt.close(fig)
            # self.logger.info(f"PlottingService - fNIRS contrast plot saved to {plot_path}")
            self.logger.info("PlottingService - fNIRS GLM contrast plotting is conceptual and needs implementation.")
        except Exception as e:
            self.logger.error(f"PlottingService - Error plotting fNIRS contrast for {participant_id}: {e}", exc_info=True)

    def plot_group_plv_by_condition_roi(self, df_plv_long, band, autonomic, output_dir):
        """Plots group-averaged PLV by condition and ROI."""
        if df_plv_long.empty:
            self.logger.warning(f"PlottingService - No {band}-{autonomic} PLV data for group plot.")
            return
        try:
            self.logger.info(f"PlottingService - Generating group {band}-{autonomic} PLV plot.")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_plv_long, x='condition', y='plv_value', hue='roi', ax=ax, errorbar='ci') # Use errorbar for confidence intervals
            ax.set_title(f'Group Average {band.capitalize()}-{autonomic.upper()} PLV by Condition and ROI')
            ax.set_ylabel('Average PLV')
            ax.set_xlabel('Condition')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f"group_plv_{band}_{autonomic}_condition_roi.png")
            fig.savefig(plot_path)
            plt.close(fig)
            self.logger.info(f"PlottingService - Group {band}-{autonomic} PLV plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"PlottingService - Error plotting group {band}-{autonomic} PLV: {e}", exc_info=True)

    def plot_group_fai_by_condition_pair(self, df_fai_long, output_dir):
        """Plots group-averaged FAI by condition and hemisphere pair."""
        if df_fai_long.empty:
            self.logger.warning("PlottingService - No FAI data for group plot.")
            return
        try:
            self.logger.info("PlottingService - Generating group FAI plot.")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_fai_long, x='condition', y='fai_value', hue='hemisphere_pair', ax=ax, errorbar='ci') # Use errorbar
            ax.set_title('Group Average FAI by Condition and Hemisphere Pair')
            ax.set_ylabel('Average FAI (log difference)')
            ax.set_xlabel('Condition')
            plt.axhline(0, color='grey', linestyle='--') # Add line at 0 for FAI
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "group_fai_condition_pair.png")
            fig.savefig(plot_path)
            plt.close(fig)
            self.logger.info(f"PlottingService - Group FAI plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"PlottingService - Error plotting group FAI: {e}", exc_info=True)

    def plot_group_correlations(self, df_agg, correlation_results_df, output_dir):
        """Plots scatter plots for group correlations."""
        if correlation_results_df.empty or df_agg.empty:
            self.logger.warning("PlottingService - No correlation results or aggregated data for group correlation plots.")
            return
        try:
            self.logger.info("PlottingService - Generating group correlation plots.")

            # Filter for correlations that were actually computed and have a p-value (even if not significant before FDR)
            computed_corrs = correlation_results_df.dropna(subset=['p-val'])

            for index, row in computed_corrs.iterrows():
                var1_name = row['name1']
                var2_name = row['name2']
                desc = row['description']
                r_val = row['r']
                p_val = row['p-val'] # Use original p-val for plotting label

                if var1_name in df_agg.columns and var2_name in df_agg.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.regplot(data=df_agg, x=var1_name, y=var2_name, ax=ax)
                    ax.set_title(f'{desc}\n(r={r_val:.3f}, p={p_val:.3f})')
                    ax.set_xlabel(var1_name)
                    ax.set_ylabel(var2_name)
                    plt.tight_layout()

                    # Sanitize filename
                    safe_desc = re.sub(r'[\\/*?:"<>|]', '', desc).replace(' ', '_')[:50] # Basic sanitization
                    plot_path = os.path.join(output_dir, f"corr_{safe_desc}.png")
                    fig.savefig(plot_path)
                    plt.close(fig)
                    self.logger.info(f"PlottingService - Correlation plot saved to {plot_path}")
                else:
                     self.logger.warning(f"PlottingService - Could not plot correlation for '{desc}': variables not found in aggregated data.")

        except Exception as e:
            self.logger.error(f"PlottingService - Error plotting group correlations: {e}", exc_info=True)

    def generate_group_plots(self, aggregated_metrics_df, statistical_results, output_dir):
        """Orchestrates generation of all group-level plots."""
        if aggregated_metrics_df.empty:
            self.logger.info("PlottingService - No aggregated metrics for group plots.")
            return
        self.logger.info("PlottingService - Generating group plots (conceptual).")
        
        # PLV Plots
        plv_types = [('alpha', 'hrv'), ('beta', 'hrv'), ('alpha', 'eda'), ('beta', 'eda')]
        for band, autonomic in plv_types:
             plv_cols = [col for col in aggregated_metrics_df.columns if col.startswith(f'plv_avg_{band}_') and f'_{autonomic}_' in col]
             if plv_cols:
                df_plv_long = aggregated_metrics_df[['participant_id'] + plv_cols].melt(
                    id_vars='participant_id', value_vars=plv_cols, var_name='plv_metric_full', value_name='plv_value'
                ).dropna(subset=['plv_value'])
                # Re-parse ROI and Condition - this is slightly inefficient but keeps plotting service independent
                def parse_plv_metric(metric_str, band_in, autonomic_in):
                    pattern = re.compile(f"plv_avg_{band_in}_(.*)_{autonomic_in}_(.*)")
                    match = pattern.match(metric_str)
                    if match: return match.group(1), match.group(2)
                    return None, None
                parsed_cols = df_plv_long['plv_metric_full'].apply(lambda x: pd.Series(parse_plv_metric(x, band, autonomic)))
                parsed_cols.columns = ['roi', 'condition']
                df_plv_long = pd.concat([df_plv_long, parsed_cols], axis=1).dropna(subset=['roi', 'condition'])
                
                if not df_plv_long.empty:
                     self.plot_group_plv_by_condition_roi(df_plv_long, band, autonomic, output_dir)

        # FAI Plots
        fai_cols = [col for col in aggregated_metrics_df.columns if col.startswith('fai_alpha_')]
        if fai_cols:
            df_fai_long = aggregated_metrics_df[['participant_id'] + fai_cols].melt(
                id_vars='participant_id', value_vars=fai_cols, var_name='fai_metric_full', value_name='fai_value'
            ).dropna(subset=['fai_value'])
            # Re-parse Pair and Condition
            def parse_fai_metric(metric_str):
                match = re.match(r"fai_alpha_(F[p|AF]?\d_F[p|AF]?\d)_(.*)", metric_str)
                if match: return match.group(1), match.group(2)
                return None, None
            parsed_fai_cols = df_fai_long['fai_metric_full'].apply(lambda x: pd.Series(parse_fai_metric(x)))
            parsed_fai_cols.columns = ['hemisphere_pair', 'condition']
            df_fai_long = pd.concat([df_fai_long, parsed_fai_cols], axis=1).dropna(subset=['hemisphere_pair', 'condition'])

            if not df_fai_long.empty:
                self.plot_group_fai_by_condition_pair(df_fai_long, output_dir)

        # Correlation Plots
        correlation_results_df = statistical_results.get('correlations')
        if correlation_results_df is not None:
             self.plot_group_correlations(aggregated_metrics_df, correlation_results_df, output_dir)

        self.logger.info("PlottingService - Group plot generation finished.")