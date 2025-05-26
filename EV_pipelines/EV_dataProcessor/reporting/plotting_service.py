# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\reporting\plotting_service.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Import pandas for type checking
import os
from ..orchestrators import config

class PlottingService:
    def __init__(self, logger, output_dir_base):
        self.logger = logger
        self.output_dir_base = output_dir_base 
        self.logger.info(f"PlottingService initialized. Plots will be saved in subdirectories of: {self.output_dir_base}")

    def _save_plot(self, fig, participant_id_or_group, plot_name, subdirectory="general"):
        """Helper function to save plots."""
        if participant_id_or_group == "GROUP":
            plot_dir = os.path.join(self.output_dir_base, subdirectory)
        else: 
            plot_dir = os.path.join(self.output_dir_base, subdirectory)
        
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{plot_name}.{config.REPORTING_FIGURE_FORMAT}")
        try:
            fig.savefig(plot_path, dpi=config.REPORTING_DPI)
            plt.close(fig) 
            self.logger.info(f"Plot saved: {plot_path}")
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to save plot {plot_name} for {participant_id_or_group}: {e}")
            plt.close(fig) 
            return None

    def plot_plv_results(self, participant_id, plv_df, analysis_name="plv_summary"):
        """
        Plots PLV results, e.g., averaged PLV per condition and band/modality.
        """
        fig = None 
        if plv_df is not None and not plv_df.empty:
            try:
                unique_bands = plv_df['eeg_band'].unique()
                unique_modalities = plv_df['modality_pair'].unique()

                if len(unique_bands) > 1 or len(unique_modalities) > 1:
                    g = sns.catplot(data=plv_df, x='condition', y='plv', 
                                    hue='modality_pair', col='eeg_band', 
                                    kind='bar', errorbar='se', capsize=.1, sharey=False,
                                    height=4, aspect=1.2) 
                    g.set_axis_labels("Condition", "Mean PLV")
                    g.set_titles("{col_name} Band")
                    fig = g.fig
                    fig.suptitle(f"Average PLV by Condition - {participant_id}", y=1.03)
                else: 
                    fig_simple, ax_simple = plt.subplots(figsize=(8,6))
                    sns.barplot(data=plv_df, x='condition', y='plv', hue='modality_pair', errorbar='se', ax=ax_simple, capsize=.1)
                    band_title = unique_bands[0] if len(unique_bands) > 0 else "Unknown Band"
                    ax_simple.set_title(f"Average PLV by Condition ({band_title}) - {participant_id}")
                    ax_simple.set_ylabel("Mean PLV")
                    fig = fig_simple 
                
                if fig:
                    for ax_g in fig.axes: 
                        ax_g.tick_params(axis='x', rotation=45)
                    fig.tight_layout(rect=[0, 0, 1, 0.96] if fig.texts else None) 
                
            except Exception as e:
                self.logger.warning(f"Could not generate detailed PLV plot for {participant_id} due to: {e}. Plotting placeholder.")
                fig, ax_placeholder = plt.subplots() 
                ax_placeholder.set_title(f"Average PLV by Condition - {participant_id}")
                ax_placeholder.text(0.5, 0.5, "PLV Plot (Data Error)", ha='center', va='center')
        
        if fig is None: 
            fig, ax_placeholder = plt.subplots()
            ax_placeholder.text(0.5, 0.5, "PLV Plot (No Data or Error)", ha='center', va='center')
            ax_placeholder.set_title(f"PLV Data - {participant_id}")

        plot_name = f"{analysis_name}"
        self._save_plot(fig, participant_id, plot_name, subdirectory="plv_plots")

    def plot_correlation(self, participant_id, x_data, y_data, x_label, y_label, title, analysis_name, corr_results=None):
        """
        Plots a scatter plot for correlation analysis with regression line and stats.
        """
        fig, ax = plt.subplots(figsize=(8,6))
        plot_title_text = f"{title}\nParticipant: {participant_id}" 
        
        stats_text = ""
        if corr_results is not None and not corr_results.empty:
            r_val = corr_results['r'].iloc[0] if 'r' in corr_results.columns else np.nan
            p_val = corr_results['p-val'].iloc[0] if 'p-val' in corr_results.columns else np.nan
            n_val = corr_results['n'].iloc[0] if 'n' in corr_results.columns else (len(x_data) if x_data is not None else 'N/A')
            
            p_corr_fdr_val = corr_results.get('p-corr-fdr') 
            if isinstance(p_corr_fdr_val, pd.Series) and not p_corr_fdr_val.empty:
                p_corr_fdr_val = p_corr_fdr_val.iloc[0]
            
            stats_text = f"n={n_val}, r={r_val:.3f}, p={p_val:.3f}"
            if p_corr_fdr_val is not None and not np.isnan(p_corr_fdr_val) and p_corr_fdr_val != p_val :
                stats_text += f", p_fdr={p_corr_fdr_val:.3f}"
        
        if x_data is not None and y_data is not None and len(x_data) == len(y_data) and len(x_data) > 0:
            sns.scatterplot(x=x_data, y=y_data, ax=ax, s=50, alpha=0.7)
            if len(x_data) > 1 : 
                 sns.regplot(x=x_data, y=y_data, ax=ax, scatter=False, color='red', line_kws={'linewidth':2})
        else:
            ax.text(0.5, 0.5, "Correlation Plot (Insufficient Data)", ha='center', va='center')

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(plot_title_text, fontsize=14)
        if stats_text: 
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

        plot_name = f"correlation_{analysis_name.replace(' ', '_').lower()}"
        fig.tight_layout()
        self._save_plot(fig, participant_id, plot_name, subdirectory="correlation_plots")

    def plot_anova_results(self, participant_id, anova_results_df, data_for_plot_df, dv_col, factor_col, title, analysis_name):
        """
        Visualizes ANOVA results.
        """
        fig, ax = plt.subplots(figsize=(8,6))
        plot_title_text = f"{title}\nParticipant/Group: {participant_id}"

        if data_for_plot_df is not None and not data_for_plot_df.empty and \
           factor_col in data_for_plot_df.columns and dv_col in data_for_plot_df.columns:
            sns.barplot(data=data_for_plot_df, x=factor_col, y=dv_col, errorbar='se', ax=ax, capsize=.1, palette="viridis")
            ax.set_ylabel(f"Mean {dv_col}", fontsize=12)
            ax.set_xlabel(factor_col.capitalize(), fontsize=12)
            
            stats_text = ""
            if anova_results_df is not None and not anova_results_df.empty:
                effect_row = anova_results_df[anova_results_df['Source'] == factor_col]
                if not effect_row.empty:
                    f_val = effect_row['F'].iloc[0]
                    p_val = effect_row['p-unc'].iloc[0]
                    p_corr_val = effect_row['p-corr-fdr'].iloc[0] if 'p-corr-fdr' in effect_row.columns else p_val
                    df1 = effect_row['ddof1'].iloc[0]
                    df2 = effect_row['ddof2'].iloc[0]
                    stats_text = f"{factor_col}: F({df1:.0f},{df2:.0f})={f_val:.2f}, p={p_val:.3f}"
                    if 'p-corr-fdr' in effect_row.columns and p_corr_val != p_val:
                        stats_text += f", p_fdr={p_corr_val:.3f}"
            if stats_text:
                 ax.set_title(f"{plot_title_text}\n{stats_text}", fontsize=14)
            else:
                 ax.set_title(plot_title_text, fontsize=14)
        else:
            ax.text(0.5, 0.5, "ANOVA Plot (No Data for Means)", ha='center', va='center')
            ax.set_title(plot_title_text, fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        plot_name = f"anova_{analysis_name.replace(' ', '_').lower()}"
        self._save_plot(fig, participant_id, plot_name, subdirectory="anova_plots")

    def plot_fnirs_contrast_results(self, participant_id, contrast_name, contrast_obj_or_path, analysis_name="fnirs_contrast"):
        """Plots fNIRS contrast results as a bar plot of effect sizes per channel."""
        
        contrast_df = None
        if isinstance(contrast_obj_or_path, pd.DataFrame): 
            contrast_df = contrast_obj_or_path
        elif hasattr(contrast_obj_or_path, 'to_dataframe'): 
            try:
                contrast_df = contrast_obj_or_path.to_dataframe()
            except Exception as e:
                self.logger.error(f"Error converting fNIRS contrast object to DataFrame: {e}")
        
        if contrast_df is None or contrast_df.empty:
            self.logger.warning(f"Cannot plot fNIRS contrast {contrast_name} for {participant_id}: Invalid or empty data.")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.text(0.5, 0.5, f"fNIRS Contrast Plot\n{contrast_name}\n(Invalid/Empty Data)", ha='center', va='center', fontsize=10)
            ax.set_title(f"fNIRS Contrast: {contrast_name} - {participant_id}")
            plot_name = f"{analysis_name}_{contrast_name.replace(' ', '_').replace('/', '_')}"
            self._save_plot(fig, participant_id, plot_name, subdirectory="fnirs_glm_plots")
            return

        # For true topographic plots (e.g., using mne_nirs.visualisation.plot_glm_surface_projection),
        # you would need:
        # 1. The MNE `info` object for fNIRS data with accurate 3D sensor locations.
        # 2. The GLM contrast object itself (not just the DataFrame).
        # 3. Potentially a surface mesh for projection.
        # Example (conceptual):
        # if hasattr(contrast_obj_or_path, 'plot_components'): # Check if it's a contrast-like object
        #    try:
        #        fig_topo = contrast_obj_or_path.plot_components(info=fnirs_info_with_locations, ...)
        #        self._save_plot(fig_topo, participant_id, f"{plot_name}_topo", subdirectory="fnirs_glm_plots")
        #    except Exception as e_topo:
        #        self.logger.error(f"Could not generate topographic fNIRS plot: {e_topo}")

        if all(col in contrast_df.columns for col in ['ch_name', 'effect', 'Chroma']):
            fig, ax = plt.subplots(figsize=(12, max(6, len(contrast_df['ch_name'].unique()) * 0.35))) 
            sns.barplot(data=contrast_df, y='ch_name', x='effect', hue='Chroma', ax=ax, dodge=True, palette={'hbo': 'red', 'hbr': 'blue'})
            ax.set_title(f"fNIRS GLM Effect Sizes: {contrast_name}\nParticipant: {participant_id}", fontsize=14)
            ax.set_xlabel("Effect Size (e.g., Beta coefficient)", fontsize=12)
            ax.set_ylabel("fNIRS Channel", fontsize=12)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8) 
            ax.tick_params(axis='y', labelsize=8) 
            fig.tight_layout()
        else: 
            fig, ax = plt.subplots(figsize=(8,6))
            ax.text(0.5, 0.5, f"fNIRS Contrast Plot\n{contrast_name}\n(Data Structure Issue or No Data)", ha='center', va='center', fontsize=10)
            ax.set_title(f"fNIRS Contrast: {contrast_name} - {participant_id}")
        
        plot_name = f"{analysis_name}_{contrast_name.replace(' ', '_').replace('/', '_')}" 
        self._save_plot(fig, participant_id, plot_name, subdirectory="fnirs_glm_plots")