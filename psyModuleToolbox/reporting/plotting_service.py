# d:\repoShaggy\EmotiView\psyModuleToolbox\reporting\plotting_service.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Import pandas for type checking
import os
# import mne # For potential future topomaps

class PlottingService:
    def __init__(self, logger, output_dir_base, 
                 reporting_figure_format_config, 
                 reporting_dpi_config):
        self.logger = logger
        self.output_dir_base = output_dir_base 
        
        # Store passed configurations
        self.reporting_figure_format = reporting_figure_format_config
        self.reporting_dpi = reporting_dpi_config
        if not all([self.reporting_figure_format, self.reporting_dpi is not None]):
             self.logger.warning("PlottingService initialized with missing critical configurations (figure_format, dpi). Using defaults if possible or errors may occur.")
        self.logger.info(f"PlottingService initialized. Plots will be saved in subdirectories of: {self.output_dir_base}")

    def _save_plot(self, fig, participant_id_or_group, plot_name, subdirectory="general"):
        """Helper function to save matplotlib figures."""
        plot_dir = os.path.join(self.output_dir_base, subdirectory)
        
        os.makedirs(plot_dir, exist_ok=True)
        # The line using 'config.REPORTING_FIGURE_FORMAT' was removed as it was redundant.
        plot_path = os.path.join(plot_dir, f"{plot_name}.{self.reporting_figure_format}") # Use instance variable
        try:
            fig.savefig(plot_path, dpi=self.reporting_dpi) # Use instance variable
            plt.close(fig) 
            self.logger.info(f"Plot saved: {plot_path}")
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to save plot {plot_name} for {participant_id_or_group}: {e}")
            plt.close(fig) 
            return None

    def plot_plv_results(self, participant_id_or_group, plv_df, analysis_name="plv_summary"):
        """
        Plots PLV results, e.g., averaged PLV per condition and band/modality.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            plv_df (pd.DataFrame): DataFrame with PLV results (expected columns: 'condition', 'plv', 'eeg_band', 'modality_pair').
            analysis_name (str): Name for the analysis (used in filename).
        """
        fig = None 
        if plv_df is not None and not plv_df.empty:
            try:
                unique_bands = plv_df['eeg_band'].unique()
                unique_modalities = plv_df['modality_pair'].unique()

                if len(unique_bands) > 1 or len(unique_modalities) > 1:
                    # Use catplot for multiple bands/modalities to create facets
                    g = sns.catplot(data=plv_df, x='condition', y='plv', 
                                    hue='modality_pair', col='eeg_band', 
                                    kind='bar', errorbar='se', capsize=.1, sharey=False,
                                    height=4, aspect=1.2) 
                    g.set_axis_labels("Condition", "Mean PLV")
                    g.set_titles("{col_name} Band") # Seaborn uses {col_name} for facet titles
                    fig = g.fig
                    fig.suptitle(f"Average PLV by Condition - {participant_id_or_group}", y=1.03)
                else: 
                    # Simpler plot if only one band and one modality (or if facets are not desired)
                    fig_simple, ax_simple = plt.subplots(figsize=(8,6))
                    sns.barplot(data=plv_df, x='condition', y='plv', hue='modality_pair', errorbar='se', ax=ax_simple, capsize=.1)
                    band_title = unique_bands[0] if len(unique_bands) > 0 else "Unknown Band"
                    ax_simple.set_title(f"Average PLV by Condition ({band_title}) - {participant_id_or_group}")
                    ax_simple.set_ylabel("Mean PLV")
                    fig = fig_simple 
                
                if fig: # Ensure a figure object was created
                    for ax_g in fig.axes: # Rotate x-axis labels for all subplots
                        ax_g.tick_params(axis='x', rotation=45)
                    # Adjust layout to prevent suptitle overlap if it exists
                    fig.tight_layout(rect=[0, 0, 1, 0.96] if fig.texts else None) 
                
            except Exception as e:
                self.logger.warning(f"Could not generate detailed PLV plot for {participant_id_or_group} due to: {e}. Plotting placeholder.")
                fig, ax_placeholder = plt.subplots() 
                ax_placeholder.set_title(f"Average PLV by Condition - {participant_id_or_group}")
                ax_placeholder.text(0.5, 0.5, "PLV Plot (Data Error)", ha='center', va='center')
        
        if fig is None: # If no data or an error occurred before fig was assigned
            fig, ax_placeholder = plt.subplots()
            ax_placeholder.text(0.5, 0.5, "PLV Plot (No Data or Error)", ha='center', va='center')
            ax_placeholder.set_title(f"PLV Data - {participant_id_or_group}")

        plot_name = f"{analysis_name}"
        self._save_plot(fig, participant_id_or_group, plot_name, subdirectory="plv_plots")

    def plot_correlation(self, participant_id_or_group, x_data, y_data, x_label, y_label, title, analysis_name, corr_results=None):
        """
        Plots a scatter plot for correlation analysis with regression line and stats.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            x_data (pd.Series or np.array): Data for the x-axis.
            y_data (pd.Series or np.array): Data for the y-axis.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-label.
            title (str): Main title for the plot.
            analysis_name (str): Name for the analysis (used in filename).
            corr_results (pd.DataFrame, optional): DataFrame from pingouin/correlation_analyzer 
                                              (should contain 'r', 'p-val', 'n', and optionally 'p-corr-fdr' or 'p-corr-bonf').
        """
        fig, ax = plt.subplots(figsize=(8,6))
        plot_title_text = f"{title}\nParticipant/Group: {participant_id_or_group}" 
        
        stats_text = ""
        if corr_results is not None and not corr_results.empty:
            # Safely extract correlation statistics
            r_val = corr_results['r'].iloc[0] if 'r' in corr_results.columns and not corr_results['r'].empty else np.nan
            p_val = corr_results['p-val'].iloc[0] if 'p-val' in corr_results.columns and not corr_results['p-val'].empty else np.nan
            n_val = corr_results['n'].iloc[0] if 'n' in corr_results.columns and not corr_results['n'].empty else (len(x_data) if x_data is not None else 'N/A')
            
            p_corr_fdr_val = corr_results.get('p-corr-fdr', pd.Series([np.nan])).iloc[0] # Use .get for safety
            p_corr_bonf_val = corr_results.get('p-corr-bonf', pd.Series([np.nan])).iloc[0] # Use .get for safety
            
            stats_text = f"n={n_val}, r={r_val:.3f}, p={p_val:.3f}"
            if p_corr_fdr_val is not None and not np.isnan(p_corr_fdr_val) and p_corr_fdr_val != p_val :
                stats_text += f", p_fdr={p_corr_fdr_val:.3f}"
            # Optionally add Bonferroni corrected p-value if different and present
            # elif p_corr_bonf_val is not None and not np.isnan(p_corr_bonf_val) and p_corr_bonf_val != p_val:
            #     stats_text += f", p_bonf={p_corr_bonf_val:.3f}"
        
        if x_data is not None and y_data is not None and len(x_data) == len(y_data) and len(x_data) > 0:
            sns.scatterplot(x=x_data, y=y_data, ax=ax, s=50, alpha=0.7)
            if len(x_data) >= 2 : # Need at least 2 points for a regression line
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
        self._save_plot(fig, participant_id_or_group, plot_name, subdirectory="correlation_plots")

    def plot_anova_results(self, participant_id_or_group, anova_results_df, data_for_plot_df, dv_col, factor_col, title, analysis_name):
        """
        Visualizes ANOVA results using a bar plot.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            anova_results_df (pd.DataFrame): DataFrame from pingouin.anova (expected columns: 'Source', 'F', 'p-unc', 'ddof1', 'ddof2', and optionally 'p-corr-fdr' or 'p-corr-bonf').
            data_for_plot_df (pd.DataFrame): DataFrame containing the data used for the plot (e.g., means/SEs per group).
                                            Expected columns: dv_col, factor_col.
            dv_col (str): Name of the dependent variable column.
            factor_col (str): Name of the column for the x-axis (e.g., main within-subject factor).
            title (str): Main title for the plot.
            analysis_name (str): Name for the analysis (used in filename).
        """
        fig, ax = plt.subplots(figsize=(8,6))
        plot_title_text = f"{title}\nParticipant/Group: {participant_id_or_group}"

        if data_for_plot_df is not None and not data_for_plot_df.empty and \
           factor_col in data_for_plot_df.columns and dv_col in data_for_plot_df.columns:
            sns.barplot(data=data_for_plot_df, x=factor_col, y=dv_col, errorbar='se', ax=ax, capsize=.1, palette="viridis")
            ax.set_ylabel(f"Mean {dv_col}", fontsize=12)
            ax.set_xlabel(factor_col.capitalize(), fontsize=12)
            
            stats_text = ""
            if anova_results_df is not None and not anova_results_df.empty:
                # Ensure 'Source' column exists before trying to filter
                if 'Source' in anova_results_df.columns:
                    effect_row = anova_results_df[anova_results_df['Source'] == factor_col]
                    if not effect_row.empty:
                        # Safely extract ANOVA statistics
                        f_val = effect_row['F'].iloc[0] if 'F' in effect_row.columns and not effect_row['F'].empty else np.nan
                        p_val = effect_row['p-unc'].iloc[0] if 'p-unc' in effect_row.columns and not effect_row['p-unc'].empty else np.nan
                        ddof1 = effect_row['ddof1'].iloc[0] if 'ddof1' in effect_row.columns and not effect_row['ddof1'].empty else np.nan
                        ddof2 = effect_row['ddof2'].iloc[0] if 'ddof2' in effect_row.columns and not effect_row['ddof2'].empty else np.nan
                        
                        p_corr_fdr_val = effect_row.get('p-corr-fdr', pd.Series([np.nan])).iloc[0]
                        p_corr_bonf_val = effect_row.get('p-corr-bonf', pd.Series([np.nan])).iloc[0]

                        stats_text = f"{factor_col}: F({ddof1:.0f},{ddof2:.0f})={f_val:.2f}, p={p_val:.3f}"
                        if p_corr_fdr_val is not None and not np.isnan(p_corr_fdr_val) and p_corr_fdr_val != p_val:
                             stats_text += f", p_fdr={p_corr_fdr_val:.3f}"
                        # Optionally add Bonferroni corrected p-value
                        # elif p_corr_bonf_val is not None and not np.isnan(p_corr_bonf_val) and p_corr_bonf_val != p_val:
                        #     stats_text += f", p_bonf={p_corr_bonf_val:.3f}"
                else:
                    self.logger.warning(f"PlottingService - 'Source' column not found in anova_results_df for plot '{title}'. Cannot display F-stats.")

            if stats_text:
                 ax.set_title(f"{plot_title_text}\n{stats_text}", fontsize=14)
            else:
                 ax.set_title(plot_title_text, fontsize=14)
        else:
            ax.text(0.5, 0.5, "ANOVA Plot (No Data for Means or Missing Columns)", ha='center', va='center')
            ax.set_title(plot_title_text, fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        plot_name = f"anova_{analysis_name.replace(' ', '_').lower()}_{factor_col}" # Include factor in filename
        self._save_plot(fig, participant_id_or_group, plot_name, subdirectory="anova_plots")

    def plot_fnirs_contrast_results(self, participant_id_or_group, contrast_name, contrast_obj_or_path, analysis_name="fnirs_contrast"):
        """Plots fNIRS contrast results as a bar plot of effect sizes per channel."""
        
        contrast_df = None
        if isinstance(contrast_obj_or_path, pd.DataFrame): 
            contrast_df = contrast_obj_or_path
        elif hasattr(contrast_obj_or_path, 'to_dataframe'): # Check if it's an MNE contrast object
            try:
                contrast_df = contrast_obj_or_path.to_dataframe()
            except Exception as e:
                self.logger.error(f"Error converting fNIRS contrast object to DataFrame: {e}")
        
        if contrast_df is None or contrast_df.empty:
            self.logger.warning(f"Cannot plot fNIRS contrast {contrast_name} for {participant_id_or_group}: Invalid or empty data.")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.text(0.5, 0.5, f"fNIRS Contrast Plot\n{contrast_name}\n(Invalid/Empty Data)", ha='center', va='center', fontsize=10)
            ax.set_title(f"fNIRS Contrast: {contrast_name} - {participant_id_or_group}")
            plot_name = f"{analysis_name}_{contrast_name.replace(' ', '_').replace('/', '_')}"
            self._save_plot(fig, participant_id_or_group, plot_name, subdirectory="fnirs_glm_plots")
            return

        # This section is for a bar plot of effect sizes per channel.
        # For true topographic plots, more complex MNE-NIRS functions would be needed.
        if all(col in contrast_df.columns for col in ['ch_name', 'effect', 'Chroma']):
            fig, ax = plt.subplots(figsize=(12, max(6, len(contrast_df['ch_name'].unique()) * 0.35))) 
            sns.barplot(data=contrast_df, y='ch_name', x='effect', hue='Chroma', ax=ax, dodge=True, palette={'hbo': 'red', 'hbr': 'blue'})
            ax.set_title(f"fNIRS GLM Effect Sizes: {contrast_name}\nParticipant/Group: {participant_id_or_group}", fontsize=14)
            ax.set_xlabel("Effect Size (e.g., Beta coefficient)", fontsize=12)
            ax.set_ylabel("fNIRS Channel", fontsize=12)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8) 
            ax.tick_params(axis='y', labelsize=8) 
            fig.tight_layout()
        else: 
            self.logger.warning(f"fNIRS contrast DataFrame for {contrast_name} is missing expected columns ('ch_name', 'effect', 'Chroma'). Plotting placeholder.")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.text(0.5, 0.5, f"fNIRS Contrast Plot\n{contrast_name}\n(Data Structure Issue or No Data)", ha='center', va='center', fontsize=10)
            ax.set_title(f"fNIRS Contrast: {contrast_name} - {participant_id_or_group}")
        
        plot_name = f"{analysis_name}_{contrast_name.replace(' ', '_').replace('/', '_')}" 
        self._save_plot(fig, participant_id_or_group, plot_name, subdirectory="fnirs_glm_plots")