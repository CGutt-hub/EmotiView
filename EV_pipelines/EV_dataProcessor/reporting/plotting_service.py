import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .. import config

class PlottingService:
    def __init__(self, logger, output_dir_base):
        self.logger = logger
        self.output_dir_base = output_dir_base
        os.makedirs(self.output_dir_base, exist_ok=True)
        self.logger.info(f"PlottingService initialized. Plots will be saved in subdirectories of: {self.output_dir_base}")

    def _save_plot(self, fig, participant_id, plot_name, subdirectory="general"):
        """Helper function to save plots."""
        participant_plot_dir = os.path.join(self.output_dir_base, participant_id, subdirectory)
        os.makedirs(participant_plot_dir, exist_ok=True)
        plot_path = os.path.join(participant_plot_dir, f"{plot_name}.{config.REPORTING_FIGURE_FORMAT}")
        try:
            fig.savefig(plot_path)
            plt.close(fig)
            self.logger.info(f"Plot saved: {plot_path}")
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to save plot {plot_name} for {participant_id}: {e}")
            plt.close(fig)
            return None

    def plot_plv_results(self, participant_id, plv_df, analysis_name="plv_summary"):
        """
        Plots PLV results, e.g., averaged PLV per condition.
        plv_df: DataFrame with columns like 'condition', 'modality_pair', 'eeg_band', 'plv_mean', 'plv_std_err'.
        """
        fig, ax = plt.subplots()
        if plv_df is not None and not plv_df.empty:
            # Example: Bar plot of mean PLV by condition, hue by modality_pair/eeg_band
            # This requires plv_df to be structured appropriately.
            try:
                sns.barplot(data=plv_df, x='condition', y='plv', hue='modality_pair', errorbar='sd', ax=ax) # 'plv' should be the value column
                ax.set_title(f"Average PLV by Condition - {participant_id}")
                ax.set_ylabel("Mean PLV")
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()
            except Exception as e:
                self.logger.warning(f"Could not generate detailed PLV plot for {participant_id} due to: {e}. Plotting placeholder.")
                ax.text(0.5, 0.5, "PLV Plot (Data Error)", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "PLV Plot (No Data)", ha='center', va='center')
            ax.set_title(f"PLV Data - {participant_id}")

        plot_name = f"{analysis_name}"
        self._save_plot(fig, participant_id, plot_name, subdirectory="plv_plots")

    def plot_correlation(self, participant_id, x_data, y_data, x_label, y_label, title, analysis_name, corr_results=None):
        """
        Plots a scatter plot for correlation analysis with regression line and stats.
        corr_results: DataFrame row from pingouin's pg.corr or a similar dict.
        """
        fig, ax = plt.subplots()
        plot_title = f"{title} - {participant_id}"
        
        if corr_results is not None:
            r_val = corr_results.get('r', np.nan)
            p_val = corr_results.get('p-val', np.nan)
            n_val = corr_results.get('n', len(x_data) if x_data is not None else 'N/A')
            if isinstance(r_val, pd.Series): r_val = r_val.iloc[0] # Handle if full df row passed
            if isinstance(p_val, pd.Series): p_val = p_val.iloc[0]
            plot_title += f"\n(n={n_val}, r={r_val:.3f}, p={p_val:.3f})"

        sns.scatterplot(x=x_data, y=y_data, ax=ax)
        # Optional: Add regression line
        # sns.regplot(x=x_data, y=y_data, ax=ax, scatter=False, color='red')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} - {participant_id}")
        ax.set_title(plot_title) # Set the potentially longer title
        plot_name = f"correlation_{analysis_name}"
        fig.tight_layout()
        self._save_plot(fig, participant_id, plot_name, subdirectory="correlation_plots")

    def plot_anova_results(self, participant_id, anova_results_df, data_for_plot_df, dv_col, factor_col, title, analysis_name):
        """
        Visualizes ANOVA results (e.g., bar plot of means with error bars).
        anova_results_df: DataFrame from pingouin's ANOVA.
        data_for_plot_df: DataFrame in long format with dv_col, factor_col, and subject_id for plotting means.
        """
        fig, ax = plt.subplots()
        plot_title = f"{title} - {participant_id}"

        if data_for_plot_df is not None and not data_for_plot_df.empty:
            sns.barplot(data=data_for_plot_df, x=factor_col, y=dv_col, errorbar='se', ax=ax, capsize=.1)
            ax.set_ylabel(f"Mean {dv_col}")
            
            # Add ANOVA stats to the plot title or as text
            if anova_results_df is not None and not anova_results_df.empty:
                # Find the row for the main effect of factor_col
                effect_row = anova_results_df[anova_results_df['Source'] == factor_col]
                if not effect_row.empty:
                    f_val = effect_row['F'].iloc[0]
                    p_val = effect_row['p-unc'].iloc[0]
                    p_corr_val = effect_row.get('p-corr-fdr', p_val) # Use corrected if available
                    df1 = effect_row['ddof1'].iloc[0]
                    df2 = effect_row['ddof2'].iloc[0]
                    plot_title += f"\n{factor_col}: F({df1},{df2})={f_val:.2f}, p={p_val:.3f}"
                    if 'p-corr-fdr' in effect_row:
                        plot_title += f", p_fdr={p_corr_val:.3f}"
        else:
            ax.text(0.5, 0.5, "ANOVA Plot (No Data for Means)", ha='center', va='center')
        ax.set_title(plot_title)
        fig.tight_layout()
        plot_name = f"anova_{analysis_name}"
        self._save_plot(fig, participant_id, plot_name, subdirectory="anova_plots")

    def plot_fnirs_contrast_results(self, participant_id, contrast_name, contrast_obj_or_path, analysis_name="fnirs_contrast"):
        """
        Placeholder for plotting fNIRS contrast results (e.g., topographic maps).
        contrast_obj_or_path: MNE contrast object or path to saved contrast DataFrame.
        """
        fig, ax = plt.subplots()
        # In a real implementation, you would load the contrast object/data
        # and use mne_nirs.visualisation.plot_glm_surface_projection or similar
        # This requires sensor locations in the fNIRS info object.
        ax.text(0.5, 0.5, f"fNIRS Contrast Plot\n{contrast_name}\n(Placeholder)", ha='center', va='center', fontsize=10)
        ax.set_title(f"fNIRS Contrast: {contrast_name} - {participant_id}")
        plot_name = f"{analysis_name}_{contrast_name.replace(' ', '_')}"
        self._save_plot(fig, participant_id, plot_name, subdirectory="fnirs_glm_plots")