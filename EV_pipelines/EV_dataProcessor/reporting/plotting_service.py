import os
import matplotlib.pyplot as plt
import mne
from .. import config # Relative import

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

    def generate_group_plots(self, aggregated_metrics_df, output_dir):
        """Generates and saves group-level plots. Conceptual."""
        if aggregated_metrics_df.empty:
            self.logger.info("PlottingService - No aggregated metrics for group plots.")
            return
        self.logger.info("PlottingService - Generating group plots (conceptual).")
        # Add logic from the original generate_group_plots function here,
        # adapting it to use aggregated_metrics_df.
        # Example: Plotting group-averaged PLV values by condition.