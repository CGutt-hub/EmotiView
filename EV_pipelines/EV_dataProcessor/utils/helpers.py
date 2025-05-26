# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\utils\helpers.py
import logging
import os
import sys
import mne
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from ..orchestrators import config # Corrected import path

# --- EEG Channel Selection Helper (for fNIRS guidance) ---

def select_eeg_channels_by_fnirs_rois(eeg_info, fnirs_active_rois, logger):
    """
    Selects EEG channels based on a list of active fNIRS ROI names.

    Args:
        eeg_info (mne.Info): MNE Info object from the EEG data.
        fnirs_active_rois (list): List of names of fNIRS ROIs identified as active.
        logger (logging.Logger): Logger instance.

    Returns:
        list: List of EEG channel names corresponding to the active fNIRS ROIs.
    """
    if not fnirs_active_rois:
        logger.warning("No active fNIRS ROIs provided for EEG channel selection. Using predefined channels if available.")
        # Fallback to predefined channels if no active ROIs
        if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
             selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
             if not selected_channels:
                 logger.error("None of the predefined EEG channels found in EEG data.")
             else:
                 logger.info(f"Using predefined EEG channels as fallback: {selected_channels}")
             return selected_channels
        else:
             logger.error("No active fNIRS ROIs and no predefined EEG channels specified. Cannot select EEG channels for PLV.")
             return []


    selected_eeg_channels = set()
    strategy = getattr(config, 'EEG_CHANNEL_SELECTION_STRATEGY_FOR_PLV', 'mapping')

    if strategy == 'mapping':
        mapping = getattr(config, 'FNIRS_ROI_TO_EEG_CHANNELS_MAP', {})
        if not mapping:
            logger.error("config.FNIRS_ROI_TO_EEG_CHANNELS_MAP is not defined for 'mapping' strategy. Cannot select EEG channels.")
            # Fallback to predefined if mapping is missing
            if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
                 selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
                 if not selected_channels:
                     logger.error("None of the predefined EEG channels found in EEG data.")
                 else:
                     logger.info(f"Using predefined EEG channels as fallback: {selected_channels}")
                 return selected_channels
            else:
                 logger.error("No active fNIRS ROIs, no mapping, and no predefined EEG channels. Cannot select EEG channels for PLV.")
                 return []


        for roi_name in fnirs_active_rois:
            if roi_name in mapping:
                # Add channels from the mapping that are actually present in the EEG data
                channels_for_roi = [ch for ch in mapping[roi_name] if ch in eeg_info['ch_names']]
                if channels_for_roi:
                    selected_eeg_channels.update(channels_for_roi)
                    logger.info(f"Mapped fNIRS ROI '{roi_name}' to EEG channels: {channels_for_roi}")
                else:
                    logger.warning(f"None of the mapped EEG channels for fNIRS ROI '{roi_name}' ({mapping[roi_name]}) found in EEG data.")
            else:
                logger.warning(f"fNIRS ROI '{roi_name}' found in active list but not in config.FNIRS_ROI_TO_EEG_CHANNELS_MAP.")

    elif strategy == 'nearest':
        logger.warning("EEG_CHANNEL_SELECTION_STRATEGY 'nearest' requires fNIRS optode locations and EEG channel locations.")
        logger.warning("This functionality is not fully implemented here and requires spatial data.")
        # Placeholder: In a real implementation, you would load fNIRS and EEG channel locations
        # For now, fall back to predefined or log error.
        if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
             selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
             if not selected_channels:
                 logger.error("None of the predefined EEG channels found in EEG data.")
             else:
                 logger.info(f"Using predefined EEG channels as fallback for 'nearest' strategy: {selected_channels}")
             return selected_channels
        else:
             logger.error("EEG_CHANNEL_SELECTION_STRATEGY 'nearest' selected, but spatial logic is not implemented and no predefined channels available. Cannot select EEG channels.")
             return []

    elif strategy == 'predefined':
         # This strategy ignores fNIRS results and just uses the predefined list
         logger.info("Using 'predefined' EEG channel selection strategy, ignoring fNIRS results.")
         if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
             selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
             if not selected_channels:
                 logger.error("None of the predefined EEG channels found in EEG data.")
             else:
                 logger.info(f"Using predefined EEG channels: {selected_channels}")
             return selected_channels
         else:
             logger.error("EEG_CHANNEL_SELECTION_STRATEGY 'predefined' selected, but config.PREDEFINED_EEG_CHANNELS_FOR_PLV is not defined. Cannot select EEG channels.")
             return []

    else:
        logger.error(f"Unknown EEG_CHANNEL_SELECTION_STRATEGY: {strategy}. Cannot select EEG channels.")
        # Fallback to predefined if strategy is invalid
        if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
             selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
             if not selected_channels:
                 logger.error("None of the predefined EEG channels found in EEG data.")
             else:
                 logger.info(f"Using predefined EEG channels as fallback for invalid strategy: {selected_channels}")
             return selected_channels
        else:
             logger.error("Unknown EEG_CHANNEL_SELECTION_STRATEGY and no predefined channels available. Cannot select EEG channels.")
             return []


    final_selected_channels = list(selected_eeg_channels)
    if not final_selected_channels:
         logger.warning("EEG channel selection based on active fNIRS ROIs resulted in an empty list.")
         # Final fallback if selection process yielded nothing
         if hasattr(config, 'DEFAULT_EEG_CHANNELS_FOR_PLV') and config.DEFAULT_EEG_CHANNELS_FOR_PLV:
             selected_channels = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in eeg_info['ch_names']]
             if not selected_channels:
                 logger.error("None of the predefined EEG channels found in EEG data.")
             else:
                 logger.info(f"Using predefined EEG channels as final fallback: {selected_channels}")
             return selected_channels
         else:
             logger.error("EEG channel selection failed and no predefined channels available. Cannot select EEG channels.")
             return []

    logger.info(f"Final selected EEG channels based on fNIRS guidance: {final_selected_channels}")
    return final_selected_channels

def apply_fdr_correction(p_values, alpha=0.05):
    """
    Applies FDR (Benjamini-Hochberg) correction to a list of p-values.
    Args:
        p_values (list or np.ndarray): List of p-values.
        alpha (float): Significance level.
    Returns:
        tuple: (rejected_hypotheses, corrected_p_values)
               rejected_hypotheses is a boolean array.
               corrected_p_values is an array of FDR-corrected p-values.
    """
    if not isinstance(p_values, (list, np.ndarray)) or len(p_values) == 0:
        return np.array([]), np.array([])
    p_values_array = np.asarray(p_values)
    if np.all(np.isnan(p_values_array)): # Handle case where all p-values are NaN
        return np.array([False] * len(p_values_array)), p_values_array
        
    return fdrcorrection(p_values_array, alpha=alpha, method='indep', is_sorted=False)