# EmotiView: Neural-Autonomic Synchrony and Embodied Integration in Emotional Processing

This repository houses the research proposal, abstract, and supporting materials for the Master's Thesis project investigating the dynamic interplay between neural activity and autonomic nervous system responses during emotional experiences.

**Principal Investigator:** Cagatay Özcan Jagiello Gutt

## Research Abstract

Emotional states are fundamentally embodied, emerging from the dynamic interplay between central neural processing and peripheral physiological adjustments orchestrated by the autonomic nervous system (ANS). While ANS outputs like heart rate variability (HRV) and electrodermal activity (EDA) reflect emotional arousal and valence, understanding the precise temporal coordination between brain activity and these peripheral signals is crucial for elucidating brain-body interactions. This study investigates neural-autonomic phase synchrony during the conscious processing of distinct emotional states (positive, negative, neutral) by quantifying the temporal alignment between cortical and physiological rhythms.

We employ a multimodal approach, simultaneously recording high-temporal-resolution electroencephalography (EEG), electrocardiography (ECG) for HRV analysis (specifically Root Mean Square of Successive Differences, RMSSD), EDA, and functional near-infrared spectroscopy (fNIRS) while participants view validated emotional video clips. Our primary analysis quantifies the Phase Locking Value (PLV) between frontal EEG oscillations (Alpha, Beta bands) and continuous signals derived from HRV (reflecting parasympathetic influence) and phasic EDA (reflecting sympathetic influence). EEG channel selection for PLV analysis is informed by task-related hemodynamic activity measured via fNIRS to focus on functionally relevant cortical areas.

We hypothesize that PLV, indicating brain-body temporal integration, will be significantly modulated by emotional content compared to neutral conditions. We further expect synchrony strength to correlate with subjective arousal ratings. By examining the phase synchrony between brain signals and ANS-mediated physiological outputs, this research provides novel insights into the dynamic, embodied mechanisms underlying emotional experience. Understanding this temporal binding is critical for models of psychophysiological function and may inform assessments of cognitive load or stress regulation capacity.

## Core Research Aims & Hypotheses

This project seeks to understand how the brain and body coordinate during emotional processing, focusing on neural-autonomic phase synchrony. Key hypotheses include:

1.  **Emotional Modulation of Synchrony:** Neural-autonomic synchrony (Phase Locking Value - PLV) will be enhanced during the processing of positive and negative emotional stimuli compared to neutral stimuli, for both brain-heart (EEG-HRV) and brain-sudomotor (EEG-EDA) coupling.
2.  **Synchrony and Subjective Arousal:** The magnitude of neural-autonomic synchrony will positively correlate with subjective ratings of emotional arousal during emotional conditions.
3.  **Baseline Vagal Tone and Task-Related Synchrony:** Individual differences in baseline parasympathetic regulation (resting-state RMSSD) will be associated with the degree of EEG-HRV synchrony during negative emotional stimuli.
4.  **Frontal Asymmetry and Branch-Specific Synchrony:** The direction of prefrontal cortical asymmetry (Frontal Asymmetry Index - FAI) will be differentially associated with the strength of phase synchrony involving distinct autonomic branches (EEG-HRV vs. EEG-EDA).

For a comprehensive understanding of the theoretical background, detailed methodology, and specific work packages, please refer to the full proposal document.

## Methodology Overview

A multimodal experimental design is employed, involving:

*   **Stimuli:** Standardized, emotionally evocative video clips (positive, negative, neutral) from the E-MOVIE database.
*   **Participants:** Healthy young adults, screened for relevant criteria.
*   **Data Acquisition:** Simultaneous recording of:
    *   **Electroencephalography (EEG):** To measure prefrontal neural dynamics.
    *   **Functional Near-Infrared Spectroscopy (fNIRS):** To localize hemodynamic activity in prefrontal and parietal regions, informing EEG channel selection.
    *   **Electrocardiography (ECG):** For Heart Rate Variability (HRV) analysis.
    *   **Electrodermal Activity (EDA):** To measure sympathetic nervous system activity.
*   **Subjective Measures:** Self-Assessment Manikin (SAM) for valence and arousal, Positive and Negative Affect Schedule (PANAS), and Behavioural Inhibition/Approach System (BIS/BAS) scales.

## Repository Contents

This repository primarily contains the core research documentation:

*   **`EV_proposal/EV_proposal.tex`**: The full LaTeX source for the Master's Thesis proposal. This document provides an in-depth description of the research background, aims, hypotheses, detailed methodology, analysis plan, and project timeline.
*   **`EV_abstract/EV_abstract.tex`**: The LaTeX source for a conference-style abstract summarizing the research.
*   **`EV_pipelines/`**: Contains the Python-based analysis pipeline developed to implement the methods described in the proposal. This includes modules for data loading, preprocessing of each modality, feature extraction, synchrony analysis, and basic reporting.
    *   `pilot_orchestrator.py`: The main script for running the analysis on pilot data.
    *   `config.py`: Central configuration file for the pipeline parameters.
*   **`rawData/pilotData/`**: Placeholder for storing raw experimental data (not version controlled by default due to size, but structure is indicated).
*   **`EV_results/`**: Directory where the pipeline saves processed data, analysis metrics, and plots.

## Analysis Pipeline

To operationalize the methods outlined in the research proposal, a modular Python-based analysis pipeline has been developed and is located in the `EV_pipelines/` directory. This pipeline is designed to:

*   Load and parse multi-modal raw data (EEG, fNIRS, ECG, EDA, questionnaires).
*   Perform standardized preprocessing steps specific to each physiological modality.
*   Extract key features and metrics (e.g., EEG power, FAI, RMSSD, fNIRS ROI activation, PLV).
*   Generate participant-level results and aggregated summaries.

While the detailed workings of the pipeline are contained within the `EV_pipelines/` subfolder (including its own configuration in `EV_pipelines/config.py`), its design and implementation are directly guided by the analytical requirements of this research project. Users interested in the technical implementation of the analysis should refer to the scripts within that directory.

**Note:** This README focuses on the research project. For detailed instructions on setting up and running the Python analysis pipeline, please refer to documentation that may be provided specifically for the `EV_pipelines/` components or by inspecting the `pilot_orchestrator.py` and `config.py` files.

## Project Status

Current Status: Piloting and Initial Pipeline Development Phase.

## Contact Information

Cagatay Özcan Jagiello Gutt
*ORCID: https://orcid.org/0000-0002-1774-532X*