# EmotiView: Neural-Autonomic Synchrony and Embodied Integration

This repository accompanies ongoing research investigating the dynamic interplay between neural activity and autonomic nervous system responses during emotional experiences. Here you'll find the research article, presentations, analysis pipeline, and results—updated in real-time as the project progresses.

**Principal Investigator:** Cagatay Özcan Jagiello Gutt

| Platform | Role | Contents |
|----------|------|----------|
| **[OSF](https://osf.io/gwfyn/overview)** | Research output | Article, documentation |
| **[GitHub](https://github.com/CGutt-hub/EmotiView)** | Technical implementation | Analysis pipeline, results, presentations, proposal |

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

### Research Output (OSF)
*   **Article**: *(Coming soon)* The research article summarizing findings and contributions.

### Technical Implementation (GitHub)
*   **`EV_results/`**: Processed data, analysis metrics, and visualizations.
*   **`EV_analysis/`**: The Nextflow-based analysis pipeline with Python modules.
*   **`EV_presentation/`**: Slides and presentation materials.
*   **`EV_proposal/`**: The original research proposal with methodology and analysis plan.

## Analysis Pipeline

The analysis pipeline is built on the **[AnalysisToolbox](https://github.com/CGutt-hub/AnalysisToolbox)**—a modular Nextflow framework for scalable, reproducible data processing with automatic result synchronization. The EmotiView-specific pipeline in `EV_analysis/` extends this framework to:

*   Load and parse multi-modal raw data (EEG, fNIRS, ECG, EDA, questionnaires).
*   Perform standardized preprocessing steps specific to each physiological modality.
*   Extract key features and metrics (e.g., EEG power, FAI, RMSSD, fNIRS ROI activation, PLV).
*   Generate participant-level results and aggregated summaries.

Configuration is managed via `EV_analysis/EV_parameters.config`. See the [AnalysisToolbox documentation](https://github.com/CGutt-hub/AnalysisToolbox) for framework details.

## Project Status

Data collection and Thesis writing.

## Contributors

| Name | Role | Contact |
|------|------|---------|
| **Cagatay Özcan Jagiello Gutt** | Principal Investigator | [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1774--532X-green?logo=orcid)](https://orcid.org/0000-0002-1774-532X) |
| **Ben Gopin** | Technical Assistant | [![Email](https://img.shields.io/badge/Email-ben.gopin001-blue?logo=gmail)](mailto:ben.gopin001@stud.fh-dortmund.de) |
| **Gerrit Jostler** | Technical Assistant | [![Email](https://img.shields.io/badge/Email-gerrit.jostler001-blue?logo=gmail)](mailto:gerrit.jostler001@stud.fh-dortmund.de) |