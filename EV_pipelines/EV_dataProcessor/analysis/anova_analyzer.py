import pandas as pd
import pingouin as pg
import numpy as np

class ANOVAAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ANOVAAnalyzer initialized.")

    def perform_rm_anova(self, data_df, dv, within, subject, effsize="np2", detailed=True):
        """
        Performs a Repeated Measures ANOVA.
        Args:
            data_df (pd.DataFrame): DataFrame in long format.
            dv (str): Dependent variable column name.
            within (str or list): Within-subject factor column name(s).
            subject (str): Subject identifier column name.
            effsize (str): Effect size to compute.
            detailed (bool): Whether to return detailed output.
        Returns:
            pd.DataFrame: ANOVA results table, or None if error.
        """
        self.logger.info(f"ANOVAAnalyzer - Performing RM ANOVA: DV='{dv}', Within='{within}', Subject='{subject}'.")
        try:
            # Ensure required columns exist
            required_cols = [dv, subject]
            if isinstance(within, str): required_cols.append(within)
            elif isinstance(within, list): required_cols.extend(within)
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                self.logger.error(f"ANOVAAnalyzer - Missing columns for RM ANOVA: {missing_cols}")
                return None
            
            # Drop rows with NaNs in relevant columns to avoid errors in pingouin
            data_df_cleaned = data_df.dropna(subset=required_cols)
            if data_df_cleaned.empty:
                self.logger.warning("ANOVAAnalyzer - DataFrame is empty after dropping NaNs for RM ANOVA.")
                return None

            aov_results = pg.rm_anova(data=data_df_cleaned, dv=dv, within=within, subject=subject, 
                                      detailed=detailed, effsize=effsize)
            self.logger.info("ANOVAAnalyzer - RM ANOVA completed.")
            return aov_results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer - Error performing RM ANOVA: {e}", exc_info=True)
            return None
    
    # You can add methods for other types of ANOVA (between-subjects, mixed)