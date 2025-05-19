import pandas as pd
import pingouin as pg
import numpy as np
import re # For parsing column names

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

    def prepare_plv_data_for_anova(self, df_agg, band, autonomic):
        """Prepares (melts and parses) PLV data from wide aggregated format to long format for ANOVA."""
        self.logger.info(f"ANOVAAnalyzer - Preparing PLV data for {band}-{autonomic} ANOVA.")
        plv_cols = [col for col in df_agg.columns if col.startswith(f'plv_avg_{band}_') and f'_{autonomic}_' in col]
        if not plv_cols:
            self.logger.warning(f"ANOVAAnalyzer - No PLV columns found for {band}-{autonomic}.")
            return pd.DataFrame() # Return empty DataFrame

        df_plv_long = df_agg[['participant_id'] + plv_cols].melt(
            id_vars='participant_id',
            value_vars=plv_cols,
            var_name='plv_metric_full',
            value_name='plv_value'
        )
        df_plv_long.dropna(subset=['plv_value'], inplace=True)
        if df_plv_long.empty:
            self.logger.warning(f"ANOVAAnalyzer - No valid data for {band}-{autonomic} after melt and NaN drop.")
            return pd.DataFrame()

        def parse_plv_metric(metric_str, band_in, autonomic_in):
            pattern = re.compile(f"plv_avg_{band_in}_(.*)_{autonomic_in}_(.*)")
            match = pattern.match(metric_str)
            if match:
                return match.group(1), match.group(2) # roi, condition
            return None, None

        parsed_cols = df_plv_long['plv_metric_full'].apply(lambda x: pd.Series(parse_plv_metric(x, band, autonomic)))
        parsed_cols.columns = ['roi', 'condition']
        df_plv_long = pd.concat([df_plv_long, parsed_cols], axis=1)
        df_plv_long.dropna(subset=['roi', 'condition'], inplace=True)
        return df_plv_long

    def prepare_fai_data_for_anova(self, df_agg):
        """Prepares (melts and parses) FAI data from wide aggregated format to long format for ANOVA."""
        self.logger.info("ANOVAAnalyzer - Preparing FAI data for ANOVA.")
        fai_cols = [col for col in df_agg.columns if col.startswith('fai_alpha_')]
        if not fai_cols:
            self.logger.warning("ANOVAAnalyzer - No FAI columns found.")
            return pd.DataFrame()

        df_fai_long = df_agg[['participant_id'] + fai_cols].melt(
            id_vars='participant_id',
            value_vars=fai_cols,
            var_name='fai_metric_full',
            value_name='fai_value'
        )
        df_fai_long.dropna(subset=['fai_value'], inplace=True)
        if df_fai_long.empty:
            self.logger.warning("ANOVAAnalyzer - No valid FAI data after melt and NaN drop.")
            return pd.DataFrame()

        def parse_fai_metric(metric_str): # fai_alpha_F4_F3_Positive
            match = re.match(r"fai_alpha_(F[p|AF]?\d_F[p|AF]?\d)_(.*)", metric_str)
            if match: return match.group(1), match.group(2) # pair, condition
            return None, None
        
        parsed_fai_cols = df_fai_long['fai_metric_full'].apply(lambda x: pd.Series(parse_fai_metric(x)))
        parsed_fai_cols.columns = ['hemisphere_pair', 'condition']
        df_fai_long = pd.concat([df_fai_long, parsed_fai_cols], axis=1)
        df_fai_long.dropna(subset=['hemisphere_pair', 'condition'], inplace=True)
        return df_fai_long

    # You can add methods for other types of ANOVA (between-subjects, mixed)