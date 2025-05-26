import pandas as pd
import pingouin as pg
import numpy as np

class CorrelationAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("CorrelationAnalyzer initialized.")

    def calculate_correlation(self, series1, series2, method='pearson', name1='Series1', name2='Series2'):
        """
        Calculates correlation between two pandas Series.
        Returns:
            dict: Correlation results (r, p-value, etc.) or None if error.
        """
        if not isinstance(series1, (pd.Series, np.ndarray)) or not isinstance(series2, (pd.Series, np.ndarray)):
            self.logger.error("CorrelationAnalyzer - Input must be pandas Series or numpy arrays.")
            return None
        
        # Convert numpy arrays to pandas Series for pingouin compatibility and NaN handling
        if isinstance(series1, np.ndarray): series1 = pd.Series(series1, name=name1)
        if isinstance(series2, np.ndarray): series2 = pd.Series(series2, name=name2)

        # Drop NaN pairs for correlation
        combined = pd.concat([series1, series2], axis=1).dropna()
        if len(combined) < 3: # Need at least 3 pairs for meaningful correlation
            self.logger.warning(f"CorrelationAnalyzer - Insufficient valid data points after NaN removal for {name1} vs {name2} (n={len(combined)}). Skipping.")
            return {'r': np.nan, 'p-val': np.nan, 'CI95%': [np.nan, np.nan], 'n': len(combined), 'name1': name1, 'name2': name2, 'method': method}

        try:
            self.logger.info(f"CorrelationAnalyzer - Calculating {method} correlation between {name1} and {name2} (n={len(combined)}).")
            corr_result = pg.corr(combined.iloc[:, 0], combined.iloc[:, 1], method=method)
            
            result_dict = corr_result.iloc[0].to_dict() # Convert first row of DataFrame to dict
            self.logger.info(f"CorrelationAnalyzer - Result ({name1} vs {name2}): r={result_dict.get('r', np.nan):.3f}, p={result_dict.get('p-val', np.nan):.3f}")
            return result_dict
        except Exception as e:
            self.logger.error(f"CorrelationAnalyzer - Error calculating correlation between {name1} and {name2}: {e}", exc_info=True)
            return None