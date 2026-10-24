"""
Evaluation Metrics Module for Rolling-window analysis

Implements academic-standard evaluation metrics:
- OOS R² (Out-of-Sample R-squared with test mean)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Diebold-Mariano Test with HAC (Newey-West) standard errors
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy import stats
import warnings


# ============================================================================
# Core Metrics
# ============================================================================

class EvaluationMetrics:
    """
    Academic-standard evaluation metrics for forecasting.

    All metrics designed for out-of-sample evaluation
    following econometric best practices.
    """

    @staticmethod
    def calculate_oos_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_mean: float = None
    ) -> float:
        """
        Calculate out-of-sample R² with explicit formula.

        R²_OOS = 1 - (SS_res / SS_tot)

        where SS_tot = Σ(y_true - ȳ_test)² uses the TEST set mean.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_mean : float
            Mean to use for SS_tot. If None, uses test mean (ȳ_test).

        Returns
        -------
        float
            Out-of-sample R² (can be negative for poor models)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Use test mean by default
        if y_mean is None:
            y_mean = np.mean(y_true)

        # Sum of squared residuals
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Total sum of squares (using test mean)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        # Avoid division by zero
        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Root Mean Squared Error.

        RMSE = sqrt(mean((y - ŷ)²))

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            RMSE
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_mae(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Error.

        MAE = mean(|y - ŷ|)

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            MAE
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def calculate_mape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        MAPE = mean(|y - ŷ| / |y|) × 100

        Handles zero values by excluding them from calculation.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        epsilon : float
            Threshold for considering values as zero

        Returns
        -------
        float
            MAPE (percentage)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Mask zero/near-zero actual values
        mask = np.abs(y_true) > epsilon

        if not np.any(mask):
            warnings.warn("All actual values near zero, MAPE undefined")
            return np.nan

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_mean: float = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_mean : float, optional
            Mean for R² calculation. If None, uses test mean (ȳ_test).

        Returns
        -------
        Dict[str, float]
            Dictionary with all metrics
        """
        return {
            'oos_r2': self.calculate_oos_r2(y_true, y_pred, y_mean),
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mae': self.calculate_mae(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred)
        }


# ============================================================================
# Diebold-Mariano Test with HAC Standard Errors
# ============================================================================

def diebold_mariano_test_hac(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    max_lag: int = None,
    loss: str = 'squared'
) -> Tuple[float, float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy with HAC correction.

    Implements the DM (1995) test with Newey-West HAC standard errors
    to account for serial correlation in forecast errors.

    H0: E[d_t] = 0 (equal predictive accuracy)
    H1: E[d_t] != 0

    Parameters
    ----------
    errors_1 : np.ndarray
        Forecast errors from model 1 (window-level MSE)
    errors_2 : np.ndarray
        Forecast errors from model 2 (window-level MSE)
    h : int
        Forecast horizon (for HAC lag selection)
    max_lag : int
        Maximum lag for HAC variance. Default: h-1 per DM (1995)
    loss : str
        Loss function: 'squared' or 'absolute'

    Returns
    -------
    Tuple[float, float, float]
        (DM statistic, HAC standard error, two-tailed p-value)

    References
    ----------
    Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive Accuracy"
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    errors_1 = np.asarray(errors_1)
    errors_2 = np.asarray(errors_2)

    if len(errors_1) != len(errors_2):
        raise ValueError("Error arrays must have same length")

    n = len(errors_1)

    if n < 10:
        warnings.warn(f"Small sample size ({n}), DM test may be unreliable")

    # Calculate loss differential
    if loss == 'squared':
        # For MSE comparison: d_t = e1_t^2 - e2_t^2
        # When errors are already MSE, just take difference
        d = errors_1 - errors_2
    elif loss == 'absolute':
        d = np.abs(errors_1) - np.abs(errors_2)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    # Mean loss differential
    d_bar = np.mean(d)

    # HAC variance estimation (Newey-West)
    if max_lag is None:
        max_lag = max(h - 1, 0)  # DM (1995) recommendation

    # Autocovariance at lag 0
    gamma_0 = np.var(d, ddof=0)

    # Compute weighted sum of autocovariances
    gamma_sum = 0
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        # Autocovariance at lag k
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        # Bartlett kernel weight
        weight = 1 - k / (max_lag + 1)
        gamma_sum += 2 * weight * gamma_k

    # HAC variance
    var_hac = gamma_0 + gamma_sum

    # Handle negative variance (can happen in small samples)
    if var_hac <= 0:
        warnings.warn("Non-positive HAC variance, using standard variance")
        var_hac = gamma_0

    # Standard error
    hac_se = np.sqrt(var_hac / n)

    # DM statistic
    if hac_se == 0:
        dm_stat = 0 if d_bar == 0 else np.sign(d_bar) * np.inf
    else:
        dm_stat = d_bar / hac_se

    # Two-tailed p-value (asymptotic normal)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, hac_se, p_value


def run_all_dm_tests(
    window_results,
    horizons: List[int] = None,
    models: List[str] = None
) -> pd.DataFrame:
    """
    Run DM tests for all model pairs at each horizon.

    Parameters
    ----------
    window_results : WindowResults
        Results container with forecast errors
    horizons : List[int]
        Horizons to test (default: [1, 3, 6, 12])
    models : List[str]
        Models to compare (default: all 6)

    Returns
    -------
    pd.DataFrame
        DM test results with columns:
        Horizon, Model_1, Model_2, DM_Statistic, HAC_SE, P_Value, Significance
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    if models is None:
        models = ['H1', 'H2', 'H3', 'H4', 'Ridge', 'XGBoost']

    results = []

    for h in horizons:
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:  # Upper triangle only (symmetric)
                    try:
                        mse1, mse2, common_windows = \
                            window_results.get_forecast_errors_for_dm_test(m1, m2, h)

                        if len(common_windows) < 5:
                            continue

                        dm_stat, hac_se, p_val = diebold_mariano_test_hac(
                            mse1, mse2, h=h
                        )

                        results.append({
                            'Horizon': h,
                            'Model_1': m1,
                            'Model_2': m2,
                            'DM_Statistic': round(dm_stat, 4),
                            'HAC_SE': round(hac_se, 6),
                            'P_Value': round(p_val, 4),
                            'Significance': _get_significance_stars(p_val),
                            'N_Windows': len(common_windows),
                            'Better_Model': m1 if dm_stat < 0 else m2 if dm_stat > 0 else 'Equal'
                        })

                    except Exception as e:
                        warnings.warn(f"DM test failed for {m1} vs {m2} h{h}: {e}")

    return pd.DataFrame(results)


def _get_significance_stars(p_value: float) -> str:
    """
    Get significance stars for p-value.

    Parameters
    ----------
    p_value : float
        P-value

    Returns
    -------
    str
        '***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.10, '' otherwise
    """
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.10:
        return '*'
    else:
        return ''


# ============================================================================
# Model Comparison Utilities
# ============================================================================

def create_dm_test_matrix(
    dm_results: pd.DataFrame,
    horizon: int,
    models: List[str] = None
) -> pd.DataFrame:
    """
    Create a matrix representation of DM test results.

    Parameters
    ----------
    dm_results : pd.DataFrame
        Results from run_all_dm_tests
    horizon : int
        Specific horizon to display
    models : List[str]
        Model order for matrix

    Returns
    -------
    pd.DataFrame
        Square matrix with DM statistics
    """
    if models is None:
        models = ['H1', 'H2', 'H3', 'H4', 'Ridge', 'XGBoost']

    # Filter by horizon
    df = dm_results[dm_results['Horizon'] == horizon]

    # Create matrix
    matrix = pd.DataFrame(
        index=models,
        columns=models,
        data=''
    )

    for _, row in df.iterrows():
        m1, m2 = row['Model_1'], row['Model_2']
        dm_stat = row['DM_Statistic']
        sig = row['Significance']

        # Upper triangle: Model_1 vs Model_2
        matrix.loc[m1, m2] = f"{dm_stat:.2f}{sig}"
        # Lower triangle: Model_2 vs Model_1 (negative stat)
        matrix.loc[m2, m1] = f"{-dm_stat:.2f}{sig}"

    # Diagonal
    for m in models:
        matrix.loc[m, m] = '-'

    return matrix


def summarize_model_rankings(
    dm_results: pd.DataFrame,
    models: List[str] = None
) -> pd.DataFrame:
    """
    Summarize model rankings based on DM test results.

    Parameters
    ----------
    dm_results : pd.DataFrame
        Results from run_all_dm_tests
    models : List[str]
        Model names

    Returns
    -------
    pd.DataFrame
        Model rankings with win/loss counts
    """
    if models is None:
        models = ['H1', 'H2', 'H3', 'H4', 'Ridge', 'XGBoost']

    # Initialize counters
    rankings = {m: {'wins': 0, 'losses': 0, 'ties': 0} for m in models}

    for _, row in dm_results.iterrows():
        m1, m2 = row['Model_1'], row['Model_2']
        dm_stat = row['DM_Statistic']
        p_val = row['P_Value']

        if p_val < 0.05:  # Significant difference
            if dm_stat < 0:
                rankings[m1]['wins'] += 1
                rankings[m2]['losses'] += 1
            else:
                rankings[m2]['wins'] += 1
                rankings[m1]['losses'] += 1
        else:
            rankings[m1]['ties'] += 1
            rankings[m2]['ties'] += 1

    # Convert to DataFrame
    df = pd.DataFrame(rankings).T
    df['total_comparisons'] = df['wins'] + df['losses'] + df['ties']
    df['win_rate'] = (df['wins'] / df['total_comparisons'] * 100).round(1)
    df = df.sort_values('wins', ascending=False)

    return df
