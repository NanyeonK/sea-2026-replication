"""Calculate XGBoost SHAP values and summarize variation across windows and districts."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
import gc

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - interpretability analysis disabled")

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================================
# SHAP Analyzer
# ============================================================================

class SHAPAnalyzer:
    """
    SHAP feature importance analysis for XGBoost models.

    Implements:
    - SHAP value calculation with GPU-to-CPU conversion
    - Temporal stability analysis (importance variation across windows)
    - Regional analysis by district

    Attributes
    ----------
    sample_size : int
        Number of samples for SHAP calculation (default: 1000)
    window_interval : int
        Calculate SHAP every N windows (default: 6)
    window_results : List[Dict]
        Stored SHAP results per window
    regional_results : Dict
        Regional SHAP breakdown
    """

    def __init__(
        self,
        sample_size: int = 1000,
        window_interval: int = 6
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP package required for this module")

        self.sample_size = sample_size
        self.window_interval = window_interval
        self.window_results = []
        self.regional_results = {}
        self.feature_names = []

    def calculate_shap_values(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        window_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate SHAP values for a window.

        Parameters
        ----------
        model : xgb.XGBRegressor
            Fitted XGBoost model
        X_train : pd.DataFrame
            Training features (for background)
        X_test : pd.DataFrame
            Test features to explain
        window_id : int
            Window identifier

        Returns
        -------
        Dict or None
            SHAP results with mean importance and values
        """
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()

            # Sample background data for SHAP
            n_background = min(100, len(X_train))
            background = X_train.sample(n=n_background, random_state=42)

            # Sample test data
            n_test = min(self.sample_size, len(X_test))
            X_sample = X_test.sample(n=n_test, random_state=42)

            # Clear GPU memory before SHAP
            self._clear_gpu_memory()

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model, background)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Calculate mean absolute SHAP importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            result = {
                'window_id': window_id,
                'feature_names': self.feature_names,
                'mean_abs_shap': mean_abs_shap,
                'shap_values': shap_values,  # Full values for detailed analysis
                'n_samples': n_test,
                'base_value': explainer.expected_value
            }

            # Store for temporal analysis
            self.window_results.append(result)

            return result

        except Exception as e:
            warnings.warn(f"SHAP calculation failed for window {window_id}: {e}")
            return None

    def calculate_regional_shap(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        df_test: pd.DataFrame,
        window_id: int,
        district_col: str = 'district',
        top_n_districts: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate SHAP values broken down by district.

        Parameters
        ----------
        model : xgb.XGBRegressor
            Fitted XGBoost model
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        df_test : pd.DataFrame
            Original test dataframe with district column
        window_id : int
            Window identifier
        district_col : str
            District column name
        top_n_districts : int
            Number of top districts to analyze

        Returns
        -------
        Dict or None
            Regional SHAP results
        """
        try:
            # Get districts with sufficient samples
            district_counts = df_test[district_col].value_counts()
            top_districts = district_counts.head(top_n_districts).index.tolist()

            regional_shap = {}

            # Sample background
            n_background = min(50, len(X_train))
            background = X_train.sample(n=n_background, random_state=42)
            explainer = shap.TreeExplainer(model, background)

            for district in top_districts:
                mask = df_test[district_col] == district
                X_district = X_test[mask]

                if len(X_district) < 10:
                    continue

                # Sample from district
                n_sample = min(200, len(X_district))
                X_sample = X_district.sample(n=n_sample, random_state=42)

                # Calculate SHAP
                shap_values = explainer.shap_values(X_sample)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                regional_shap[district] = {
                    'mean_abs_shap': mean_abs_shap,
                    'n_samples': n_sample,
                    'top_features': self._get_top_features(mean_abs_shap, n=5)
                }

            result = {
                'window_id': window_id,
                'districts': regional_shap
            }

            # Store
            self.regional_results[window_id] = result

            return result

        except Exception as e:
            warnings.warn(f"Regional SHAP failed for window {window_id}: {e}")
            return None

    def calculate_temporal_stability(self) -> pd.DataFrame:
        """
        Calculate temporal stability of SHAP importance across windows.

        Returns
        -------
        pd.DataFrame
            Stability metrics per feature:
            - feature: Feature name
            - mean_importance: Mean SHAP importance
            - std_importance: Std of SHAP importance
            - cv: Coefficient of variation
            - rank_mean: Mean rank across windows
            - rank_std: Std of rank
        """
        if len(self.window_results) < 2:
            warnings.warn("Need at least 2 windows for temporal stability")
            return pd.DataFrame()

        n_features = len(self.feature_names)
        n_windows = len(self.window_results)

        # Collect importance values per window
        importance_matrix = np.zeros((n_windows, n_features))
        rank_matrix = np.zeros((n_windows, n_features))

        for i, result in enumerate(self.window_results):
            importance = result['mean_abs_shap']
            importance_matrix[i, :] = importance
            # Calculate ranks (1 = most important)
            rank_matrix[i, :] = n_features - np.argsort(np.argsort(importance))

        # Calculate stability metrics
        stability = []
        for j, feature in enumerate(self.feature_names):
            imp = importance_matrix[:, j]
            ranks = rank_matrix[:, j]

            stability.append({
                'feature': feature,
                'mean_importance': np.mean(imp),
                'std_importance': np.std(imp),
                'cv': np.std(imp) / np.mean(imp) if np.mean(imp) > 0 else np.nan,
                'rank_mean': np.mean(ranks),
                'rank_std': np.std(ranks),
                'min_rank': np.min(ranks),
                'max_rank': np.max(ranks)
            })

        df = pd.DataFrame(stability)
        df = df.sort_values('mean_importance', ascending=False)

        return df

    def get_aggregated_importance(self) -> pd.DataFrame:
        """
        Get aggregated SHAP importance across all analyzed windows.

        Returns
        -------
        pd.DataFrame
            Aggregated importance with feature rankings
        """
        if len(self.window_results) == 0:
            return pd.DataFrame()

        # Average importance across windows
        all_importance = np.array([r['mean_abs_shap'] for r in self.window_results])
        mean_importance = np.mean(all_importance, axis=0)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap_importance': mean_importance,
            'windows_analyzed': len(self.window_results)
        })

        df = df.sort_values('mean_shap_importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        return df

    def should_calculate_shap(self, window_id: int) -> bool:
        """Check if SHAP should be calculated for this window."""
        return window_id % self.window_interval == 0

    def _get_top_features(
        self,
        importance: np.ndarray,
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        indices = np.argsort(importance)[::-1][:n]
        return [(self.feature_names[i], importance[i]) for i in indices]

    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_results(self, output_dir: str, timestamp: str = None):
        """
        Save SHAP results to files.

        Parameters
        ----------
        output_dir : str
            Output directory path
        timestamp : str
            Timestamp for file naming
        """
        import os
        import json

        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save aggregated importance
        agg_df = self.get_aggregated_importance()
        if len(agg_df) > 0:
            agg_path = os.path.join(output_dir, f'shap_importance_{timestamp}.csv')
            agg_df.to_csv(agg_path, index=False)
            print(f"  Saved: {agg_path}")

        # Save temporal stability
        stability_df = self.calculate_temporal_stability()
        if len(stability_df) > 0:
            stab_path = os.path.join(output_dir, f'shap_temporal_stability_{timestamp}.csv')
            stability_df.to_csv(stab_path, index=False)
            print(f"  Saved: {stab_path}")

        # Save regional results as JSON
        if self.regional_results:
            regional_path = os.path.join(output_dir, f'shap_regional_{timestamp}.json')
            # Convert numpy arrays to lists for JSON serialization
            regional_json = {}
            for window_id, result in self.regional_results.items():
                regional_json[str(window_id)] = {
                    'districts': {
                        district: {
                            'n_samples': data['n_samples'],
                            'top_features': data['top_features']
                        }
                        for district, data in result['districts'].items()
                    }
                }
            with open(regional_path, 'w', encoding='utf-8') as f:
                json.dump(regional_json, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {regional_path}")
