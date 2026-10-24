"""Construct rolling estimation windows and collect multi-horizon forecasts."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Generator
import warnings

# Default prediction horizons
PREDICTION_HORIZONS = [1, 3, 6, 12]

# ============================================================================
# Rolling Window Engine
# ============================================================================


class RollingWindowEngine:
    """
    Multi-horizon rolling window evaluation engine.

    Implements the 3-year training strategy:
    1. Initial training: 30 months
    2. Validation: 6 months
    3. Final training: 36 months
    4. Prediction: Multiple horizons [1, 3, 6, 12] months

    Attributes
    ----------
    initial_training_months : int
        Initial training period (default: 30)
    validation_months : int
        Validation period (default: 6)
    final_training_months : int
        Final training period (default: 36)
    prediction_horizons : List[int]
        Prediction horizons in months (default: [1, 3, 6, 12])
    step_months : int
        Window step size (default: 1)
    """

    def __init__(
        self,
        initial_training_months: int = 30,
        validation_months: int = 6,
        final_training_months: int = 36,
        prediction_horizons: List[int] = None,
        step_months: int = 1,
    ):
        self.initial_training_months = initial_training_months
        self.validation_months = validation_months
        self.final_training_months = final_training_months
        self.prediction_horizons = prediction_horizons or PREDICTION_HORIZONS
        self.step_months = step_months

    def create_rolling_windows(self, dates: pd.Series) -> List[Dict[str, Any]]:
        """
        Create rolling windows with multi-horizon support.

        Parameters
        ----------
        dates : pd.Series
            Transaction dates (datetime or Period)

        Returns
        -------
        List[Dict]
            List of window definitions with:
            - window_id
            - train periods
            - validation periods
            - prediction periods for each horizon
        """
        # Convert to Period if datetime
        if pd.api.types.is_datetime64_any_dtype(dates):
            periods = dates.dt.to_period("M")
        else:
            periods = dates

        # Get unique months sorted (convert to PeriodIndex for sorting)
        unique_months = pd.PeriodIndex(periods.unique()).sort_values()
        n_months = len(unique_months)

        # Calculate minimum required months
        min_required = self.final_training_months + max(self.prediction_horizons)

        if n_months < min_required:
            raise ValueError(
                f"Insufficient data: {n_months} months available, "
                f"need at least {min_required} for longest horizon"
            )

        windows = []
        window_id = 0

        # Start from month that allows full training + shortest horizon prediction
        # Each horizon will have different valid window counts
        start_idx = self.final_training_months

        while start_idx + min(self.prediction_horizons) <= n_months:
            # Calculate periods for this window
            train_end_idx = start_idx - 1
            train_start_idx = train_end_idx - self.final_training_months + 1

            if train_start_idx < 0:
                start_idx += self.step_months
                continue

            # Window definition
            window = {
                "window_id": window_id,
                # Initial training (2.5 years)
                "initial_train_start": unique_months[train_start_idx],
                "initial_train_end": unique_months[
                    train_start_idx + self.initial_training_months - 1
                ],
                # Validation (0.5 years)
                "validation_start": unique_months[train_start_idx + self.initial_training_months],
                "validation_end": unique_months[train_end_idx],
                # Final training (3 years)
                "final_train_start": unique_months[train_start_idx],
                "final_train_end": unique_months[train_end_idx],
                # Prediction horizons
                "prediction_horizons": {},
            }

            # Add prediction periods for each horizon
            for h in self.prediction_horizons:
                pred_idx = start_idx + h - 1
                if pred_idx < n_months:
                    window["prediction_horizons"][h] = {
                        "pred_month": unique_months[pred_idx],
                        "valid": True,
                    }
                else:
                    window["prediction_horizons"][h] = {"pred_month": None, "valid": False}

            windows.append(window)
            window_id += 1
            start_idx += self.step_months

        print(f"Created {len(windows)} rolling windows")
        print(f"  Date range: {unique_months[0]} to {unique_months[-1]}")
        print(f"  Horizons: {self.prediction_horizons}")

        # Count valid windows per horizon
        for h in self.prediction_horizons:
            valid_count = sum(1 for w in windows if w["prediction_horizons"][h]["valid"])
            print(f"  Horizon {h}m: {valid_count} valid windows")

        return windows

    def get_window_data(
        self, window: Dict, df: pd.DataFrame, dates: pd.Series, phase: str = "final_train"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get data for a specific phase of a window.

        Parameters
        ----------
        window : Dict
            Window definition
        df : pd.DataFrame
            Full dataframe
        dates : pd.Series
            Transaction dates
        phase : str
            'initial_train', 'validation', 'final_train', or 'prediction_h{horizon}'

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (Filtered dataframe, dates)
        """
        # Convert dates to Period if needed
        if pd.api.types.is_datetime64_any_dtype(dates):
            periods = dates.dt.to_period("M")
        else:
            periods = dates

        if phase == "initial_train":
            mask = (periods >= window["initial_train_start"]) & (
                periods <= window["initial_train_end"]
            )
        elif phase == "validation":
            mask = (periods >= window["validation_start"]) & (periods <= window["validation_end"])
        elif phase == "final_train":
            mask = (periods >= window["final_train_start"]) & (periods <= window["final_train_end"])
        elif phase.startswith("prediction_h"):
            horizon = int(phase.replace("prediction_h", ""))
            if not window["prediction_horizons"][horizon]["valid"]:
                return df.iloc[:0], dates.iloc[:0]
            pred_month = window["prediction_horizons"][horizon]["pred_month"]
            mask = periods == pred_month
        else:
            raise ValueError(f"Unknown phase: {phase}")

        return df[mask], dates[mask]

    def iterate_windows(
        self, windows: List[Dict], df: pd.DataFrame, dates: pd.Series
    ) -> Generator[Tuple[int, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generator that yields window data for training and prediction.

        Parameters
        ----------
        windows : List[Dict]
            List of window definitions
        df : pd.DataFrame
            Full dataframe
        dates : pd.Series
            Transaction dates

        Yields
        ------
        Tuple[int, Dict, pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]
            (window_id, window_info, train_df, val_df, {horizon: test_df})
        """
        for window in windows:
            window_id = window["window_id"]

            # Get training data
            train_df, train_dates = self.get_window_data(window, df, dates, "final_train")

            # Get validation data
            val_df, val_dates = self.get_window_data(window, df, dates, "validation")

            # Get test data for each horizon
            test_dfs = {}
            for h in self.prediction_horizons:
                if window["prediction_horizons"][h]["valid"]:
                    test_df, test_dates = self.get_window_data(
                        window, df, dates, f"prediction_h{h}"
                    )
                    if len(test_df) > 0:
                        test_dfs[h] = test_df

            yield window_id, window, train_df, val_df, test_dfs

    def get_valid_windows_per_horizon(self, windows: List[Dict]) -> Dict[int, int]:
        """
        Count valid windows for each prediction horizon.

        Parameters
        ----------
        windows : List[Dict]
            List of window definitions

        Returns
        -------
        Dict[int, int]
            {horizon: count of valid windows}
        """
        counts = {}
        for h in self.prediction_horizons:
            counts[h] = sum(1 for w in windows if w["prediction_horizons"][h]["valid"])
        return counts


# ============================================================================
# Window Result Storage
# ============================================================================


class WindowResults:
    """Store model performance metrics and predictions by window and horizon."""

    MODELS = ["H1", "H2", "H3", "H4", "Ridge", "XGBoost"]

    def __init__(self, horizons: List[int] = None):
        self.results = []
        self.horizons = horizons or PREDICTION_HORIZONS
        self.forecast_errors = {}  # For DM test
        self.predictions = {}  # Store (y_true, y_pred) for each key

    def add_result(
        self,
        window_id: int,
        window_info: Dict,
        model_name: str,
        horizon: int,
        metrics: Dict[str, float],
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
    ):
        """
        Add a single result entry.

        Parameters
        ----------
        window_id : int
            Window identifier
        window_info : Dict
            Window definition
        model_name : str
            Model name (H1, H2, H3, H4, Ridge, XGBoost)
        horizon : int
            Prediction horizon
        metrics : Dict[str, float]
            Dictionary of metrics (oos_r2, rmse, mae, mape)
        y_true : np.ndarray
            Actual values (for storing forecast errors)
        y_pred : np.ndarray
            Predicted values
        """
        result = {
            "window_id": window_id,
            "train_start": str(window_info["final_train_start"]),
            "train_end": str(window_info["final_train_end"]),
            "horizon": horizon,
            "test_month": str(window_info["prediction_horizons"][horizon]["pred_month"]),
            "model": model_name,
            "n_train": window_info.get("n_train", 0),
            "n_test": len(y_true) if y_true is not None else 0,
            "mean_y_true": float(np.mean(y_true)) if y_true is not None else None,
            "mean_y_pred": float(np.mean(y_pred)) if y_pred is not None else None,
            "std_y_true": float(np.std(y_true)) if y_true is not None else None,
            "std_y_pred": float(np.std(y_pred)) if y_pred is not None else None,
            **metrics,
        }
        self.results.append(result)

        # Store forecast errors and predictions
        if y_true is not None and y_pred is not None:
            key = (window_id, model_name, horizon)
            self.forecast_errors[key] = y_true - y_pred
            self.predictions[key] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "window_id": window_id,
                "model": model_name,
                "horizon": horizon,
                "test_month": str(window_info["prediction_horizons"][horizon]["pred_month"]),
            }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.results)

    def get_aggregated_results(self) -> pd.DataFrame:
        """
        Get mean/std metrics aggregated across windows.

        Returns
        -------
        pd.DataFrame
            Aggregated results with columns:
            Model, Horizon, N_Windows, Mean_R2, Std_R2, etc.
        """
        df = self.to_dataframe()

        if len(df) == 0:
            return pd.DataFrame()

        # Group by model and horizon
        agg = (
            df.groupby(["model", "horizon"])
            .agg(
                {
                    "oos_r2": ["mean", "std", "count"],
                    "rmse": ["mean", "std"],
                    "mae": ["mean", "std"],
                    "mape": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Flatten column names
        agg.columns = ["_".join(col).strip() for col in agg.columns.values]
        agg = agg.reset_index()

        # Rename columns
        agg = agg.rename(
            columns={
                "oos_r2_mean": "OOS_R2_Mean",
                "oos_r2_std": "OOS_R2_Std",
                "oos_r2_count": "N_Windows",
                "rmse_mean": "RMSE_Mean",
                "rmse_std": "RMSE_Std",
                "mae_mean": "MAE_Mean",
                "mae_std": "MAE_Std",
                "mape_mean": "MAPE_Mean",
                "mape_std": "MAPE_Std",
                "model": "Model",
                "horizon": "Horizon",
            }
        )

        return agg

    def get_forecast_errors_for_dm_test(
        self, model1: str, model2: str, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Get paired forecast errors for Diebold-Mariano test.

        Parameters
        ----------
        model1 : str
            First model name
        model2 : str
            Second model name
        horizon : int
            Prediction horizon

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[int]]
            (errors1, errors2, common_window_ids)
        """
        errors1 = {}
        errors2 = {}

        for (w_id, model, h), errors in self.forecast_errors.items():
            if h == horizon:
                if model == model1:
                    errors1[w_id] = errors
                elif model == model2:
                    errors2[w_id] = errors

        # Find common windows
        common_windows = sorted(set(errors1.keys()) & set(errors2.keys()))

        if len(common_windows) == 0:
            return np.array([]), np.array([]), []

        # Compute mean squared error per window
        mse1 = np.array([np.mean(errors1[w] ** 2) for w in common_windows])
        mse2 = np.array([np.mean(errors2[w] ** 2) for w in common_windows])

        return mse1, mse2, common_windows

    def save_all_predictions(self, output_dir: str) -> str:
        """
        Save all model predictions to CSV with shared actual values.

        Format: window_id, horizon, test_month, obs_idx, y_true, y_pred_H1, y_pred_H2, ...

        Parameters
        ----------
        output_dir : str
            Output directory path

        Returns
        -------
        str
            Path to saved file
        """
        import os

        # Organize by (window_id, horizon, obs_idx) with all model predictions
        pred_dict = {}

        for key, pred_info in self.predictions.items():
            window_id, model, horizon = key
            # Convert to numpy arrays to avoid index issues
            y_true = np.asarray(pred_info["y_true"])
            y_pred = np.asarray(pred_info["y_pred"])
            test_month = pred_info["test_month"]

            for i in range(len(y_true)):
                row_key = (window_id, horizon, test_month, i)
                if row_key not in pred_dict:
                    pred_dict[row_key] = {
                        "window_id": window_id,
                        "horizon": horizon,
                        "test_month": test_month,
                        "obs_idx": i,
                        "y_true": y_true[i],
                    }
                pred_dict[row_key][f"y_pred_{model}"] = y_pred[i]

        if pred_dict:
            records = list(pred_dict.values())
            df = pd.DataFrame(records)

            # Ensure column order
            base_cols = ["window_id", "horizon", "test_month", "obs_idx", "y_true"]
            pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
            df = df[base_cols + pred_cols]

            filepath = os.path.join(output_dir, "all_predictions.csv")
            df.to_csv(filepath, index=False)
            print(f"  Saved all predictions: {filepath} ({len(records):,} rows)")
            return filepath
        return None

    def load_model_predictions(self, model_names: List[str], filepath: str) -> int:
        """
        Load predictions for specified models from CSV.

        Parameters
        ----------
        model_names : List[str]
            List of model names to load (e.g., ['XGBoost'])
        filepath : str
            Path to predictions CSV

        Returns
        -------
        int
            Number of (window, model, horizon) combinations loaded
        """
        import os
        from evaluation_metrics import EvaluationMetrics

        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found")
            return 0

        df = pd.read_csv(filepath)
        metrics_calc = EvaluationMetrics()
        count = 0

        for model_name in model_names:
            pred_col = f"y_pred_{model_name}"
            if pred_col not in df.columns:
                print(f"  Warning: {pred_col} not found in {filepath}")
                continue

            # Group by window and horizon
            grouped = df.groupby(["window_id", "horizon", "test_month"])

            for (window_id, horizon, test_month), group in grouped:
                y_true = group["y_true"].values
                y_pred = group[pred_col].values

                # Skip if any NaN
                if np.isnan(y_pred).any():
                    continue

                # Calculate metrics using overall mean as approximation
                y_train_mean = y_true.mean()
                model_metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_train_mean)

                # Add to results
                result = {
                    "window_id": int(window_id),
                    "train_start": "loaded",
                    "train_end": "loaded",
                    "horizon": int(horizon),
                    "test_month": test_month,
                    "model": model_name,
                    "n_train": 0,
                    "n_test": len(y_true),
                    "mean_y_true": float(np.mean(y_true)),
                    "mean_y_pred": float(np.mean(y_pred)),
                    "std_y_true": float(np.std(y_true)),
                    "std_y_pred": float(np.std(y_pred)),
                    **model_metrics,
                }
                self.results.append(result)

                # Store forecast errors for DM test
                key = (int(window_id), model_name, int(horizon))
                self.forecast_errors[key] = y_true - y_pred
                self.predictions[key] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "window_id": int(window_id),
                    "model": model_name,
                    "horizon": int(horizon),
                    "test_month": test_month,
                }
                count += 1

            print(f"  Loaded {model_name}: {count} (window, horizon) combinations")

        return count

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """
        Get all predictions as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: window_id, model, horizon, test_month, obs_idx, y_true, y_pred
        """
        records = []
        for key, pred_info in self.predictions.items():
            window_id, model, horizon = key
            y_true = pred_info["y_true"]
            y_pred = pred_info["y_pred"]
            test_month = pred_info["test_month"]

            for i in range(len(y_true)):
                records.append(
                    {
                        "window_id": window_id,
                        "model": model,
                        "horizon": horizon,
                        "test_month": test_month,
                        "obs_idx": i,
                        "y_true": y_true[i],
                        "y_pred": y_pred[i],
                    }
                )

        return pd.DataFrame(records)


# ============================================================================
# Multi-Horizon Evaluation
# ============================================================================


def evaluate_window_multi_horizon(
    window_id: int,
    window_info: Dict,
    models: Dict[str, Any],
    test_data: Dict[int, Tuple[pd.DataFrame, pd.Series]],
    metrics_calculator,
) -> List[Dict]:
    """
    Evaluate all models across all horizons for a single window.

    Train once, predict multiple horizons.

    Parameters
    ----------
    window_id : int
        Window identifier
    window_info : Dict
        Window definition
    models : Dict[str, Any]
        Dictionary of fitted models
    test_data : Dict[int, Tuple[pd.DataFrame, pd.Series]]
        Test data for each horizon: {horizon: (X_test, y_test)}
    metrics_calculator : EvaluationMetrics
        Metrics calculator instance

    Returns
    -------
    List[Dict]
        Results for each model-horizon combination

    Notes
    -----
    R² is calculated using test mean (ȳ_test), not training mean.
    """
    results = []

    for horizon, (X_test, y_test) in test_data.items():
        if len(y_test) == 0:
            continue

        for model_name, model_wrapper in models.items():
            try:
                # Get predictions
                y_pred = model_wrapper.predict(X_test)

                # Calculate metrics (uses ȳ_test for R²)
                metrics = metrics_calculator.calculate_all_metrics(y_test.values, y_pred)

                results.append(
                    {
                        "window_id": window_id,
                        "model": model_name,
                        "horizon": horizon,
                        "metrics": metrics,
                        "y_true": y_test.values,
                        "y_pred": y_pred,
                    }
                )

            except Exception as e:
                warnings.warn(f"Prediction failed for {model_name} h{horizon}: {e}")

    return results
