"""Hedonic, Ridge and XGBoost model specifications and estimation helpers."""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Tuple, List, Optional, Any
import warnings

# Try to import optional dependencies
try:
    import statsmodels.api as sm

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - Hedonic models will not work")

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not available - XGBoost model will not work")

# ============================================================================
# Hedonic Variable Set Specifications
# ============================================================================

HEDONIC_VARIABLE_SETS = {
    "H1": {
        "name": "Basic Hedonic Model",
        "description": "Core structural + district fixed effects (no temporal)",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": [],  # NO temporal for H1
        "dummies": {
            "district": True,  # 24 dummies (ref: 중구)
            "month": False,  # NO month FE for H1
        },
        "binary": [],
        "count": [],
        "expected_coefficients": 28,  # 3 + 24 + intercept
    },
    "H2": {
        "name": "Temporal Hedonic Model",
        "description": "H1 + time_trend + month fixed effects",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": ["time_trend"],  # Linear time trend only
        "dummies": {
            "district": True,  # 24 dummies
            "month": True,  # 11 month dummies (ref: January)
        },
        "binary": [],
        "count": [],
        "expected_coefficients": 40,  # 3 + 1 + 11 + 24 + intercept
    },
    "H3": {
        "name": "Extended Hedonic Model",
        "description": "H2 + amenity variables",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": ["time_trend"],  # Linear time trend
        "dummies": {
            "district": True,  # 24 dummies
            "month": True,  # 11 month dummies
        },
        "binary": ["has_parking", "has_elevator", "has_subway"],
        "count": ["total_parking", "elevator_count", "cctv_count", "edu_count"],
        "expected_coefficients": 47,  # 3 + 1 + 11 + 24 + 3 + 4 + intercept
    },
    "H3g": {
        "name": "District Fixed Effects Hedonic Model",
        "description": "H3 district fixed effects specification, labelled H3g in exhibits",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": ["time_trend"],
        "dummies": {
            "district": True,
            "month": True,
        },
        "binary": ["has_parking", "has_elevator", "has_subway"],
        "count": ["total_parking", "elevator_count", "cctv_count", "edu_count"],
        "expected_coefficients": 47,
    },
    "H3c": {
        "name": "Apartment Complex Fixed Effects Hedonic Model",
        "description": "H3 with apartment complex fixed effects replacing district effects",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": ["time_trend"],
        "dummies": {
            "district": False,
            "apt_complex": True,
            "month": True,
        },
        "binary": ["has_parking", "has_elevator", "has_subway"],
        "count": ["total_parking", "elevator_count", "cctv_count", "edu_count"],
        "expected_coefficients": -1,
    },
    "H4": {
        "name": "District-Specific Models",
        "description": "H3 estimated separately for each district (no district FE)",
        "continuous": ["area_numeric", "floor", "building_age"],
        "temporal": ["time_trend"],  # Linear time trend
        "dummies": {
            "district": False,  # NOT included (separate models by district)
            "month": True,  # 11 month dummies
        },
        "binary": ["has_parking", "has_elevator", "has_subway"],
        "count": ["total_parking", "elevator_count", "cctv_count", "edu_count"],
        "n_models": 25,
        "expected_coefficients_per_model": 23,  # 3 + 1 + 11 + 3 + 4 + intercept
    },
    "H5": {
        "name": "Flexible Hedonic Model",
        "description": "H3 + interactions + squared terms + log(area)",
        "continuous": [
            # Base structural
            "area_numeric",
            "floor",
            "building_age",
            # Non-linear transformations
            "log_area",
            "area_sq",
            "building_age_sq",
            # Simple interaction
            "area_x_floor",
        ],
        "temporal": ["time_trend"],  # Linear time trend
        "dummies": {
            "district": True,  # 24 district dummies
            "month": True,  # 11 month dummies
        },
        "binary": ["has_parking", "has_elevator", "has_subway"],
        "count": ["total_parking", "elevator_count", "cctv_count", "edu_count"],
        # Note: area × district interactions created separately (24, reference 중구 excluded)
        # Total: 3 base + 3 nonlinear + 1 interaction + 1 trend + 11 month + 24 district + 3 binary + 4 count + 24 area×district + intercept
        "expected_coefficients": 75,  # Approximate (3+3+1+1+11+24+3+4+24+1)
        "interactions": True,  # Flag to create area × district interactions
    },
}

# ============================================================================
# Model Factory
# ============================================================================


class ModelFactory:
    """Create and fit hedonic OLS, Ridge and XGBoost models."""

    def __init__(
        self, gpu_available: bool = False, gpu_device: str = "cpu", quick_tune: bool = False
    ):
        self.gpu_available = gpu_available
        self.gpu_device = gpu_device
        self.quick_tune = quick_tune

    # -------------------------------------------------------------------------
    # Hedonic OLS Models
    # -------------------------------------------------------------------------

    def create_hedonic_model(
        self, X: pd.DataFrame, y: pd.Series, specification: str = "H1"
    ) -> Optional[Any]:
        """
        Fit an OLS model for a hedonic feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already dummy-encoded)
        y : pd.Series
            Target variable
        specification : str
            Hedonic specification label

        Returns
        -------
        sm.RegressionResultsWrapper or None
            Fitted OLS model
        """
        if not STATSMODELS_AVAILABLE:
            warnings.warn("statsmodels not available")
            return None

        # Add constant for intercept
        X_const = sm.add_constant(X, has_constant="add")

        # Fit OLS
        try:
            model = sm.OLS(y, X_const).fit()
            return model
        except Exception as e:
            warnings.warn(f"Hedonic {specification} fitting failed: {e}")
            return None

    def create_h4_district_models(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        district_col: str = "district",
        min_observations: int = 100,
    ) -> Dict[str, Any]:
        """
        Create H4 specification: separate OLS models for each district.

        Memory-efficient approach: processes one district at a time.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe with district column
        X : pd.DataFrame
            Feature matrix (without district dummies)
        y : pd.Series
            Target variable
        district_col : str
            Name of district column
        min_observations : int
            Minimum observations required per district

        Returns
        -------
        Dict[str, Dict]
            Dictionary with results for each district
        """
        if not STATSMODELS_AVAILABLE:
            warnings.warn("statsmodels not available")
            return {}

        results = {}
        districts = df[district_col].unique()

        for district in districts:
            # Filter data for this district
            mask = df[district_col] == district
            X_district = X[mask].copy()
            y_district = y[mask].copy()

            # Skip if insufficient observations
            n_obs = len(X_district)
            if n_obs < min_observations:
                print(f"    Skipping {district}: only {n_obs} observations")
                continue

            # Add constant
            X_const = sm.add_constant(X_district, has_constant="add")

            # Fit OLS
            try:
                model = sm.OLS(y_district, X_const).fit()
                results[district] = {
                    "model": model,
                    "n_obs": n_obs,
                    "r2": model.rsquared,
                    "adj_r2": model.rsquared_adj,
                    "coefficients": model.params,
                    "std_errors": model.bse,
                    "pvalues": model.pvalues,
                }
            except Exception as e:
                print(f"    H4 model failed for {district}: {e}")

        return results

    # -------------------------------------------------------------------------
    # Ridge Regression
    # -------------------------------------------------------------------------

    def create_ridge_model(
        self, X: pd.DataFrame, y: pd.Series, alpha: float = None, tune: bool = True
    ) -> Tuple[Ridge, float]:
        """
        Create and fit Ridge regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (dummy-encoded)
        y : pd.Series
            Target variable
        alpha : float
            Regularization strength (if not tuning)
        tune : bool
            Whether to tune alpha via CV

        Returns
        -------
        Tuple[Ridge, float]
            (Fitted model, best alpha)
        """
        if tune:
            alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0]
            if self.quick_tune:
                alpha_range = [0.1, 1.0, 10.0]

            # Cross-validation to find best alpha
            from sklearn.linear_model import RidgeCV

            ridge_cv = RidgeCV(alphas=alpha_range, cv=3)
            ridge_cv.fit(X, y)
            best_alpha = ridge_cv.alpha_

            # Refit with best alpha
            model = Ridge(alpha=best_alpha)
            model.fit(X, y)
            return model, best_alpha
        else:
            alpha = alpha or 1.0
            model = Ridge(alpha=alpha)
            model.fit(X, y)
            return model, alpha

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------

    def get_xgboost_param_grid(self) -> Dict:
        """Get hyperparameter grid for XGBoost (efficient)."""
        if self.quick_tune:
            return {
                "n_estimators": [100, 200],
                "max_depth": [8, 10],
                "learning_rate": [0.1, 0.15],
                "subsample": [0.8, 1.0],
            }
        else:
            # Efficient grid: reduced combos, early stopping handles n_estimators
            return {
                "n_estimators": [300, 500],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9],
                "gamma": [0, 0.1],
            }

    def create_xgboost_model(
        self, X: pd.DataFrame, y: pd.Series, tune: bool = True, n_iter: int = 20
    ) -> Tuple[Any, Dict]:
        """
        Create and fit XGBoost model with optional GPU acceleration.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (label-encoded)
        y : pd.Series
            Target variable
        tune : bool
            Whether to tune hyperparameters
        n_iter : int
            Number of random search iterations

        Returns
        -------
        Tuple[xgb.XGBRegressor, Dict]
            (Fitted model, best params)
        """
        if not XGB_AVAILABLE:
            warnings.warn("xgboost not available")
            return None, {}

        # Base parameters
        base_params = {"random_state": 42, "tree_method": "hist", "n_jobs": -1}

        # Add GPU settings if available
        if self.gpu_available and self.gpu_device == "cuda":
            base_params["device"] = "cuda"

        if tune:
            # RandomizedSearchCV with early stopping via callbacks
            param_grid = self.get_xgboost_param_grid()
            # Early stopping: use high n_estimators, let early_stopping_rounds trim
            base_params["early_stopping_rounds"] = 30
            base_params["n_estimators"] = 500  # max, early stopping will cut
            xgb_model = xgb.XGBRegressor(**base_params)

            effective_n_iter = min(n_iter, 8) if not self.quick_tune else min(n_iter, 5)
            search = RandomizedSearchCV(
                xgb_model,
                param_grid,
                n_iter=effective_n_iter,
                cv=2,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=1,  # XGBoost handles parallelism internally
            )
            # Use eval_set for early stopping within CV
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
            search.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            best_model = search.best_estimator_
            best_params = search.best_params_
            best_params["effective_n_iter"] = effective_n_iter
            best_params["cv_folds"] = 2
            best_params["early_stopping_rounds"] = 30
            best_params["actual_best_iteration"] = getattr(best_model, "best_iteration", None)
            return best_model, best_params
        else:
            # Default parameters with early stopping
            default_params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "early_stopping_rounds": 30,
            }
            model = xgb.XGBRegressor(**base_params, **default_params)
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return model, default_params


# ============================================================================
# Hedonic Model Builder
# ============================================================================


class HedonicModelBuilder:
    """Build hedonic feature matrices using dummy encoding."""

    def __init__(self, dummy_pipeline=None):
        if dummy_pipeline is None:
            from data_preprocessor import DummyEncodingPipeline

            dummy_pipeline = DummyEncodingPipeline()
        self.dummy_pipeline = dummy_pipeline

    def build_feature_matrix(
        self, df: pd.DataFrame, specification: str = "H1"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build feature matrix for specified Hedonic model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with all columns (must have interaction features for H5)
        specification : str
            Model specification ('H1', 'H2', 'H3', 'H4', 'H5')

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (Feature matrix, list of feature names)
        """
        spec = HEDONIC_VARIABLE_SETS[specification]
        features = []

        # 1. Continuous (structural) variables
        continuous_cols = [c for c in spec["continuous"] if c in df.columns]
        continuous_df = df[continuous_cols].copy()
        features.extend(continuous_cols)

        # 2. Temporal variables (time_trend for H2+)
        if spec.get("temporal"):
            temporal_cols = [c for c in spec["temporal"] if c in df.columns]
            temporal_df = df[temporal_cols].copy()
            features.extend(temporal_cols)
        else:
            temporal_df = pd.DataFrame(index=df.index)

        # 3. Dummy variables (district and/or month)
        dummy_config = spec["dummies"]
        dummy_df = self.dummy_pipeline.transform(
            df,
            include_district=dummy_config.get("district", False),
            include_year=False,  # No year FE (using time_trend instead)
            include_month=dummy_config.get("month", False),  # Month FE for H2+
            include_apt_complex=dummy_config.get("apt_complex", False),
        )
        features.extend(dummy_df.columns.tolist())

        # 4. Binary variables
        if spec.get("binary"):
            binary_cols = [c for c in spec["binary"] if c in df.columns]
            binary_df = df[binary_cols].astype(float)
            features.extend(binary_cols)
        else:
            binary_df = pd.DataFrame(index=df.index)

        # 5. Count variables
        if spec.get("count"):
            count_cols = [c for c in spec["count"] if c in df.columns]
            count_df = df[count_cols].copy()
            features.extend(count_cols)
        else:
            count_df = pd.DataFrame(index=df.index)

        # 6. Interaction features (H5 only)
        if spec.get("interactions"):
            # Import SEOUL_DISTRICTS from data_preprocessor
            from data_preprocessor import SEOUL_DISTRICTS

            # area × district interactions
            # Drop the reference district's interaction. Including all 25 makes the design
            # exactly rank deficient, because each transaction lies in exactly one district and
            # so base area equals the sum of the 25 interactions. Excluding the same reference
            # category the district dummies use (중구) restores full rank. This is a
            # reparameterisation: the column space, and therefore every fitted value, is
            # unchanged. See DummyEncodingPipeline(reference_district='중구').
            REFERENCE_DISTRICT = "중구"
            interaction_cols = [
                f"area_x_district_{district}"
                for district in SEOUL_DISTRICTS
                if district != REFERENCE_DISTRICT
            ]
            available_interactions = [c for c in interaction_cols if c in df.columns]

            if available_interactions:
                interaction_df = df[available_interactions].copy()
                features.extend(available_interactions)
            else:
                interaction_df = pd.DataFrame(index=df.index)
        else:
            interaction_df = pd.DataFrame(index=df.index)

        # Combine all features
        X = pd.concat(
            [continuous_df, temporal_df, dummy_df, binary_df, count_df, interaction_df], axis=1
        )

        # Ensure all columns are numeric (convert object dtypes)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Fill missing values with column median
        X = X.fillna(X.median())

        # Ensure float dtype for statsmodels compatibility
        X = X.astype(float)

        return X, features

    def get_expected_coefficient_count(self, specification: str) -> int:
        """Get expected number of coefficients for a specification."""
        spec = HEDONIC_VARIABLE_SETS[specification]
        if "expected_coefficients" in spec:
            return spec["expected_coefficients"]
        return spec.get("expected_coefficients_per_model", 0)


# ============================================================================
# Model Wrapper for Predictions
# ============================================================================


class ModelWrapper:
    """
    Unified wrapper for making predictions across different model types.

    Handles:
    - Hedonic OLS (statsmodels)
    - Ridge (sklearn)
    - XGBoost

    Attributes
    ----------
    model : Any
        Fitted model object
    model_type : str
        Type of model ('hedonic', 'ridge', 'xgboost', 'h4')
    h4_models : Dict
        For H4, dictionary of district-specific models
    """

    def __init__(self, model: Any, model_type: str, h4_models: Dict = None):
        self.model = model
        self.model_type = model_type
        self.h4_models = h4_models

    def predict(self, X: pd.DataFrame, df: pd.DataFrame = None) -> np.ndarray:
        """
        Make predictions using the wrapped model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        df : pd.DataFrame
            Original dataframe (needed for H4 to get district)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.model_type == "hedonic":
            # Add constant for statsmodels
            X_const = sm.add_constant(X, has_constant="add")
            return self.model.predict(X_const)

        elif self.model_type == "h4":
            # District-specific predictions
            if df is None:
                raise ValueError("df required for H4 predictions")

            predictions = np.full(len(X), np.nan)

            for district, model_info in self.h4_models.items():
                mask = df["district"] == district
                if mask.sum() > 0:
                    X_district = X[mask]
                    X_const = sm.add_constant(X_district, has_constant="add")
                    try:
                        predictions[mask] = model_info["model"].predict(X_const)
                    except Exception:
                        pass

            return predictions

        elif self.model_type in ["ridge", "xgboost"]:
            return self.model.predict(X)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @property
    def r2(self) -> Optional[float]:
        """Get R² if available."""
        if self.model_type == "hedonic" and hasattr(self.model, "rsquared"):
            return self.model.rsquared
        return None
