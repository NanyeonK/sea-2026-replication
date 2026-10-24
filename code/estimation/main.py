#!/usr/bin/env python3
"""
R&R Revision Main Analysis Script

Multi-horizon rolling window analysis with 6 models:
- H1, H2, H3, H4: Hedonic OLS models
- Ridge: L2 regularized regression
- XGBoost: Gradient boosting with GPU acceleration

Usage:
    python main_rnr_analysis.py [--quick-test] [--windows N]

Arguments:
    --quick-test    Run with 2 windows for testing
    --windows N     Run with N windows (default: all)
    --no-shap       Disable SHAP analysis
    --no-dm         Disable Diebold-Mariano tests
"""

import sys
import os
import argparse
import warnings
from datetime import datetime
import gc

# Use the same August 6 engine as the exhibit scripts.
from pathlib import Path
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'engine'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import yaml

# Import modules
from data_preprocessor import (
    DualEncodingProcessor,
    extract_district,
    SEOUL_DISTRICTS
)
from model_definitions import (
    ModelFactory,
    HedonicModelBuilder,
    ModelWrapper,
    HEDONIC_VARIABLE_SETS
)
from rolling_window_engine import (
    RollingWindowEngine,
    WindowResults,
    PREDICTION_HORIZONS
)
from evaluation_metrics import (
    EvaluationMetrics,
    run_all_dm_tests
)
from output_generator import OutputGenerator

# Optional imports
try:
    from shap_analyzer import SHAPAnalyzer
    SHAP_ENABLED = True
except ImportError:
    SHAP_ENABLED = False
    print("Warning: SHAP module not available")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
except:
    GPU_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            'config',
            'analysis_config.yaml'
        )

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # Default configuration
    return {
        'data': {
            'input_path': '../../data/raw/seoul_apartment_transactions_with_building_info.csv',
            'output_dir': '../../output/rnr/'
        },
        'rolling_window': {
            'initial_training_months': 30,
            'validation_months': 6,
            'final_training_months': 36,
            'prediction_horizons': [1, 3, 6, 12],
            'step_months': 1
        },
        'shap': {
            'enabled': True,
            'sample_size': 1000,
            'window_interval': 6
        }
    }


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_analysis(
    config: dict,
    max_windows: int = None,
    enable_shap: bool = True,
    enable_dm_test: bool = True,
    enable_xgb: bool = True,
    quick_tune: bool = False,
    sample_type: str = "full",
    xgb_refit_months: int = 12
):
    """
    Run the full R&R analysis pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    max_windows : int
        Maximum windows to process (None = all)
    enable_shap : bool
        Enable SHAP analysis
    enable_dm_test : bool
        Enable Diebold-Mariano tests
    quick_tune : bool
        Use quick hyperparameter tuning
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*60}")
    print(f"R&R REVISION ANALYSIS")
    print(f"Started: {datetime.now()}")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Step 1: Load and Prepare Data
    # -------------------------------------------------------------------------
    print("Step 1: Loading and preparing data...")

    processor = DualEncodingProcessor()
    data_path = os.path.join(
        os.path.dirname(__file__),
        config['data']['input_path']
    )
    df = processor.load_and_prepare_data(data_path)

    # Apply subsample filter if requested
    if sample_type == "subsample":
        amenity_mask = df['total_parking'].notna() & df['elevator_count'].notna()
        n_before = len(df)
        df = df[amenity_mask].copy()
        df = df.reset_index(drop=True)
        print(f"  SUBSAMPLE filter: {n_before:,} -> {len(df):,} (amenity-enriched only)")
    print(f"  Sample type: {sample_type}")

    # Fit encoding pipelines
    processor.fit(df)

    # Get target and dates
    y = processor.get_target(df)
    dates = df['date']

    print(f"  Total observations: {len(df):,}")
    print(f"  Features available: {len(processor.get_feature_columns(df))}")
    print(f"  Target: log_price_per_sqm (mean={y.mean():.4f}, std={y.std():.4f})")

    # -------------------------------------------------------------------------
    # Step 2: Initialize Components
    # -------------------------------------------------------------------------
    print("\nStep 2: Initializing components...")

    # Rolling window engine
    rw_config = config['rolling_window']
    rw_engine = RollingWindowEngine(
        initial_training_months=rw_config['initial_training_months'],
        validation_months=rw_config['validation_months'],
        final_training_months=rw_config['final_training_months'],
        prediction_horizons=rw_config['prediction_horizons'],
        step_months=rw_config['step_months']
    )

    # Create windows
    windows = rw_engine.create_rolling_windows(dates)

    if max_windows is not None:
        windows = windows[:max_windows]
        print(f"  Limited to {max_windows} windows for testing")

    # Model factory
    model_factory = ModelFactory(
        gpu_available=GPU_AVAILABLE,
        gpu_device='cuda' if GPU_AVAILABLE else 'cpu',
        quick_tune=quick_tune
    )

    # Hedonic model builder
    hedonic_builder = HedonicModelBuilder(processor.dummy_pipeline)

    # Results storage
    results = WindowResults(horizons=rw_config['prediction_horizons'])

    # Metrics calculator
    metrics = EvaluationMetrics()

    # SHAP analyzer
    shap_analyzer = None
    if enable_shap and SHAP_ENABLED:
        shap_config = config.get('shap', {})
        shap_analyzer = SHAPAnalyzer(
            sample_size=shap_config.get('sample_size', 1000),
            window_interval=shap_config.get('window_interval', 6)
        )

    # Setup output directory early (needed for saving XGBoost models)
    # Keep the full and amenity samples in separate output folders.
    output_dir = os.path.join(
        os.path.dirname(__file__),
        config['data']['output_dir'],
        sample_type
    )
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 3: Rolling Window Evaluation
    # -------------------------------------------------------------------------
    print(f"\nStep 3: Running rolling window evaluation ({len(windows)} windows)...")

    _cached_xgb = None  # XGB acceleration: reuse model between refit intervals

    for window_id, window, train_df, val_df, test_dfs in rw_engine.iterate_windows(windows, df, dates):
        if len(train_df) == 0:
            continue

        print(f"\n  Window {window_id + 1}/{len(windows)}: "
              f"{window['final_train_start']} to {window['final_train_end']}")
        print(f"    Train size: {len(train_df):,}")

        # Update window info
        window['n_train'] = len(train_df)

        # Get training target
        y_train = y.loc[train_df.index]
        y_train_mean = y_train.mean()

        # -----------------------------------------------------------------
        # Train all 6 models
        # -----------------------------------------------------------------
        trained_models = {}

        # H1, H2, H3: Hedonic OLS
        # Determine hedonic specs: always H1-H3,H5; subsample adds H3g,H3c
        hedonic_specs = ['H1', 'H2', 'H3', 'H5']
        if sample_type == 'subsample':
            hedonic_specs = ['H1', 'H2', 'H3', 'H3g', 'H3c', 'H5']

        for spec in hedonic_specs:
            X_train_hed, _ = hedonic_builder.build_feature_matrix(train_df, spec)
            model = model_factory.create_hedonic_model(X_train_hed, y_train, spec)
            if model is not None:
                trained_models[spec] = {
                    'model': ModelWrapper(model, 'hedonic'),
                    'builder': hedonic_builder,
                    'spec': spec
                }
                print(f"    {spec}: R² = {model.rsquared:.4f}")

        # H4: District-specific
        X_train_h4, _ = hedonic_builder.build_feature_matrix(train_df, 'H4')
        h4_models = model_factory.create_h4_district_models(
            train_df, X_train_h4, y_train,
            min_observations=100
        )
        if h4_models:
            trained_models['H4'] = {
                'model': ModelWrapper(None, 'h4', h4_models),
                'builder': hedonic_builder,
                'spec': 'H4'
            }
            print(f"    H4: {len(h4_models)} district models")

        # Ridge
        X_train_ridge, _ = processor.get_hedonic_features(train_df, 'H3')
        ridge_model, ridge_alpha = model_factory.create_ridge_model(
            X_train_ridge, y_train, tune=True
        )
        trained_models['Ridge'] = {
            'model': ModelWrapper(ridge_model, 'ridge'),
            'X_type': 'hedonic',
            'spec': 'H3'
        }
        print(f"    Ridge: alpha = {ridge_alpha:.4f}")

        # XGBoost (optional) - refit every xgb_refit_months windows
        if enable_xgb and XGB_AVAILABLE:
            need_refit = (window_id % xgb_refit_months == 0) or (window_id == 0) or (_cached_xgb is None)
            if need_refit:
                feature_cols = processor.get_feature_columns(train_df)
                X_train_xgb = processor.get_xgboost_features(train_df, feature_cols)
                xgb_model, xgb_params = model_factory.create_xgboost_model(
                    X_train_xgb, y_train, tune=True, n_iter=10 if quick_tune else 20
                )
                _cached_xgb = {
                    'model': ModelWrapper(xgb_model, 'xgboost'),
                    'feature_cols': feature_cols,
                    'raw_model': xgb_model,
                    'params': xgb_params,
                    'refit_window': window_id
                }
                print(f"    XGBoost: REFIT at window {window_id}, depth={xgb_params.get('max_depth', 'N/A')}")

                # Save XGBoost model for later SHAP analysis
                model_save_dir = os.path.join(output_dir, f'{sample_type}_xgb_models')
                os.makedirs(model_save_dir, exist_ok=True)

                import joblib
                model_path = os.path.join(model_save_dir, f'xgb_window_{window_id}.pkl')
                joblib.dump({
                    'model': xgb_model,
                    'feature_cols': feature_cols,
                    'params': xgb_params,
                    'window_id': window_id,
                    'train_size': len(train_df)
                }, model_path)
                print(f"    Saved: {model_path}")
            else:
                print(f"    XGBoost: REUSED from window {_cached_xgb['refit_window']}")

            trained_models['XGBoost'] = {
                'model': _cached_xgb['model'],
                'feature_cols': _cached_xgb['feature_cols'],
                'raw_model': _cached_xgb['raw_model']
            }
        elif not enable_xgb:
            print(f"    XGBoost: DISABLED")

        # -----------------------------------------------------------------
        # Evaluate at each horizon
        # -----------------------------------------------------------------
        for horizon, test_df in test_dfs.items():
            if len(test_df) == 0:
                continue

            y_test = y.loc[test_df.index]
            print(f"    Horizon {horizon}m: {len(test_df)} test samples")

            # Evaluate each model
            for model_name, model_info in trained_models.items():
                try:
                    # Get appropriate X_test
                    if model_name in ['H1', 'H2', 'H3', 'H3g', 'H3c', 'H5']:
                        X_test, _ = hedonic_builder.build_feature_matrix(
                            test_df, model_info['spec']
                        )
                    elif model_name == 'H4':
                        X_test, _ = hedonic_builder.build_feature_matrix(test_df, 'H4')
                    elif model_name == 'Ridge':
                        X_test, _ = processor.get_hedonic_features(test_df, 'H3')
                    elif model_name == 'XGBoost':
                        X_test = processor.get_xgboost_features(
                            test_df, model_info['feature_cols']
                        )
                    else:
                        continue

                    # Get predictions
                    if model_name == 'H4':
                        y_pred = model_info['model'].predict(X_test, test_df)
                    else:
                        y_pred = model_info['model'].predict(X_test)

                    # Handle NaN predictions
                    valid_mask = ~np.isnan(y_pred)
                    if valid_mask.sum() == 0:
                        continue

                    y_test_valid = y_test.values[valid_mask]
                    y_pred_valid = y_pred[valid_mask]

                    # Calculate metrics (uses ȳ_test for R²)
                    model_metrics = metrics.calculate_all_metrics(
                        y_test_valid, y_pred_valid
                    )

                    # Store results
                    results.add_result(
                        window_id=window_id,
                        window_info=window,
                        model_name=model_name,
                        horizon=horizon,
                        metrics=model_metrics,
                        y_true=y_test_valid,
                        y_pred=y_pred_valid
                    )

                except Exception as e:
                    warnings.warn(f"Evaluation failed for {model_name} h{horizon}: {e}")

        # -----------------------------------------------------------------
        # SHAP Analysis (every N windows)
        # -----------------------------------------------------------------
        if shap_analyzer and shap_analyzer.should_calculate_shap(window_id):
            if 'XGBoost' in trained_models:
                print(f"    Computing SHAP values...")
                xgb_info = trained_models['XGBoost']
                X_train_xgb = processor.get_xgboost_features(train_df, xgb_info['feature_cols'])

                # Use first available test set for SHAP
                first_horizon = min(test_dfs.keys())
                test_df_shap = test_dfs[first_horizon]
                X_test_xgb = processor.get_xgboost_features(test_df_shap, xgb_info['feature_cols'])

                shap_analyzer.calculate_shap_values(
                    xgb_info['model'].model,
                    X_train_xgb,
                    X_test_xgb,
                    window_id
                )

        # Clear memory
        gc.collect()
        if GPU_AVAILABLE:
            import torch
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 3.5: Save/Load Predictions
    # -------------------------------------------------------------------------
    all_pred_path = os.path.join(output_dir, 'all_predictions.csv')

    # Always save all predictions (including XGBoost if enabled)
    if results.predictions:
        print("\nSaving all model predictions...")
        results.save_all_predictions(output_dir)

    # Load XGBoost predictions if we didn't run XGBoost but file exists
    if not enable_xgb:
        if os.path.exists(all_pred_path):
            print("\nLoading saved XGBoost predictions...")
            results.load_model_predictions(['XGBoost'], all_pred_path)
        else:
            print("\nWarning: XGBoost disabled and no saved predictions found")
            print(f"  Expected: {all_pred_path}")

    # -------------------------------------------------------------------------
    # Step 4: Diebold-Mariano Tests
    # -------------------------------------------------------------------------
    dm_results = None
    if enable_dm_test:
        print("\nStep 4: Running Diebold-Mariano tests...")
        dm_results = run_all_dm_tests(results)
        print(f"  Completed {len(dm_results)} pairwise tests")

    # -------------------------------------------------------------------------
    # Step 5: Generate Outputs
    # -------------------------------------------------------------------------
    print("\nStep 5: Generating output files...")

    output_gen = OutputGenerator(output_dir, timestamp, sample_type=sample_type)
    outputs = output_gen.generate_all_outputs(
        results,
        dm_results=dm_results,
        shap_analyzer=shap_analyzer,
        df=df
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Windows processed: {len(windows)}")
    if enable_xgb:
        if sample_type == 'subsample':
            print(f"  Models evaluated: 8 (H1, H2, H3, H3g, H3c, H4, H5, Ridge, XGBoost)")
        else:
            print(f"  Models evaluated: 7 (H1, H2, H3, H4, H5, Ridge, XGBoost)")
    else:
        if sample_type == 'subsample':
            print(f"  Models evaluated: 7 (H1, H2, H3, H3g, H3c, H4, H5, Ridge)")
        else:
            print(f"  Models evaluated: 6 (H1, H2, H3, H4, H5, Ridge)")
    print(f"  Horizons: {rw_config['prediction_horizons']}")
    print(f"  Output directory: {output_dir}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Completed: {datetime.now()}")

    return results, dm_results, outputs


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='R&R Revision Multi-Horizon Analysis'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Run with 2 windows for testing'
    )
    parser.add_argument(
        '--windows', type=int, default=None,
        help='Number of windows to process'
    )
    parser.add_argument(
        '--no-shap', action='store_true',
        help='Disable SHAP analysis'
    )
    parser.add_argument(
        '--no-dm', action='store_true',
        help='Disable Diebold-Mariano tests'
    )
    parser.add_argument(
        '--no-xgb', action='store_true',
        help='Disable XGBoost training (run Hedonic and Ridge only)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--xgb-refit-months', type=int, default=12,
        help='Refit XGBoost every N windows/months (default: 12 = annual)'
    )
    parser.add_argument(
        '--sample-type', type=str, default='full',
        choices=['full', 'subsample'],
        help='Sample type: full (all data) or subsample (amenity-enriched only)'
    )

    args = parser.parse_args()

    if not args.no_xgb and not XGB_AVAILABLE:
        parser.error("XGBoost is required; install requirements.txt before estimation.")

    # Load config
    config = load_config(args.config)

    # Determine number of windows
    max_windows = args.windows
    if args.quick_test:
        max_windows = 2
        print("Running in quick test mode (2 windows)")

    # Run analysis
    results, dm_results, outputs = run_analysis(
        config=config,
        max_windows=max_windows,
        enable_shap=not args.no_shap,
        enable_dm_test=not args.no_dm,
        enable_xgb=not args.no_xgb,
        quick_tune=args.quick_test,
        sample_type=args.sample_type,
        xgb_refit_months=args.xgb_refit_months
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
