"""Write model performance summaries, window results and comparison tables."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os

# ============================================================================
# Output Generator
# ============================================================================

class OutputGenerator:
    """
    Write performance and comparison outputs.

    Attributes
    ----------
    output_dir : str
        Output directory path
    timestamp : str
        Timestamp for file naming
    """

    def __init__(self, output_dir: str, timestamp: str = None, sample_type: str = 'full'):
        self.output_dir = output_dir
        self.timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sample_type = sample_type

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

    def generate_table2_main(
        self,
        window_results,
        output_filename: str = None
    ) -> pd.DataFrame:
        """
        Generate Table 2: Model Performance by Horizon.

        Parameters
        ----------
        window_results : WindowResults
            Results container
        output_filename : str
            Output filename

        Returns
        -------
        pd.DataFrame
            Aggregated results table
        """
        if output_filename is None:
            output_filename = f'{self.sample_type}_table2_multi_horizon.csv'

        df = window_results.get_aggregated_results()

        if len(df) == 0:
            print("  Warning: No results to aggregate for Table 2")
            return df

        # Sort by Model then Horizon
        model_order = ['H1', 'H2', 'H3', 'H3g', 'H3c', 'H4', 'H5', 'Ridge', 'XGBoost']
        df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
        df = df.sort_values(['Model', 'Horizon'])

        # Save
        filepath = os.path.join(self.output_dir, output_filename)
        df.to_csv(filepath, index=False)
        print(f"  Saved Table 2: {filepath}")

        return df

    def generate_window_level_csv(
        self,
        window_results,
        output_filename: str = None
    ) -> pd.DataFrame:
        """
        Generate window-level detailed results.

        Parameters
        ----------
        window_results : WindowResults
            Results container
        output_filename : str
            Output filename

        Returns
        -------
        pd.DataFrame
            Full window-level results
        """
        if output_filename is None:
            output_filename = f'{self.sample_type}_window_level_results.csv'

        df = window_results.to_dataframe()

        if len(df) == 0:
            print("  Warning: No window-level results")
            return df

        # Sort
        df = df.sort_values(['window_id', 'model', 'horizon'])

        # Save
        filepath = os.path.join(self.output_dir, output_filename)
        df.to_csv(filepath, index=False)
        print(f"  Saved window-level: {filepath}")

        return df

    def generate_hedonic_appendix(
        self,
        window_results,
        output_filename: str = None
    ) -> pd.DataFrame:
        """
        Generate Hedonic specification comparison (Appendix table).

        Compares H1 vs H2 vs H3 vs H4 performance.

        Parameters
        ----------
        window_results : WindowResults
            Results container
        output_filename : str
            Output filename

        Returns
        -------
        pd.DataFrame
            Hedonic comparison table
        """
        if output_filename is None:
            output_filename = f'{self.sample_type}_hedonic_spec_comparison.csv'

        df = window_results.to_dataframe()

        if len(df) == 0:
            return df

        # Filter to Hedonic models only
        hedonic_models = ['H1', 'H2', 'H3', 'H4']
        df_hedonic = df[df['model'].isin(hedonic_models)]

        # Pivot for comparison
        pivot = df_hedonic.pivot_table(
            values=['oos_r2', 'rmse', 'mae', 'mape'],
            index=['horizon'],
            columns=['model'],
            aggfunc=['mean', 'std']
        ).round(4)

        # Flatten column names
        pivot.columns = ['_'.join(map(str, col)).strip() for col in pivot.columns.values]
        pivot = pivot.reset_index()

        # Save
        filepath = os.path.join(self.output_dir, output_filename)
        pivot.to_csv(filepath, index=False)
        print(f"  Saved Hedonic appendix: {filepath}")

        return pivot

    def generate_dm_test_csv(
        self,
        dm_results: pd.DataFrame,
        output_filename: str = 'dm_tests_hac.csv'
    ) -> pd.DataFrame:
        """
        Generate DM test results CSV.

        Parameters
        ----------
        dm_results : pd.DataFrame
            Results from run_all_dm_tests
        output_filename : str
            Output filename

        Returns
        -------
        pd.DataFrame
            DM test results
        """
        if len(dm_results) == 0:
            print("  Warning: No DM test results")
            return dm_results

        # Sort
        dm_results = dm_results.sort_values(['Horizon', 'Model_1', 'Model_2'])

        # Save
        filepath = os.path.join(self.output_dir, output_filename)
        dm_results.to_csv(filepath, index=False)
        print(f"  Saved DM tests: {filepath}")

        return dm_results

    def generate_dm_test_matrices(
        self,
        dm_results: pd.DataFrame,
        output_prefix: str = 'dm_matrix'
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate DM test matrices for each horizon (for LaTeX tables).

        Parameters
        ----------
        dm_results : pd.DataFrame
            Results from run_all_dm_tests
        output_prefix : str
            Output filename prefix

        Returns
        -------
        Dict[int, pd.DataFrame]
            {horizon: matrix_dataframe}
        """
        from evaluation_metrics import create_dm_test_matrix

        matrices = {}
        horizons = dm_results['Horizon'].unique()

        for h in horizons:
            matrix = create_dm_test_matrix(dm_results, h)
            matrices[h] = matrix

            # Save
            filepath = os.path.join(self.output_dir, f'{output_prefix}_h{h}.csv')
            matrix.to_csv(filepath)
            print(f"  Saved DM matrix h{h}: {filepath}")

        return matrices

    def generate_summary_statistics(
        self,
        df: pd.DataFrame,
        output_filename: str = None
    ) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        output_filename : str
            Output filename

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if output_filename is None:
            output_filename = f'{self.sample_type}_summary_statistics.csv'

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = df[numeric_cols].describe().T
        stats['missing'] = df[numeric_cols].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)

        # Save
        filepath = os.path.join(self.output_dir, output_filename)
        stats.to_csv(filepath)
        print(f"  Saved summary stats: {filepath}")

        return stats

    def generate_all_outputs(
        self,
        window_results,
        dm_results: pd.DataFrame = None,
        shap_analyzer=None,
        df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all output files at once.

        Parameters
        ----------
        window_results : WindowResults
            Main results container
        dm_results : pd.DataFrame
            DM test results (optional)
        shap_analyzer : SHAPAnalyzer
            SHAP analyzer (optional)
        df : pd.DataFrame
            Full dataset (optional)

        Returns
        -------
        Dict[str, pd.DataFrame]
            All generated dataframes
        """
        print(f"\nGenerating outputs to: {self.output_dir}")

        outputs = {}

        # Table 2
        outputs['table2'] = self.generate_table2_main(window_results)

        # Window-level
        outputs['window_level'] = self.generate_window_level_csv(window_results)

        # Hedonic appendix
        outputs['hedonic_appendix'] = self.generate_hedonic_appendix(window_results)

        # DM tests
        if dm_results is not None and len(dm_results) > 0:
            outputs['dm_tests'] = self.generate_dm_test_csv(dm_results)
            outputs['dm_matrices'] = self.generate_dm_test_matrices(dm_results)

        # SHAP
        if shap_analyzer is not None:
            shap_analyzer.save_results(self.output_dir, self.timestamp)

        # Summary statistics
        if df is not None:
            outputs['summary_stats'] = self.generate_summary_statistics(df)

        print(f"\nAll outputs generated with timestamp: {self.timestamp}")

        return outputs

# ============================================================================
# LaTeX Table Formatters
# ============================================================================

def format_table2_latex(df: pd.DataFrame, caption: str = None) -> str:
    """
    Format Table 2 as LaTeX tabular.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated results dataframe
    caption : str
        Table caption

    Returns
    -------
    str
        LaTeX formatted table
    """
    if len(df) == 0:
        return ""

    # Column headers
    headers = ['Model', 'Horizon', 'N', 'OOS R²', 'RMSE', 'MAE', 'MAPE']

    # Begin table
    latex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + (caption or 'Model Performance by Forecast Horizon') + '}',
        r'\label{tab:model_performance}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule'
    ]

    # Header row
    latex.append(' & '.join(headers) + r' \\')
    latex.append(r'\midrule')

    # Data rows
    current_model = None
    for _, row in df.iterrows():
        model = row['Model']
        horizon = row['Horizon']

        # Add horizontal line between models
        if current_model is not None and model != current_model:
            latex.append(r'\midrule')
        current_model = model

        # Format values
        values = [
            model,
            str(int(horizon)),
            str(int(row['N_Windows'])),
            f"{row['OOS_R2_Mean']:.3f} ({row['OOS_R2_Std']:.3f})",
            f"{row['RMSE_Mean']:.4f}",
            f"{row['MAE_Mean']:.4f}",
            f"{row['MAPE_Mean']:.2f}"
        ]
        latex.append(' & '.join(values) + r' \\')

    # End table
    latex.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Note: Standard deviations in parentheses. OOS R² = Out-of-sample R-squared.',
        r'\item RMSE = Root Mean Squared Error. MAE = Mean Absolute Error. MAPE = Mean Absolute Percentage Error.',
        r'\end{tablenotes}',
        r'\end{table}'
    ])

    return '\n'.join(latex)


def format_dm_matrix_latex(matrix: pd.DataFrame, horizon: int) -> str:
    """
    Format DM test matrix as LaTeX tabular.

    Parameters
    ----------
    matrix : pd.DataFrame
        DM test matrix
    horizon : int
        Prediction horizon

    Returns
    -------
    str
        LaTeX formatted table
    """
    models = matrix.columns.tolist()
    n_models = len(models)

    latex = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{Diebold-Mariano Test Results (Horizon = {horizon} month)}}',
        f'\\label{{tab:dm_h{horizon}}}',
        r'\begin{tabular}{l' + 'c' * n_models + '}',
        r'\toprule'
    ]

    # Header
    latex.append(' & ' + ' & '.join(models) + r' \\')
    latex.append(r'\midrule')

    # Data rows
    for model in models:
        row_values = [model]
        for col_model in models:
            row_values.append(str(matrix.loc[model, col_model]))
        latex.append(' & '.join(row_values) + r' \\')

    latex.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\small',
        r'\item Note: DM statistics with significance levels (* p<0.10, ** p<0.05, *** p<0.01).',
        r'\item Positive values indicate row model outperforms column model.',
        r'\end{tablenotes}',
        r'\end{table}'
    ])

    return '\n'.join(latex)
