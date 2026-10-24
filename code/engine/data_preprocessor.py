"""Prepare transaction data and model-specific feature matrices.

Hedonic and Ridge models use dummy encoding; XGBoost uses label encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict, Any

# ============================================================================
# Constants
# ============================================================================

# 25 Seoul Districts (alphabetical order in Korean)
SEOUL_DISTRICTS = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
]

# Price-related columns to exclude from features
PRICE_VARIABLES_TO_EXCLUDE = [
    "거래금액",
    "price_numeric",
    "transaction_price",
    "deal_amount",
    "sale_price",
    "unit_price",
    "total_price",
    "contract_amount",
    "deposit",
    "monthly_rent",
    "price_per_sqm",
    "log_price_per_sqm",
]

# ============================================================================
# Temporal Feature Functions (for Forecasting)
# ============================================================================


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features for forecasting Hedonic models.

    Uses continuous time trend for out-of-sample extrapolation.
    Month seasonality is handled via month dummies in the model.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'year' and 'month' columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features:
        - time_trend: Linear time trend (2015-01 = 0, 2016-01 = 1, etc.)

    Notes
    -----
    - time_trend enables linear extrapolation to future periods
    - Month seasonality uses standard month dummies (11 dummies, ref: January)
    """
    result = df.copy()

    # Linear time trend: (year - 2015) + (month - 1) / 12
    # 2015-01 = 0, 2015-06 = 5/12, 2016-01 = 1, 2020-01 = 5, etc.
    result["time_trend"] = (df["year"] - 2015) + (df["month"] - 1) / 12

    return result


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction and non-linear features for flexible hedonic specifications.

    For H5 (Flexible Hedonic Model):
    - Squared terms: area², building_age²
    - Interaction terms: area × floor, area × district
    - Log transformations: log(area)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: area_numeric, floor, building_age, district

    Returns
    -------
    pd.DataFrame
        DataFrame with added interaction/non-linear features:
        - area_sq: area²
        - building_age_sq: building_age²
        - log_area: log(area_numeric)
        - area_x_floor: area × floor
        - area_x_district_*: area × district dummy (one per district)

    Notes
    -----
    - Use for H5 flexible hedonic specification
    - Allows the hedonic specification to be flexible in area and age
    - The log-area transformation imputes missing values using its median
    """
    result = df.copy()

    # Squared terms
    if "area_numeric" in df.columns:
        result["area_sq"] = df["area_numeric"] ** 2
        result["log_area"] = np.log(df["area_numeric"].replace(0, np.nan)).fillna(
            np.log(df["area_numeric"].replace(0, np.nan)).median()
        )

    if "building_age" in df.columns:
        result["building_age_sq"] = df["building_age"] ** 2

    # Simple interaction: area × floor
    if "area_numeric" in df.columns and "floor" in df.columns:
        result["area_x_floor"] = df["area_numeric"] * df["floor"]

    # area × district interactions (create dummy × area for each district)
    if "area_numeric" in df.columns and "district" in df.columns:
        for district in SEOUL_DISTRICTS:
            district_dummy = (df["district"] == district).astype(float)
            result[f"area_x_district_{district}"] = df["area_numeric"] * district_dummy

    return result


def create_alternative_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create alternative time trend specifications for robustness checks.

    Alternative trend variables:
    - Quadratic trend: time_trend²
    - Piecewise trend: pre/post COVID-19 (2020-03 break)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'year' and 'month' columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added time trend variants:
        - time_trend_sq: (time_trend)²
        - time_trend_pre_covid: time_trend before 2020-03, 0 after
        - time_trend_post_covid: 0 before 2020-03, time_trend after
        - post_covid_dummy: 1 if >= 2020-03, else 0

    Notes
    -----
    - Use for robustness checks on time specification
    - Break date: March 2020
    """
    result = df.copy()

    # Base time_trend (if not already created)
    if "time_trend" not in result.columns:
        result["time_trend"] = (df["year"] - 2015) + (df["month"] - 1) / 12

    # Quadratic trend
    result["time_trend_sq"] = result["time_trend"] ** 2

    # Piecewise trend with COVID-19 break (2020-03)
    covid_break_ym = 202003

    if "contract_ym" in df.columns:
        post_covid = df["contract_ym"] >= covid_break_ym
    else:
        post_covid = (df["year"] >= 2020) & (df["month"] >= 3)

    result["post_covid_dummy"] = post_covid.astype(float)
    result["time_trend_pre_covid"] = result["time_trend"] * (~post_covid).astype(float)
    result["time_trend_post_covid"] = result["time_trend"] * post_covid.astype(float)

    return result


# ============================================================================
# Utility Functions
# ============================================================================


def extract_district(df: pd.DataFrame, source_col: str = "시군구") -> pd.Series:
    """
    Extract district (구) from 시군구 column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    source_col : str
        Source column name (default: '시군구')

    Returns
    -------
    pd.Series
        District names (25 unique values for Seoul)

    Examples
    --------
    >>> "서울특별시 강남구 역삼동" -> "강남구"
    >>> "서울특별시 송파구 잠실동" -> "송파구"
    """
    pattern = r"(\S+구)"
    extracted = df[source_col].str.extract(pattern, expand=False)
    return extracted


def generate_repeat_sales_flag(
    df: pd.DataFrame, id_columns: List[str] = None, min_transactions: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate repeat sales flag and property ID.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_columns : List[str]
        Columns to create property ID (default: 시군구, 단지명, 번지, floor, area_numeric)
    min_transactions : int
        Minimum transactions to be considered repeat sale

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (repeat_sales_flag, property_id)
    """
    if id_columns is None:
        id_columns = ["시군구", "단지명", "번지", "floor", "area_numeric"]

    # Create composite property ID
    property_id = df[id_columns].astype(str).agg("_".join, axis=1)

    # Count transactions per property
    transaction_counts = property_id.map(property_id.value_counts())

    # Flag properties with min_transactions or more
    repeat_sales_flag = transaction_counts >= min_transactions

    return repeat_sales_flag, property_id


# ============================================================================
# Dummy Encoding Pipeline (for Hedonic and Ridge)
# ============================================================================


class DummyEncodingPipeline:
    """
    Creates dummy variables for Hedonic regression and Ridge models.

    Uses dummy encoding with reference categories:
    - District: 24 dummies (reference: 중구)
    - Year: observed levels excluding the reference year (default: 2015)
    - Month: 11 dummies (reference: January)

    Attributes
    ----------
    reference_district : str
        Reference district (dropped from dummies)
    reference_year : int
        Reference year (dropped from dummies)
    reference_month : int
        Reference month (dropped from dummies)
    district_order : List[str]
        Fixed order of districts for reproducibility
    fitted : bool
        Whether the pipeline has been fitted
    """

    def __init__(
        self, reference_district: str = "중구", reference_year: int = 2015, reference_month: int = 1
    ):
        self.reference_district = reference_district
        self.reference_year = reference_year
        self.reference_month = reference_month
        self.district_order = SEOUL_DISTRICTS.copy()
        self.fitted = False

        # Store years/months found during fit
        self.years_found = []
        self.months_found = list(range(1, 13))

    def fit(self, df: pd.DataFrame) -> "DummyEncodingPipeline":
        """
        Fit the pipeline to learn available years.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'year' column

        Returns
        -------
        self
        """
        if "year" in df.columns:
            self.years_found = sorted(df["year"].dropna().unique().astype(int).tolist())
        self.fitted = True
        return self

    def create_district_dummies(
        self, df: pd.DataFrame, district_col: str = "district"
    ) -> pd.DataFrame:
        """
        Create 24 district dummy variables (drop reference 중구).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with district column
        district_col : str
            Name of district column

        Returns
        -------
        pd.DataFrame
            DataFrame with 24 district dummy columns
        """
        # Create dummies for all districts
        dummies = pd.get_dummies(df[district_col], prefix="district", drop_first=False)

        # Drop reference category
        ref_col = f"district_{self.reference_district}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])

        # Ensure consistent column order (excluding reference)
        expected_cols = [
            f"district_{d}" for d in self.district_order if d != self.reference_district
        ]

        # Reindex to ensure all expected columns exist
        for col in expected_cols:
            if col not in dummies.columns:
                dummies[col] = 0

        return dummies[expected_cols]

    def create_apt_complex_dummies(
        self,
        df: pd.DataFrame,
        complex_col: str = "apt_complex",
        top_n: int = 200,
        reference: str = "_Other",
    ) -> pd.DataFrame:
        """
        Create apt complex dummy variables for H3c.
        Uses top-N most frequent complexes; rest grouped as _Other (reference).
        """
        import pandas as pd

        col = complex_col if complex_col in df.columns else "단지명"
        if col not in df.columns:
            return pd.DataFrame(index=df.index)

        series = df[col].fillna("_Unknown")

        # Get top-N complexes from training data
        if not hasattr(self, "_apt_complex_categories"):
            top = series.value_counts().head(top_n).index.tolist()
            self._apt_complex_categories = top

        mapped = series.where(series.isin(self._apt_complex_categories), "_Other")
        dummies = pd.get_dummies(mapped, prefix="aptcx")

        # Drop reference
        ref_col = "aptcx__Other"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])

        # Ensure consistent columns (from training categories)
        expected_cols = [f"aptcx_{c}" for c in self._apt_complex_categories]
        for c in expected_cols:
            if c not in dummies.columns:
                dummies[c] = 0
        # Keep only expected columns in order
        dummies = dummies[[c for c in expected_cols if c in dummies.columns]]

        return dummies

    def create_year_dummies(self, df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
        """
        Create year dummy variables (drop reference 2015).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with year column
        year_col : str
            Name of year column

        Returns
        -------
        pd.DataFrame
            DataFrame with year dummy columns (excluding reference)
        """
        # Create dummies
        dummies = pd.get_dummies(df[year_col].astype(int), prefix="year", drop_first=False)

        # Ensure all years from fitted data are present
        expected_cols = [f"year_{y}" for y in self.years_found if y != self.reference_year]
        for col in expected_cols:
            if col not in dummies.columns:
                dummies[col] = 0

        # Drop reference year if present
        ref_col = f"year_{self.reference_year}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])

        # Keep only expected columns in consistent order
        return dummies[[c for c in expected_cols if c in dummies.columns]]

    def create_month_dummies(self, df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
        """
        Create month dummy variables (drop reference January).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with month column
        month_col : str
            Name of month column

        Returns
        -------
        pd.DataFrame
            DataFrame with 11 month dummy columns (excluding January)
        """
        # Create dummies
        dummies = pd.get_dummies(df[month_col].astype(int), prefix="month", drop_first=False)

        # Ensure all months are present (except reference)
        expected_cols = [f"month_{m}" for m in self.months_found if m != self.reference_month]
        for col in expected_cols:
            if col not in dummies.columns:
                dummies[col] = 0

        # Drop reference month if present
        ref_col = f"month_{self.reference_month}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])

        # Keep only expected columns in consistent order
        return dummies[[c for c in expected_cols if c in dummies.columns]]

    def transform(
        self,
        df: pd.DataFrame,
        include_district: bool = True,
        include_year: bool = False,
        include_month: bool = False,
        include_apt_complex: bool = False,
    ) -> pd.DataFrame:
        """
        Full transformation pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with district, year, month columns
        include_district : bool
            Include 24 district dummies
        include_year : bool
            Include year dummies
        include_month : bool
            Include month dummies

        Returns
        -------
        pd.DataFrame
            DataFrame with all requested dummy variables
        """
        result_dfs = []

        if include_district and "district" in df.columns:
            result_dfs.append(self.create_district_dummies(df))

        if include_year and "year" in df.columns:
            result_dfs.append(self.create_year_dummies(df))

        if include_month and "month" in df.columns:
            result_dfs.append(self.create_month_dummies(df))

        if include_apt_complex:
            result_dfs.append(self.create_apt_complex_dummies(df))

        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        return pd.DataFrame(index=df.index)

    def fit_transform(
        self,
        df: pd.DataFrame,
        include_district: bool = True,
        include_year: bool = False,
        include_month: bool = False,
        include_apt_complex: bool = False,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(
            df, include_district, include_year, include_month, include_apt_complex
        )


# ============================================================================
# Label Encoding Pipeline (for XGBoost)
# ============================================================================


class LabelEncodingPipeline:
    """Encode categorical features for XGBoost.

    Retain the most frequent categories, group remaining values as Other and
    apply one LabelEncoder per categorical column.
    """

    def __init__(self, top_n_categories: int = 100):
        self.top_n_categories = top_n_categories
        self.encoders: Dict[str, LabelEncoder] = {}
        self.top_categories: Dict[str, List] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "LabelEncodingPipeline":
        """
        Fit encoders on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data

        Returns
        -------
        self
        """
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            # Get top N categories
            self.top_categories[col] = (
                df[col].value_counts().head(self.top_n_categories).index.tolist()
            )

            # Create and fit encoder
            le = LabelEncoder()
            values = df[col].where(df[col].isin(self.top_categories[col]), "Other").astype(str)
            le.fit(values)
            self.encoders[col] = le

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Transformed data with label-encoded categorical columns
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        result = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col in self.encoders:
                values = df[col].where(df[col].isin(self.top_categories[col]), "Other").astype(str)
                result[col] = self.encoders[col].transform(values)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


# ============================================================================
# Dual Encoding Processor (Combined)
# ============================================================================


class DualEncodingProcessor:
    """Prepare shared transaction data for hedonic, Ridge and XGBoost models."""

    def __init__(
        self,
        reference_district: str = "중구",
        reference_year: int = 2015,
        reference_month: int = 1,
        top_n_categories: int = 100,
    ):
        self.dummy_pipeline = DummyEncodingPipeline(
            reference_district=reference_district,
            reference_year=reference_year,
            reference_month=reference_month,
        )
        self.label_pipeline = LabelEncodingPipeline(top_n_categories=top_n_categories)
        self.base_data = None
        self.fitted = False

    def load_and_prepare_data(self, filepath: str, transaction_type: str = "매매") -> pd.DataFrame:
        """
        Load and prepare base data with required columns.

        Parameters
        ----------
        filepath : str
            Path to raw CSV file
        transaction_type : str
            Transaction type to filter (default: '매매' for sales)

        Returns
        -------
        pd.DataFrame
            Cleaned and prepared dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Raw data: {len(df):,} rows, {len(df.columns)} columns")

        # Filter by transaction type
        if "transaction_type" in df.columns and transaction_type:
            df = df[df["transaction_type"] == transaction_type].copy()
            print(f"  After filtering '{transaction_type}': {len(df):,} rows")

        # Extract district (구) from 시군구.
        if "시군구" in df.columns:
            df["district"] = extract_district(df)
            print(f"  Extracted district (구): {df['district'].nunique()} unique districts")
            # Validate districts are 구 names
            valid_districts = df["district"].dropna().unique()
            invalid = [d for d in valid_districts if not d.endswith("구")]
            if invalid:
                print(f"  Warning: Found non-district values: {invalid[:5]}...")

        # Create date column from contract_ym (always needed for rolling windows)
        if "contract_ym" in df.columns:
            df["date"] = pd.to_datetime(
                df["contract_ym"].astype(str), format="%Y%m", errors="coerce"
            )
            # Extract year and month if not already present
            if "year" not in df.columns:
                df["year"] = df["date"].dt.year
            if "month" not in df.columns:
                df["month"] = df["date"].dt.month
            print(
                f"  Extracted temporal: {df['year'].nunique()} years, "
                f"{df['month'].nunique()} months"
            )

        # Create target variable
        if "log_price_per_sqm" not in df.columns and "price_per_sqm" in df.columns:
            df["log_price_per_sqm"] = np.log(df["price_per_sqm"])
            print("  Created log_price_per_sqm")

        # Handle floor column
        if "floor" in df.columns:
            df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

        # Calculate building age if needed
        if "building_age" not in df.columns and "build_year" in df.columns:
            df["building_age"] = df["year"] - df["build_year"]

        # Create temporal features for forecasting (time_trend)
        df = create_temporal_features(df)
        print("  Created temporal features: time_trend")

        # Create interaction features for H5
        df = create_interaction_features(df)
        print("  Created interaction features for H5")

        # Drop rows with missing critical values
        critical_cols = ["district", "year", "month", "log_price_per_sqm"]
        existing_critical = [c for c in critical_cols if c in df.columns]
        before_drop = len(df)
        df = df.dropna(subset=existing_critical)
        print(f"  Dropped {before_drop - len(df):,} rows with missing critical values")

        self.base_data = df
        print(f"  Final data: {len(df):,} rows")
        return df

    def get_feature_columns(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
        """
        Get feature columns excluding target and specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        exclude_columns : List[str]
            Additional columns to exclude

        Returns
        -------
        List[str]
            List of feature column names
        """
        exclude = set(PRICE_VARIABLES_TO_EXCLUDE)
        if exclude_columns:
            exclude.update(exclude_columns)

        # Also exclude utility columns
        exclude.update(["date", "property_id", "repeat_sales_flag"])

        feature_cols = [c for c in df.columns if c not in exclude]
        return feature_cols

    def fit(self, df: pd.DataFrame) -> "DualEncodingProcessor":
        """
        Fit both encoding pipelines.

        Parameters
        ----------
        df : pd.DataFrame
            Training data

        Returns
        -------
        self
        """
        self.dummy_pipeline.fit(df)
        self.label_pipeline.fit(df)
        self.fitted = True
        return self

    def get_hedonic_features(
        self, df: pd.DataFrame, specification: str = "H1"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get feature matrix for Hedonic model specification.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        specification : str
            'H1', 'H2', 'H3', or 'H4'

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (Feature matrix, list of feature names)
        """
        from model_definitions import HEDONIC_VARIABLE_SETS

        spec = HEDONIC_VARIABLE_SETS[specification]
        features_list = []

        # 1. Continuous variables (structural)
        continuous_cols = [c for c in spec["continuous"] if c in df.columns]
        continuous_df = df[continuous_cols].copy()
        features_list.extend(continuous_cols)

        # 2. Temporal variables (time_trend for H2+)
        if spec.get("temporal"):
            temporal_cols = [c for c in spec["temporal"] if c in df.columns]
            temporal_df = df[temporal_cols].copy()
            features_list.extend(temporal_cols)
        else:
            temporal_df = pd.DataFrame(index=df.index)

        # 3. Dummy variables (district and/or month)
        dummy_config = spec["dummies"]
        dummy_df = self.dummy_pipeline.transform(
            df,
            include_district=dummy_config.get("district", False),
            include_year=False,  # No year FE (using time_trend)
            include_month=dummy_config.get("month", False),  # Month FE for H2+
        )
        features_list.extend(dummy_df.columns.tolist())

        # 4. Binary variables
        if spec.get("binary"):
            binary_cols = [c for c in spec["binary"] if c in df.columns]
            binary_df = df[binary_cols].astype(int)
            features_list.extend(binary_cols)
        else:
            binary_df = pd.DataFrame(index=df.index)

        # 5. Count variables
        if spec.get("count"):
            count_cols = [c for c in spec["count"] if c in df.columns]
            count_df = df[count_cols].copy()
            features_list.extend(count_cols)
        else:
            count_df = pd.DataFrame(index=df.index)

        # Combine all features
        X = pd.concat([continuous_df, temporal_df, dummy_df, binary_df, count_df], axis=1)

        # Fill missing values
        X = X.fillna(X.median())

        return X, features_list

    def get_xgboost_features(
        self, df: pd.DataFrame, feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Get feature matrix for XGBoost model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        feature_columns : List[str]
            Specific feature columns to use

        Returns
        -------
        pd.DataFrame
            Label-encoded feature matrix
        """
        if feature_columns is None:
            feature_columns = self.get_feature_columns(df)

        X = df[feature_columns].copy()
        X = self.label_pipeline.transform(X)

        # Ensure all columns are numeric (for SHAP compatibility)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(X.median())

        # Convert to float for consistent dtype
        X = X.astype(float)

        return X

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Get target variable (log_price_per_sqm).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.Series
            Target variable
        """
        return df["log_price_per_sqm"].copy()


# ============================================================================
# Validation Functions
# ============================================================================


def validate_district_values(df: pd.DataFrame, district_col: str = "district") -> Dict[str, Any]:
    """
    Validate district column values against expected 25 Seoul districts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    district_col : str
        Name of district column

    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    expected = set(SEOUL_DISTRICTS)
    actual = set(df[district_col].dropna().unique())

    return {
        "n_unique": len(actual),
        "expected": 25,
        "match": actual == expected,
        "missing_count": df[district_col].isna().sum(),
        "unexpected_values": actual - expected,
        "missing_districts": expected - actual,
    }


def validate_encoding_shapes(X_dummy: pd.DataFrame, specification: str) -> Dict[str, Any]:
    """
    Validate feature matrix shape against expected Hedonic specification.

    Parameters
    ----------
    X_dummy : pd.DataFrame
        Feature matrix
    specification : str
        'H1', 'H2', 'H3', 'H4'

    Returns
    -------
    Dict[str, Any]
        Validation results with expected vs actual shapes
    """
    # H1: 3 structural + 24 district = 27
    # H2: 3 structural + 1 time_trend + 11 month + 24 district = 39
    # H3: H2 + 3 binary + 4 count = 46
    # H4: 3 + 1 + 11 + 3 + 4 = 22 per district (no district dummies)
    expected_shapes = {
        "H1": 27,  # 3 + 24
        "H2": 39,  # 3 + 1 + 11 + 24
        "H3": 46,  # 3 + 1 + 11 + 24 + 3 + 4
        "H4": 22,  # 3 + 1 + 11 + 3 + 4 (per district, no district FE)
    }

    expected = expected_shapes.get(specification, 0)
    actual = X_dummy.shape[1]

    return {
        "specification": specification,
        "expected_features": expected,
        "actual_features": actual,
        "match": actual == expected,
        "feature_names": X_dummy.columns.tolist(),
    }
