# pipeline/data_validator.py
# Validates the loaded DataFrame for quality issues before training.
# Produces a structured ValidationReport that is logged to MLflow.

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.output import ValidationReport

logger = get_logger(__name__)


class DataValidator:
    """
    Validates the research centers DataFrame against known
    domain constraints established during EDA.

    Checks performed:
        1. Missing values per column
        2. Duplicate rows
        3. Numeric range violations (out-of-bounds values)
        4. Data type correctness
        5. Minimum row count for clustering (need > K * 2 rows)
        6. Feature completeness (all selected features present)
        7. Cross-column consistency (soft warnings)

    A validation failure raises ValueError and halts the pipeline.
    Warnings are collected and logged but do not halt training.
    """

    # Domain-defined valid ranges from EDA + config
    RANGE_CONSTRAINTS = {
        "internalFacilitiesCount": (
            settings.min_facilities, settings.max_facilities
        ),
        "hospitals_10km":          (
            settings.min_hospitals, settings.max_hospitals
        ),
        "pharmacies_10km":         (
            settings.min_pharmacies, settings.max_pharmacies
        ),
        "facilityDiversity_10km":  (
            settings.min_diversity, settings.max_diversity
        ),
        "facilityDensity_10km":    (
            settings.min_density, None    # no upper bound
        ),
    }

    EXPECTED_DTYPES = {
        "internalFacilitiesCount": "int",
        "hospitals_10km":          "int",
        "pharmacies_10km":         "int",
        "facilityDiversity_10km":  "float",
        "facilityDensity_10km":    "float",
    }

    def __init__(self, df: pd.DataFrame):
        self.df       = df.copy()
        self.errors   : List[str] = []
        self.warnings : List[str] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def validate(self) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Run all validation checks.

        Returns:
            Tuple of (cleaned DataFrame, ValidationReport)

        Raises:
            ValueError: If any hard validation errors are found
        """
        logger.info("Starting data validation...")

        missing_counts    = self._check_missing_values()
        duplicate_count   = self._check_duplicates()
        out_of_range      = self._check_ranges()
        self._check_dtypes()
        self._check_min_rows()
        self._check_feature_completeness()
        self._cross_column_checks()

        report = ValidationReport(
            isValid          = len(self.errors) == 0,
            rowCount         = len(self.df),
            columnCount      = len(self.df.columns),
            missingValues    = missing_counts,
            outOfRangeValues = out_of_range,
            duplicateRows    = duplicate_count,
            warnings         = self.warnings,
            errors           = self.errors,
        )

        if not report.isValid:
            logger.error(f"Validation FAILED with {len(self.errors)} errors")
            for err in self.errors:
                logger.error(f"  ✗ {err}")
            raise ValueError(
                f"Data validation failed:\n" +
                "\n".join(f"  - {e}" for e in self.errors)
            )

        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  ⚠ {w}")

        logger.info(
            f"Validation passed "
            f"{len(self.warnings)} warnings, 0 errors"
        )
        return self.df, report

    # ── Private validation methods ────────────────────────────────────────────

    def _check_missing_values(self) -> Dict[str, int]:
        """
        Check for NaN values in all feature columns.
        Missing values in clustering features are hard errors
        sklearn K-Means cannot handle NaN.
        """
        missing = self.df[settings.features_list].isnull().sum().to_dict()
        for col, count in missing.items():
            if count > 0:
                self.errors.append(
                    f"Column '{col}' has {count} missing value(s). "
                    "K-Means requires complete feature vectors."
                )
        return {k: int(v) for k, v in missing.items()}

    def _check_duplicates(self) -> int:
        """
        Check for duplicate rows.
        Duplicate centers inflate K-Means centroid estimation.
        We remove them and log a warning.
        """
        n_dupes = self.df.duplicated().sum()
        if n_dupes > 0:
            self.warnings.append(
                f"{n_dupes} duplicate row(s) found and removed."
            )
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed {n_dupes} duplicate rows")
        return int(n_dupes)

    def _check_ranges(self) -> Dict[str, int]:
        """
        Check each feature against its known valid range.
        Out-of-range values are hard errors — they likely indicate
        data entry mistakes or unit errors.
        """
        out_of_range = {}
        for col, (lo, hi) in self.RANGE_CONSTRAINTS.items():
            if col not in self.df.columns:
                continue
            n_below = int((self.df[col] < lo).sum()) if lo is not None else 0
            n_above = int((self.df[col] > hi).sum()) if hi is not None else 0
            n_violations = n_below + n_above
            out_of_range[col] = n_violations
            if n_violations > 0:
                self.errors.append(
                    f"Column '{col}' has {n_violations} out-of-range value(s). "
                    f"Expected range: [{lo}, {hi if hi is not None else '∞'}]"
                )
        return out_of_range

    def _check_dtypes(self) -> None:
        """
        Validate that numeric columns have compatible data types.
        Attempt auto-coercion for fixable type mismatches (e.g. float → int).
        """
        for col, expected_kind in self.EXPECTED_DTYPES.items():
            if col not in self.df.columns:
                continue
            actual_dtype = str(self.df[col].dtype)
            if expected_kind == "int" and "int" not in actual_dtype:
                try:
                    self.df[col] = self.df[col].astype(int)
                    self.warnings.append(
                        f"Column '{col}' coerced from {actual_dtype} to int."
                    )
                except Exception:
                    self.errors.append(
                        f"Column '{col}' expected integer type, "
                        f"got {actual_dtype} and coercion failed."
                    )
            elif expected_kind == "float" and "float" not in actual_dtype:
                try:
                    self.df[col] = self.df[col].astype(float)
                    self.warnings.append(
                        f"Column '{col}' coerced from {actual_dtype} to float."
                    )
                except Exception:
                    self.errors.append(
                        f"Column '{col}' expected float type, "
                        f"got {actual_dtype} and coercion failed."
                    )

    def _check_min_rows(self) -> None:
        """
        Ensure enough rows exist for K-Means to form K clusters.
        Minimum requirement: rows > K * 3 (at least 3 points per cluster).
        """
        min_required = settings.n_clusters * 3
        if len(self.df) < min_required:
            self.errors.append(
                f"Insufficient data: {len(self.df)} rows found, "
                f"minimum {min_required} required for "
                f"K={settings.n_clusters} clustering."
            )

    def _check_feature_completeness(self) -> None:
        """Verify all selected features exist in the DataFrame."""
        missing_features = [
            f for f in settings.features_list
            if f not in self.df.columns
        ]
        if missing_features:
            self.errors.append(
                f"Missing feature columns: {missing_features}. "
                "These are required for model training."
            )

    def _cross_column_checks(self) -> None:
        """
        Soft cross-column consistency checks (warnings only).
        Based on the strong correlations (0.80-0.90) observed in EDA:
        a center with very high internal facilities but zero external
        access is a potential data quality flag.
        """
        suspicious = self.df[
            (self.df["internalFacilitiesCount"] > 8) &
            (self.df["hospitals_10km"] == 0) &
            (self.df["pharmacies_10km"] == 0)
        ]
        if len(suspicious) > 0:
            self.warnings.append(
                f"{len(suspicious)} center(s) have high internal facilities "
                "(>8) but zero hospitals and zero pharmacies nearby. "
                "This is unusual given the strong EDA correlations — "
                "please verify these records."
            )


def run_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Convenience function to run validation and return results.
    """
    validator = DataValidator(df)
    return validator.validate()