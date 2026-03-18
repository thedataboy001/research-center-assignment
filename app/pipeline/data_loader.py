# pipeline/data_loader.py
# Responsible for loading raw data from CSV.
# Separated from validation and preprocessing so each concern
# can be tested, logged, and monitored independently.

import pandas as pd
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads the research centers dataset from a CSV file.

    Responsibilities:
        - Locate the data file
        - Load into a pandas DataFrame
        - Perform basic structural checks (columns present, not empty)
        - Return the raw DataFrame to the validation step

    Does NOT perform validation logic — that is DataValidator's job.
    """

    REQUIRED_COLUMNS = [
        "researchCenterId",
        "researchCenterName",
        "city",
        "latitude",
        "longitude",
        "internalFacilitiesCount",
        "hospitals_10km",
        "pharmacies_10km",
        "facilityDiversity_10km",
        "facilityDensity_10km",
    ]

    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path or settings.data_path)

    def load(self) -> pd.DataFrame:
        """
        Load the CSV file and perform structural validation.

        Returns:
            pd.DataFrame: Raw loaded dataset

        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If required columns are missing or file is empty
        """
        logger.info(f"Loading data from: {self.data_path}")

        # ── Check file exists ─────────────────────────────────────────────────
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(
                f"Data file not found at path: {self.data_path}. "
                "Please ensure research_centers.csv is in the project root."
            )

        # ── Load CSV ──────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise ValueError(f"Could not parse CSV file: {e}")

        # ── Empty file check ─────────────────────────────────────────────────
        if df.empty:
            raise ValueError(
                "Loaded dataset is empty. Cannot proceed with training."
            )

        # ── Required columns check ───────────────────────────────────────────
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info(
            f"Data loaded successfully "
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )
        return df


def test_data_loader():
    """Simple test to verify DataLoader functionality."""
    loader = DataLoader()
    try:
        df = loader.load()
        print("DataLoader test passed. Sample data:")
        print(df.head())
    except Exception as e:
        print(f"DataLoader test failed: {e}")

if __name__ == "__main__":
    test_data_loader()