import os
import yaml
from dataclasses import dataclass
from src.Logger import Logger

logger = Logger.get_logger(__file__)


def load_params(stage: str) -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[stage]


@dataclass
class RawDataConfig:
    """Raw data ingestion configuration."""
    _params = load_params("data_ingestion")
    kaggle_data_source: str = _params["kaggle_data_source"]
    ingestion_location: str = _params["ingestion_location"]


class DataIngestion:
    def __init__(self):
        os.makedirs(RawDataConfig.ingestion_location, exist_ok=True)

    @staticmethod
    def trigger_ingestion():
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                RawDataConfig.kaggle_data_source,
                output_dir=RawDataConfig.ingestion_location,
            )
            logger.info(
                f"Successfully ingested dataset at {RawDataConfig.ingestion_location}"
            )
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    DataIngestion.trigger_ingestion()
