# config.py
import json
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Dict

@dataclass
class Config:
    """
    Class for storing and managing application configuration.

    Attributes:
        LOGGING_LEVEL (int): Logging level (default: logging.INFO).
        DATA_PATH (str): Path to the training data.
        LABELS_PATH (str): Path to the labels file.
        MODEL_PATH (str): Path to save the model.
        PRED_PATH (str): Path to the prediction data.
        SEGMENT_LENGTH (int): Length of the data segment.
        BATCH_SIZE (int): Batch size.
        EPOCHS (int): Number of training epochs.
        LEARNING_RATE (float): Learning rate.
        TEST_SIZE (float): Size of the test set.
        LOW_PASS_F (float): Lower filtering threshold.
        HIGH_PASS_F (float): Upper filtering threshold.
        LEARNING_RATE_DECAY (float): Learning rate decay coefficient.
    """
    LOGGING_LEVEL: int = logging.INFO
    DATA_PATH: str = r"D:/EEGBASE/Penetratio/EDF/"
    LABELS_PATH: str = r"D:/EEGBASE/Penetratio/File.csv"
    MODEL_PATH: str = r"D:/EEGBASE/Penetratio/model.pth"
    PRED_PATH: str = r"D:/EEGBASE/Penetratio/TEST/"
    SEGMENT_LENGTH: int = 1000
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    TEST_SIZE: float = 0.2
    LOW_PASS_F: float = 0.5
    HIGH_PASS_F: float = 70.0
    LEARNING_RATE_DECAY: float = 0.8

    @classmethod
    def setup_logging(cls) -> None:
        """Set up logging."""
        logging.basicConfig(level=cls.LOGGING_LEVEL, format='%(levelname)s - %(message)s')

    @classmethod
    def save_to_file(cls, file_path: str = "config.json") -> None:
        """
        Saves the current configuration to a file.

        Args:
            file_path (str): Path to the file where the configuration will be saved.
        """
        config_dict = {field.name: getattr(cls, field.name) for field in fields(cls)}
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Settings saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str = "config.json") -> None:
        """
        Loads configuration from a file.

        Args:
            file_path (str): Path to the configuration file.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data: Dict[str, Any] = json.load(f)
            for key, value in data.items():
                if hasattr(cls, key):
                    field_type = cls.__annotations__.get(key)
                    if field_type:
                        try:
                            if field_type == int:
                                value = int(float(value))
                            elif field_type == float:
                                value = float(value)
                        except (ValueError, TypeError):
                            logging.warning(
                                f"Cannot convert {key}={value} to {field_type}. Using default value."
                            )
                            value = getattr(cls, key)  # Use the default value
                    setattr(cls, key, value)
            logging.info(f"Settings loaded from {file_path}")
        else:
            logging.info(f"File {file_path} not found. Using default settings.")

    @classmethod
    def log_config(cls) -> None:
        """Logs the current configuration."""
        logging.info("Current configuration:")
        for field in fields(cls):
            value = getattr(cls, field.name)
            logging.info(f"{field.name}: {value}")

# Set up logging and load configuration
Config.setup_logging()
Config.load_from_file()
Config.log_config()  # Log the current configuration
logging.info("Configuration successfully loaded.")