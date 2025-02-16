# config.py
import json
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Dict

@dataclass
class Config:
    """
    Класс для хранения и управления конфигурацией приложения.

    Атрибуты:
        LOGGING_LEVEL (int): Уровень логирования (по умолчанию: logging.INFO).
        DATA_PATH (str): Путь к данным для обучения.
        LABELS_PATH (str): Путь к файлу с метками.
        MODEL_PATH (str): Путь для сохранения модели.
        PRED_PATH (str): Путь к данным для предсказания.
        SEGMENT_LENGTH (int): Длина сегмента данных.
        BATCH_SIZE (int): Размер батча.
        EPOCHS (int): Количество эпох обучения.
        LEARNING_RATE (float): Скорость обучения.
        TEST_SIZE (float): Размер тестового набора.
        LOW_PASS_F (float): Нижний порог фильтрации.
        HIGH_PASS_F (float): Верхний порог фильтрации.
        LEARNING_RATE_DECAY (float): Коэффициент уменьшения скорости обучения.
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
        """Настройка логирования."""
        logging.basicConfig(level=cls.LOGGING_LEVEL, format='%(levelname)s - %(message)s')

    @classmethod
    def save_to_file(cls, file_path: str = "config.json") -> None:
        """
        Сохраняет текущую конфигурацию в файл.

        Аргументы:
            file_path (str): Путь к файлу для сохранения конфигурации.
        """
        config_dict = {field.name: getattr(cls, field.name) for field in fields(cls)}
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Настройки сохранены в {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str = "config.json") -> None:
        """
        Загружает конфигурацию из файла.

        Аргументы:
            file_path (str): Путь к файлу с конфигурацией.
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
                                f"Невозможно преобразовать {key}={value} в {field_type}. Используется значение по умолчанию."
                            )
                            value = getattr(cls, key)  # Используем значение по умолчанию
                    setattr(cls, key, value)
            logging.info(f"Настройки загружены из {file_path}")
        else:
            logging.info(f"Файл {file_path} не найден. Используются настройки по умолчанию.")

    @classmethod
    def log_config(cls) -> None:
        """Логирует текущую конфигурацию."""
        logging.info("Текущая конфигурация:")
        for field in fields(cls):
            value = getattr(cls, field.name)
            logging.info(f"{field.name}: {value}")

# Настройка логирования и загрузка конфигурации
Config.setup_logging()
Config.load_from_file()
Config.log_config()  # Логируем текущую конфигурацию
logging.info("Конфигурация успешно загружена.")