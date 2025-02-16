# data_processor.py
import logging
import os
import warnings
from typing import Tuple
import mne
import numpy as np
import pandas as pd
from config import Config

class EEGProcessor:
    """
    Класс для обработки данных EEG.

    Методы:
        load_labels: Загружает метки из CSV-файла.
        describe_first_file: Возвращает информацию о первом файле в директории.
        process_edf: Обрабатывает EDF-файлы и возвращает данные для обучения.
        load_and_process_data: Загружает и обрабатывает данные.
    """

    @staticmethod
    def load_labels(csv_path: str) -> pd.DataFrame:
        """
        Загружает метки из CSV-файла.

        Аргументы:
            csv_path (str): Путь к CSV-файлу с метками.

        Возвращает:
            pd.DataFrame: DataFrame с метками.

        Исключения:
            Exception: Если произошла ошибка при загрузке файла.
        """
        try:
            return pd.read_csv(csv_path, delimiter=';')
        except Exception as e:
            logging.error(f"Ошибка при загрузке меток из {csv_path}: {e}")
            raise

    @staticmethod
    def describe_first_file(dir_path: str) -> str:
        """
        Возвращает информацию о первом EDF-файле в директории.

        Аргументы:
            dir_path (str): Путь к директории с EDF-файлами.

        Возвращает:
            str: Информация о первом файле.
        """
        for file in os.listdir(dir_path):
            if file.endswith('.EDF'):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        raw = mne.io.read_raw_edf(os.path.join(dir_path, file), preload=True, verbose=False)
                    info = [
                        f"Информация о первом файле в директории: {file}",
                        f"Длительность записи: {raw.times[-1]:.2f} секунд",
                        f"Частота дискретизации: {raw.info['sfreq']} Гц",
                        f"Количество каналов: {raw.info['nchan']}",
                    ]
                    if raw.annotations:
                        info.append(f"События (метки): {raw.annotations.description}")
                    else:
                        info.append("События (метки) отсутствуют.")
                    return "\n".join(info)
                except Exception as e:
                    logging.error(f"Ошибка при анализе файла {file}: {str(e)}")
                    return f"Ошибка при анализе файла {file}: {str(e)}"
        return "Файлы .EDF не найдены в директории."

    @staticmethod
    def process_edf(dir_path: str, labels_df: pd.DataFrame, seg_len: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Обрабатывает EDF-файлы и возвращает данные для обучения.

        Аргументы:
            dir_path (str): Путь к директории с EDF-файлами.
            labels_df (pd.DataFrame): DataFrame с метками.
            seg_len (int): Длина сегмента данных.

        Возвращает:
            Tuple[np.ndarray, np.ndarray, int, int]:
                - x_data_array: Массив признаков.
                - y_data_array: Массив меток.
                - processed_files: Количество обработанных файлов.
                - skipped_files: Количество пропущенных файлов.
        """
        x_data, y_data = [], []
        skipped_files = processed_files = 0

        for file in os.listdir(dir_path):
            if file.endswith('.EDF'):
                labels = labels_df.loc[labels_df['Filename'] == file, 'key'].dropna().values
                if labels.size == 0:
                    skipped_files += 1
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        raw = mne.io.read_raw_edf(os.path.join(dir_path, file), preload=True, verbose=False)
                    raw.filter(Config.LOW_PASS_F, Config.HIGH_PASS_F, verbose=False)
                    raw.crop(2 / raw.info['sfreq'], (len(raw.times) - 2) / raw.info['sfreq'])
                    for i in range(len(raw.times) // seg_len):
                        start, stop = i * seg_len, (i + 1) * seg_len
                        if stop > len(raw.times):
                            break
                        segment = raw.get_data(start=start, stop=stop)
                        segment_min, segment_max = np.min(segment), np.max(segment)
                        if segment_max - segment_min > 0:
                            segment = (segment - segment_min) / (segment_max - segment_min)
                        x_data.append(segment)
                        y_data.extend(labels)
                    processed_files += 1
                except Exception as e:
                    logging.error(f"Ошибка при обработке файла {file}: {str(e)}")

        logging.info(f"Обработано файлов: {processed_files}, пропущено файлов (без меток): {skipped_files}")
        x_data_array = np.array(x_data).reshape(-1, raw.info['nchan'], seg_len, 1)
        y_data_array = np.array(y_data)
        return x_data_array, y_data_array, processed_files, skipped_files

    @staticmethod
    def load_and_process_data(data_path: str, labels_path: str, segment_length: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, str]:
        """
        Загружает и обрабатывает данные.

        Аргументы:
            data_path (str): Путь к данным.
            labels_path (str): Путь к файлу с метками.
            segment_length (int): Длина сегмента данных.

        Возвращает:
            Tuple[np.ndarray, np.ndarray, int, int, int, int, str]:
                - features: Массив признаков.
                - labels: Массив меток.
                - n_cls: Количество классов.
                - n_chan: Количество каналов.
                - processed_files: Количество обработанных файлов.
                - skipped_files: Количество пропущенных файлов.
                - file_info: Информация о первом файле.
        """
        try:
            labels = EEGProcessor.load_labels(labels_path)
            file_info = EEGProcessor.describe_first_file(data_path)  # Получаем информацию о первом файле
            features, labels, processed_files, skipped_files = EEGProcessor.process_edf(data_path, labels, segment_length)
            unique_labels, labels = np.unique(labels, return_inverse=True)
            n_chan = features.shape[1]
            logging.info(f"Количество классов: {len(unique_labels)}, Классы: {', '.join(map(str, unique_labels))}, "
                         f"Размер тренировочного набора: {features.shape}, Количество каналов: {n_chan}")
            return features, labels, len(unique_labels), n_chan, processed_files, skipped_files, file_info
        except Exception as e:
            logging.error(f"Ошибка при загрузке и обработке данных: {e}")
            raise