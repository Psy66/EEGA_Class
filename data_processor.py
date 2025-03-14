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
    Class for processing EEG data.

    Methods:
        load_labels: Loads labels from a CSV file.
        describe_first_file: Returns information about the first file in the directory.
        process_edf: Processes EDF files and returns training data.
        load_and_process_data: Loads and processes data.
    """

    @staticmethod
    def load_labels(csv_path: str) -> pd.DataFrame:
        """
        Loads labels from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing labels.

        Returns:
            pd.DataFrame: DataFrame with labels.

        Raises:
            Exception: If an error occurs while loading the file.
        """
        try:
            return pd.read_csv(csv_path, delimiter=';')
        except Exception as e:
            logging.error(f"Error loading labels from {csv_path}: {e}")
            raise

    @staticmethod
    def describe_first_file(dir_path: str) -> str:
        """
        Returns information about the first EDF file in the directory.

        Args:
            dir_path (str): Path to the directory containing EDF files.

        Returns:
            str: Information about the first file.
        """
        for file in os.listdir(dir_path):
            if file.endswith('.EDF'):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        raw = mne.io.read_raw_edf(os.path.join(dir_path, file), preload=True, verbose=False)
                    info = [
                        f"Information about the first file in the directory: {file}",
                        f"Recording duration: {raw.times[-1]:.2f} seconds",
                        f"Sampling frequency: {raw.info['sfreq']} Hz",
                        f"Number of channels: {raw.info['nchan']}",
                    ]
                    if raw.annotations:
                        info.append(f"Events (labels): {raw.annotations.description}")
                    else:
                        info.append("No events (labels) found.")
                    return "\n".join(info)
                except Exception as e:
                    logging.error(f"Error analyzing file {file}: {str(e)}")
                    return f"Error analyzing file {file}: {str(e)}"
        return "No .EDF files found in the directory."

    @staticmethod
    def process_edf(dir_path: str, labels_df: pd.DataFrame, seg_len: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Processes EDF files and returns training data.

        Args:
            dir_path (str): Path to the directory containing EDF files.
            labels_df (pd.DataFrame): DataFrame containing labels.
            seg_len (int): Length of the data segment.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, int]:
                - x_data_array: Array of features.
                - y_data_array: Array of labels.
                - processed_files: Number of processed files.
                - skipped_files: Number of skipped files.
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
                    logging.error(f"Error processing file {file}: {str(e)}")

        logging.info(f"Processed files: {processed_files}, skipped files (no labels): {skipped_files}")
        x_data_array = np.array(x_data).reshape(-1, raw.info['nchan'], seg_len, 1)
        y_data_array = np.array(y_data)
        return x_data_array, y_data_array, processed_files, skipped_files

    @staticmethod
    def load_and_process_data(data_path: str, labels_path: str, segment_length: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, str]:
        """
        Loads and processes data.

        Args:
            data_path (str): Path to the data.
            labels_path (str): Path to the labels file.
            segment_length (int): Length of the data segment.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, int, int, int, str]:
                - features: Array of features.
                - labels: Array of labels.
                - n_cls: Number of classes.
                - n_chan: Number of channels.
                - processed_files: Number of processed files.
                - skipped_files: Number of skipped files.
                - file_info: Information about the first file.
        """
        try:
            labels = EEGProcessor.load_labels(labels_path)
            file_info = EEGProcessor.describe_first_file(data_path)  # Get information about the first file
            features, labels, processed_files, skipped_files = EEGProcessor.process_edf(data_path, labels, segment_length)
            unique_labels, labels = np.unique(labels, return_inverse=True)
            n_chan = features.shape[1]
            logging.info(f"Number of classes: {len(unique_labels)}, Classes: {', '.join(map(str, unique_labels))}, "
                         f"Training set size: {features.shape}, Number of channels: {n_chan}")
            return features, labels, len(unique_labels), n_chan, processed_files, skipped_files, file_info
        except Exception as e:
            logging.error(f"Error loading and processing data: {e}")
            raise