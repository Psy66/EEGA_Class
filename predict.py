# predict.py
import logging
import os
import warnings
from collections import Counter
from typing import Dict, List
import mne
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm
from config import Config
from model import CNN

def setup_logging() -> None:
    """Настройка логирования."""
    logging.basicConfig(level=Config.LOGGING_LEVEL, format='%(levelname)s - %(message)s')

def load_model(model_path: str, n_cls: int, n_chan: int, device: torch.device) -> CNN:
    """
    Загружает модель из файла.

    Аргументы:
        model_path (str): Путь к файлу модели.
        n_cls (int): Количество классов.
        n_chan (int): Количество каналов.
        device (torch.device): Устройство для вычислений (CPU/GPU).

    Возвращает:
        CNN: Загруженная модель.
    """
    model = CNN(n_cls, n_chan).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_file(file_path: str, segment_length: int, model: CNN, device: torch.device, class_labels: List[str]) -> Dict[str, str]:
    """
    Обрабатывает один файл и возвращает результаты предсказания.

    Аргументы:
        file_path (str): Путь к файлу.
        segment_length (int): Длина сегмента данных.
        model (CNN): Модель для предсказания.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        class_labels (List[str]): Список меток классов.

    Возвращает:
        Dict[str, str]: Результаты предсказания.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(Config.LOW_PASS_F, Config.HIGH_PASS_F, verbose=False)
    raw.crop(2 / raw.info['sfreq'], (len(raw.times) - 2) / raw.info['sfreq'])

    num_segments = len(raw.times) // segment_length
    x_new_data = [raw.get_data(start=i * segment_length, stop=(i + 1) * segment_length)
                  for i in range(num_segments) if (i + 1) * segment_length <= len(raw.times)]

    x_new_data = [(segment - np.min(segment)) / (np.max(segment) - np.min(segment))
                  for segment in x_new_data if np.max(segment) - np.min(segment) > 0]

    x_new_tensor = torch.tensor(np.array(x_new_data).reshape(-1, raw.info['nchan'], segment_length, 1),
                                dtype=torch.float32, device=device).squeeze()

    with torch.no_grad():
        outputs = model(x_new_tensor.unsqueeze(1).to(device))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_classes = torch.max(outputs, 1)[1].cpu().numpy()

    class_counter = Counter(predicted_classes)
    most_common_class, most_common_count = class_counter.most_common(1)[0]
    mean_probabilities = np.mean(probabilities, axis=0)

    return {
        "file": os.path.basename(file_path),
        "class": class_labels[most_common_class],
        "confidence": most_common_count / sum(class_counter.values()),
        "segments": len(x_new_data),
        "segment_duration": segment_length / raw.info['sfreq'],
        "mean_probabilities": mean_probabilities.tolist(),
    }

def predict_new_data(model_path: str, pred_path: str, segment_length: int, n_cls: int, n_chan: int, class_labels: List[str]) -> List[List[str]]:
    """
    Предсказывает классы для новых данных.

    Аргументы:
        model_path (str): Путь к файлу модели.
        pred_path (str): Путь к данным для предсказания.
        segment_length (int): Длина сегмента данных.
        n_cls (int): Количество классов.
        n_chan (int): Количество каналов.
        class_labels (List[str]): Список меток классов.

    Возвращает:
        List[List[str]]: Результаты предсказания.
    """
    setup_logging()

    if not os.path.exists(pred_path):
        logging.error(f"Директория {pred_path} не найдена.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, n_cls, n_chan, device)

    results = []
    files = [file for file in os.listdir(pred_path) if file.endswith('.EDF')]

    if not files:
        logging.warning(f"Нет файлов .EDF в директории {pred_path}.")
        return results

    for file in tqdm(files, desc="Обработка файлов"):
        try:
            result = process_file(os.path.join(pred_path, file), segment_length, model, device, class_labels)
            results.append([
                result["file"],
                result["class"],
                f"{result['confidence']:.2%}",
                result["segments"],
                f"{result['segment_duration']:.2f} сек",
            ])
        except Exception as e:
            logging.error(f"Ошибка при обработке файла {file}: {str(e)}")
            results.append([file, "Ошибка", "N/A", "N/A", "N/A"])

    headers = ["Файл", "Класс", "Уверенность", "Сегментов", "Длина сегмента"]
    logging.info("\nРезультаты предсказания:\n" + tabulate(results, headers=headers, tablefmt="pretty"))

    return results