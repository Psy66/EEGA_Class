# controller.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple
import torch
from tabulate import tabulate
from config import Config
from data_processor import EEGProcessor
from model import CNN
from predict import predict_new_data
from trainer import EEGTrainer

class EEGController:
    """
    Контроллер для управления логикой приложения EEG Analysis Tool.

    Атрибуты:
        ui (EEGUI): Графический интерфейс пользователя.
        device (torch.device): Устройство для вычислений (CPU/GPU).
    """

    def __init__(self, ui=None):
        """
        Инициализация контроллера.

        Аргументы:
            ui (EEGUI, optional): Графический интерфейс пользователя. По умолчанию None.
        """
        self.ui = ui
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.ui:
            self.ui.log_status(f"Используемое устройство: {self.device}")

    def set_ui(self, ui):
        """
        Устанавливает графический интерфейс пользователя.

        Аргументы:
            ui (EEGUI): Графический интерфейс пользователя.
        """
        self.ui = ui
        self.ui.log_status(f"Используемое устройство: {self.device}")

    @staticmethod
    def get_settings() -> List[Tuple[str, str]]:
        """
        Возвращает текущие настройки.

        Возвращает:
            List[Tuple[str, str]]: Список кортежей (название настройки, значение).
        """
        return [
            ("Расположение файлов для обучения", Config.DATA_PATH),
            ("Расположение csv файла с метками", Config.LABELS_PATH),
            ("Расположение сохранённой модели", Config.MODEL_PATH),
            ("Расположение файлов для предсказаний", Config.PRED_PATH),
            ("Длина одного сегмента данных", Config.SEGMENT_LENGTH),
            ("Размер батча", Config.BATCH_SIZE),
            ("Количество эпох обучения", Config.EPOCHS),
            ("Скорость обучения", Config.LEARNING_RATE),
            ("Размер тестового набора", Config.TEST_SIZE),
            ("Нижний порог", Config.LOW_PASS_F),
            ("Верхний порог", Config.HIGH_PASS_F),
            ("Коэффициент уменьшения скорости", Config.LEARNING_RATE_DECAY),
        ]

    def select_path(self, setting: str, is_file: bool = False) -> None:
        """
        Выбирает путь к файлу или директории.

        Аргументы:
            setting (str): Название настройки.
            is_file (bool, optional): Если True, выбирается файл. По умолчанию False (выбор директории).
        """
        initial_dir = os.path.dirname(self.ui.entries[setting].get()) if self.ui.entries[setting].get() else os.getcwd()
        if is_file:
            path = filedialog.askopenfilename(initialdir=initial_dir, title=f"Выберите файл для {setting}")
        else:
            path = filedialog.askdirectory(initialdir=initial_dir, title=f"Выберите папку для {setting}")
        if path:
            self.ui.entries[setting].delete(0, tk.END)
            self.ui.entries[setting].insert(0, path)

    def load_data(self) -> None:
        """Загружает данные в отдельном потоке."""
        threading.Thread(target=self._load_data_thread, daemon=True).start()

    def _load_data_thread(self) -> None:
        """Поток для загрузки данных."""
        self.ui.log_status("Загрузка данных...")
        self.ui.update_progress(0)
        try:
            self.x, self.y, self.n_cls, self.n_chan, processed_files, skipped_files, file_info = EEGProcessor.load_and_process_data(
                Config.DATA_PATH, Config.LABELS_PATH, Config.SEGMENT_LENGTH
            )
            labels_df = EEGProcessor.load_labels(Config.LABELS_PATH)
            self.class_labels = labels_df['key'].dropna().unique().tolist()
            total_files = len([f for f in os.listdir(Config.DATA_PATH) if f.endswith('.EDF')])
            self.ui.log_status(file_info)
            self.ui.log_status("Данные успешно загружены и предобработаны!")
            self.ui.log_status(f"Всего файлов в папке: {total_files}")
            self.ui.log_status(f"Обработано файлов: {processed_files}")
            self.ui.log_status(f"Пропущено файлов (без меток): {skipped_files}")
            self.ui.log_status(f"Количество классов: {self.n_cls}, Классы: {', '.join(self.class_labels)}")
            self.ui.log_status(f"Размер тренировочного набора: {self.x.shape}")
            self.ui.log_status(f"Количество каналов: {self.n_chan}")
            self.ui.update_progress(100)
        except Exception as e:
            self.handle_error("Ошибка при загрузке данных", e)

    def train_model(self) -> None:
        """Запускает обучение модели в отдельном потоке."""
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self) -> None:
        """Поток для обучения модели."""
        self.ui.log_status("Идет обучение модели...")
        self.ui.update_progress(0)
        try:
            model = CNN(self.n_cls, self.n_chan).to(self.device)
            trainer = EEGTrainer(model, self.device)
            train_loader, x_test_tensor, y_test_tensor = trainer.create_data_loaders(self.x, self.y)
            epoch_results = trainer.train(train_loader, Config.EPOCHS, Config.LEARNING_RATE)
            train_losses = [float(result[1]) for result in epoch_results]
            train_accuracies = [float(result[2]) for result in epoch_results]
            self.ui.plot_training_results(train_losses, train_accuracies)
            table = tabulate(epoch_results, headers=["Эпоха", "Средняя потеря", "Точность", "Скорость обучения"],
                             tablefmt="pretty")
            self.ui.log_status("\nРезультаты обучения:")
            self.ui.log_status(table)
            metrics = trainer.evaluate(x_test_tensor, y_test_tensor)
            self.ui.log_status("\nModel Metrics:")
            self.ui.log_status(f"Precision: {metrics['precision']:.4f}")
            self.ui.log_status(f"Recall: {metrics['recall']:.4f}")
            self.ui.log_status(f"F1 Score: {metrics['f1']:.4f}")
            self.ui.log_status(f"Accuracy: {metrics['accuracy']:.4f}")
            torch.save(model.state_dict(), Config.MODEL_PATH)
            self.ui.log_status(f"Model saved to {Config.MODEL_PATH}")
            self.ui.update_progress(100)
        except Exception as e:
            self.handle_error("Ошибка при обучении модели", e)

    def predict_data(self) -> None:
        """Запускает предсказание в отдельном потоке."""
        threading.Thread(target=self._predict_data_thread, daemon=True).start()

    def _predict_data_thread(self) -> None:
        """Поток для предсказания."""
        self.ui.log_status("Идет предсказание...")
        self.ui.update_progress(0)
        try:
            results = predict_new_data(Config.MODEL_PATH, Config.PRED_PATH, Config.SEGMENT_LENGTH, self.n_cls,
                                       self.n_chan, self.class_labels)
            self.ui.log_status("\nРезультаты предсказания:")
            self.ui.status_text.config(state="normal")
            self.ui.status_text.config(state="disabled")
            table_header = "| {:<35} | {:<5} | {:<11} | {:<9} | {:<14} |\n".format(
                "Файл", "Класс", "Уверенность", "Сегментов", "Длина сегмента"
            )
            separator = "-" * len(table_header) + "\n"
            self.ui.status_text.config(state="normal")
            self.ui.status_text.insert(tk.END, separator)
            self.ui.status_text.insert(tk.END, table_header)
            self.ui.status_text.insert(tk.END, separator)

            for result in results:
                if isinstance(result, list) and len(result) >= 3:
                    file_name, predicted_class, confidence, segments, segment_length = result
                    confidence_percent = float(confidence.strip('%'))
                    table_row = "| {:<35} | {:<5} | {:<11} | {:<9} | {:<14} |\n".format(
                        file_name, predicted_class, confidence, segments, segment_length
                    )
                    if confidence_percent >= 90:
                        self.ui.status_text.tag_config("green", foreground="green")
                        self.ui.status_text.insert(tk.END, table_row, "green")
                    elif confidence_percent >= 70:
                        self.ui.status_text.tag_config("orange", foreground="orange")
                        self.ui.status_text.insert(tk.END, table_row, "orange")
                    else:
                        self.ui.status_text.tag_config("red", foreground="red")
                        self.ui.status_text.insert(tk.END, table_row, "red")

            self.ui.status_text.insert(tk.END, separator)
            self.ui.status_text.config(state="disabled")
            self.ui.log_status("Предсказание завершено.")
            self.ui.update_progress(100)
        except Exception as e:
            self.handle_error("Ошибка при предсказании", e)

    def save_settings(self) -> None:
        """Сохраняет настройки."""
        try:
            key_mapping = {
                "Расположение файлов для обучения": "DATA_PATH",
                "Расположение csv файла с метками": "LABELS_PATH",
                "Расположение сохранённой модели": "MODEL_PATH",
                "Расположение файлов для предсказаний": "PRED_PATH",
                "Длина одного сегмента данных": "SEGMENT_LENGTH",
                "Размер батча": "BATCH_SIZE",
                "Количество эпох обучения": "EPOCHS",
                "Скорость обучения": "LEARNING_RATE",
                "Размер тестового набора": "TEST_SIZE",
                "Нижний порог": "LOW_PASS_F",
                "Верхний порог": "HIGH_PASS_F",
                "Коэффициент уменьшения скорости": "LEARNING_RATE_DECAY",
            }
            for russian_key, entry in self.ui.entries.items():
                value = entry.get()
                english_key = key_mapping.get(russian_key)
                if english_key:
                    if english_key in ["SEGMENT_LENGTH", "BATCH_SIZE", "EPOCHS"]:
                        value = int(value)
                    elif english_key in ["LEARNING_RATE", "TEST_SIZE", "LOW_PASS_F", "HIGH_PASS_F",
                                         "LEARNING_RATE_DECAY"]:
                        value = float(value)
                    else:
                        value = str(value)
                    setattr(Config, english_key, value)
            Config.save_to_file()
            self.ui.log_status("Конфигурация успешно сохранена и обновлена.")
        except Exception as e:
            self.handle_error("Ошибка при сохранении конфигурации", e)

    def handle_error(self, message: str, error: Exception) -> None:
        """
        Обрабатывает ошибки и выводит сообщение.

        Аргументы:
            message (str): Сообщение об ошибке.
            error (Exception): Объект исключения.
        """
        self.ui.log_status(f"{message}: {error}")
        messagebox.showerror("Error", f"{message}: {error}")
        self.ui.update_progress(0)