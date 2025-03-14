# controller.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Callable
import torch
from tabulate import tabulate
from config import Config
from data_processor import EEGProcessor
from model import CNN
from predict import predict_new_data
from trainer import EEGTrainer

class EEGController:
    """
    Controller for managing the logic of the EEG Analysis Tool application.

    Attributes:
        ui (EEGUI): Graphical user interface.
        device (torch.device): Device for computation (CPU/GPU).
    """

    def __init__(self, ui=None):
        """
        Initializes the controller.

        Args:
            ui (EEGUI, optional): Graphical user interface. Defaults to None.
        """
        self.ui = ui
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.ui:
            self.ui.log_status(f"Using device: {self.device}")

    def set_ui(self, ui):
        """
        Sets the graphical user interface.

        Args:
            ui (EEGUI): Graphical user interface.
        """
        self.ui = ui
        self.ui.log_status(f"Using device: {self.device}")

    @staticmethod
    def get_settings() -> List[Tuple[str, str]]:
        """
        Returns the current settings.

        Returns:
            List[Tuple[str, str]]: List of tuples (setting name, value).
        """
        return [
            ("Location of training files", Config.DATA_PATH),
            ("Location of the CSV file with labels", Config.LABELS_PATH),
            ("Location of the saved model", Config.MODEL_PATH),
            ("Location of files for predictions", Config.PRED_PATH),
            ("Length of one data segment", Config.SEGMENT_LENGTH),
            ("Batch size", Config.BATCH_SIZE),
            ("Number of training epochs", Config.EPOCHS),
            ("Learning rate", Config.LEARNING_RATE),
            ("Test set size", Config.TEST_SIZE),
            ("Lower threshold for EEG signal filtering", Config.LOW_PASS_F),
            ("Upper threshold for EEG signal filtering", Config.HIGH_PASS_F),
            ("Learning rate decay factor", Config.LEARNING_RATE_DECAY),
        ]

    def select_path(self, setting: str, is_file: bool = False) -> None:
        """
        Selects a file or directory path.

        Args:
            setting (str): Setting name.
            is_file (bool, optional): If True, selects a file. Defaults to False (selects a directory).
        """
        initial_dir = os.path.dirname(self.ui.entries[setting].get()) if self.ui.entries[setting].get() else os.getcwd()
        if is_file:
            path = filedialog.askopenfilename(initialdir=initial_dir, title=f"Select file for {setting}")
        else:
            path = filedialog.askdirectory(initialdir=initial_dir, title=f"Select folder for {setting}")
        if path:
            self.ui.entries[setting].delete(0, tk.END)
            self.ui.entries[setting].insert(0, path)

    @staticmethod
    def run_in_thread(func: Callable) -> Callable:
        """
        Decorator to run a function in a separate thread.

        Args:
            func (Callable): Function to run in a separate thread.

        Returns:
            Callable: Wrapped function.
        """
        def wrapper(*args, **kwargs):
            threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
        return wrapper

    @staticmethod
    def handle_errors(func: Callable) -> Callable:
        """
        Decorator to handle errors.

        Args:
            func (Callable): Function to wrap.

        Returns:
            Callable: Wrapped function.
        """
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.handle_error(f"Error in function {func.__name__}", e)
        return wrapper

    @run_in_thread
    @handle_errors
    def load_data(self) -> None:
        """Loads data."""
        self.ui.log_status("Loading data...")
        self.ui.update_progress(0)
        self.x, self.y, self.n_cls, self.n_chan, processed_files, skipped_files, file_info = EEGProcessor.load_and_process_data(
            Config.DATA_PATH, Config.LABELS_PATH, Config.SEGMENT_LENGTH
        )
        labels_df = EEGProcessor.load_labels(Config.LABELS_PATH)
        self.class_labels = labels_df['key'].dropna().unique().tolist()
        total_files = len([f for f in os.listdir(Config.DATA_PATH) if f.endswith('.EDF')])
        self.ui.log_status(file_info)
        self.ui.log_status("Data successfully loaded and preprocessed!")
        self.ui.log_status(f"Total files in folder: {total_files}")
        self.ui.log_status(f"Processed files: {processed_files}")
        self.ui.log_status(f"Skipped files (no labels): {skipped_files}")
        self.ui.log_status(f"Number of classes: {self.n_cls}, Classes: {', '.join(self.class_labels)}")
        self.ui.log_status(f"Training set size: {self.x.shape}")
        self.ui.log_status(f"Number of channels: {self.n_chan}")
        self.ui.update_progress(100)

    @run_in_thread
    @handle_errors
    def train_model(self) -> None:
        """Starts model training."""
        self.ui.log_status("Training model...")
        self.ui.update_progress(0)
        model = CNN(self.n_cls, self.n_chan).to(self.device)
        trainer = EEGTrainer(model, self.device)
        train_loader, x_test_tensor, y_test_tensor = trainer.create_data_loaders(self.x, self.y)
        epoch_results = trainer.train(train_loader, Config.EPOCHS, Config.LEARNING_RATE)
        train_losses = [float(result[1]) for result in epoch_results]
        train_accuracies = [float(result[2]) for result in epoch_results]
        self.ui.plot_training_results(train_losses, train_accuracies)
        table = tabulate(epoch_results, headers=["Epoch", "Average Loss", "Accuracy", "Learning Rate"],
                         tablefmt="pretty")
        self.ui.log_status("\nTraining results:")
        self.ui.log_status(table)
        metrics = trainer.evaluate(x_test_tensor, y_test_tensor)
        self.ui.log_status("\nModel Metrics:")
        self.ui.log_status(f"Precision: {metrics['precision']:.4f}")
        self.ui.log_status(f"Recall: {metrics['recall']:.4f}")
        self.ui.log_status(f"F1 Score: {metrics['f1']:.4f}")
        self.ui.log_status(f"Accuracy: {metrics['accuracy']:.4f}")
        torch.save(model.state_dict(), Config.MODEL_PATH)
        self.ui.log_status(f"Model saved to: {Config.MODEL_PATH}")
        self.ui.update_progress(100)

    @run_in_thread
    @handle_errors
    def predict_data(self) -> None:
        """Starts prediction."""
        self.ui.log_status("Running prediction...")
        self.ui.update_progress(0)
        results = predict_new_data(Config.MODEL_PATH, Config.PRED_PATH, Config.SEGMENT_LENGTH, self.n_cls,
                                   self.n_chan, self.class_labels)
        self.ui.log_status("\nPrediction results:")
        self.ui.status_text.config(state="normal")
        self.ui.status_text.config(state="disabled")
        table_header = "| {:<35} | {:<5} | {:<11} | {:<9} | {:<14} |\n".format(
            "File", "Class", "Confidence", "Segments", "Segment Length"
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
        self.ui.log_status("Prediction completed.")
        self.ui.update_progress(100)

    @handle_errors
    def save_settings(self) -> None:
        """Saves settings."""
        key_mapping = {
            "Location of training files": "DATA_PATH",
            "Location of the CSV file with labels": "LABELS_PATH",
            "Location of the saved model": "MODEL_PATH",
            "Location of files for predictions": "PRED_PATH",
            "Length of one data segment": "SEGMENT_LENGTH",
            "Batch size": "BATCH_SIZE",
            "Number of training epochs": "EPOCHS",
            "Learning rate": "LEARNING_RATE",
            "Test set size": "TEST_SIZE",
            "Lower threshold for EEG signal filtering": "LOW_PASS_F",
            "Upper threshold for EEG signal filtering": "HIGH_PASS_F",
            "Learning rate decay factor": "LEARNING_RATE_DECAY",
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
        self.ui.log_status("Configuration successfully saved and updated.")

    def handle_error(self, message: str, error: Exception) -> None:
        """
        Handles errors and displays a message.

        Args:
            message (str): Error message.
            error (Exception): Exception object.
        """
        self.ui.log_status(f"{message}: {error}")
        messagebox.showerror("Error", f"{message}: {error}")
        self.ui.update_progress(0)