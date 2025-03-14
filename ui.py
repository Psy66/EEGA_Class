# ui.py
import tkinter as tk
from functools import partial
from tkinter import scrolledtext, ttk
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EEGUI:
    """
    Class for creating the graphical user interface (GUI) for EEG analysis.

    Attributes:
        root (tk.Tk): Root window of the application.
        controller (EEGController): Controller for managing application logic.
    """

    def __init__(self, root: tk.Tk, controller):
        """
        Initializes the GUI.

        Args:
            root (tk.Tk): Root window of the application.
            controller (EEGController): Controller for managing application logic.
        """
        self.root = root
        self.controller = controller
        self.root.title("EEG Analysis Tool")
        self.root.geometry("1450x1000")
        self.setup_ui()

    def setup_ui(self) -> None:
        """Sets up the user interface."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.setup_left_frame()
        self.setup_right_frame()

    def setup_left_frame(self) -> None:
        """Sets up the left panel of the interface."""
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill="both", expand=True)
        self.setup_buttons_frame()
        self.setup_settings_frame()

    def setup_buttons_frame(self) -> None:
        """Sets up the panel with buttons."""
        self.buttons_frame = ttk.LabelFrame(self.left_frame, text="Actions")
        self.buttons_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Load Data", command=self.controller.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Train", command=self.controller.train_model).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Predict", command=self.controller.predict_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Save Settings", command=self.controller.save_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5, pady=5)

    def setup_settings_frame(self) -> None:
        """Sets up the panel with settings."""
        self.settings_frame = ttk.LabelFrame(self.left_frame, text="Settings")
        self.settings_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.entries = {}
        for i, (setting, value) in enumerate(self.controller.get_settings()):
            ttk.Label(self.settings_frame, text=f"{setting}:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(self.settings_frame, width=40)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.entries[setting] = entry
            if setting in ["Location of training files", "Location of the CSV file with labels",
                           "Location of the saved model", "Location of files for predictions"]:
                btn_text = "Select File" if setting == "Location of the CSV file with labels" or setting == "Location of the saved model" else "Select Folder"
                btn_command = partial(self.controller.select_path, setting, is_file=setting == "Location of the CSV file with labels" or setting == "Location of the saved model")
                ttk.Button(self.settings_frame, text=btn_text, command=btn_command).grid(row=i, column=2, padx=5, pady=5)

    def setup_right_frame(self) -> None:
        """Sets up the right panel of the interface."""
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill="both", expand=True)
        self.setup_status_frame()
        self.setup_plot_frame()

    def setup_status_frame(self) -> None:
        """Sets up the status panel."""
        self.status_frame = ttk.LabelFrame(self.right_frame, text="Status")
        self.status_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.status_text = scrolledtext.ScrolledText(self.status_frame, wrap=tk.WORD, state="disabled")
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.progress = ttk.Progressbar(self.status_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=5)

    def setup_plot_frame(self) -> None:
        """Sets up the panel for displaying plots."""
        self.plot_frame = ttk.LabelFrame(self.right_frame, text="Training Plot")
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax2 = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def log_status(self, message: str) -> None:
        """
        Logs a message to the status panel.

        Args:
            message (str): Message to log.
        """
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.config(state="disabled")
        self.status_text.yview(tk.END)

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar.

        Args:
            value (int): Progress value (0-100).
        """
        self.progress["value"] = value
        self.root.update_idletasks()

    def plot_training_results(self, train_losses: List[float], train_accuracies: List[float]) -> None:
        """
        Displays the loss and accuracy plot.

        Args:
            train_losses (List[float]): List of losses.
            train_accuracies (List[float]): List of accuracies.
        """
        self.ax.clear()
        self.ax2.clear()
        epochs = range(1, len(train_losses) + 1)
        self.ax.set_xticks(epochs)
        self.ax.plot(epochs, train_losses, label="Training Loss", color="red", linewidth=1, marker="o", markersize=5)
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss", color="red")
        self.ax.tick_params(axis="y", labelcolor="red")
        self.ax2.plot(epochs, train_accuracies, label="Training Accuracy", color="blue", linewidth=1, marker="o", markersize=5)
        self.ax2.set_ylabel("Accuracy", color="blue")
        self.ax2.tick_params(axis="y", labelcolor="blue")
        self.ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        self.ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax2.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_title("Training Loss and Accuracy")
        self.ax2.yaxis.set_label_coords(1.1, 0.5)
        self.fig.legend(loc="upper right")
        self.canvas.draw()