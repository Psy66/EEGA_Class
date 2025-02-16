# EEGApp.py
import tkinter as tk
from ui import EEGUI
from controller import EEGController

if __name__ == "__main__":
    root = tk.Tk()
    controller = EEGController()
    ui = EEGUI(root, controller)
    controller.set_ui(ui)
    root.mainloop()