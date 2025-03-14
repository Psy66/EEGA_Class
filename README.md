# EEG Classification Software

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![MNE](https://img.shields.io/badge/MNE-0.23%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project is designed to classify EEG signals using a convolutional neural network (CNN). The program uses the PyTorch library to train the model on data recorded in EDF files. The model is trained on signal segments that are filtered, normalized, and split into parts of a specified length. Classification is performed based on the labels specified in the CSV file.

---

## 🌟 Key Features

- **Data Loading**: Support for EDF files (extension must be in uppercase, e.g., `.EDF`).
- **Preprocessing**:
  - Signal filtering (lower and upper thresholds).
  - Amplitude normalization.
  - Splitting into segments of a specified length.
- **Model Training**: Using CNN to classify EEG segments.
- **Prediction**: Classifying new data using the trained model.
- **GPU support**: Ability to use a graphics card to speed up training (requires PyTorch with CUDA support).

---

## 💻 Requirements

1. **Data**:
   - EDF files with EEG signals.
   - CSV file with class labels. Format:
     ```
     Filename,key
     file1.EDF,1
     file2.EDF,2
     ```
     - `Filename`: Full file name (case sensitive).
     - `key`: Classification code (number).

2. **Environment**:
   - Python 3.8 or higher.
   - Installed dependencies (see `requirements.txt`).

3. **Hardware**:
   - For GPU use: CUDA-capable graphics card (recommended).

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/psy66/EEG_Class.git
   cd EEG_Class
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install PyTorch with CUDA support (if required):
   - Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

---

## ⚙️ Project Structure

### Modules

1. **`config.py`**:
   - Stores and manages the program configuration.
   - Settings for data paths, model parameters, training hyperparameters.
   - Ability to save and load the configuration to/from a file.

2. **`data_processor.py`**:
   - Loads and preprocesses data:
     - Reads EDF files.
     - Filters and normalizes signals.
     - Splits into segments.
     - Forms tensors for training.

3. **`model.py`**:
   - Implements CNN for EEG classification.
   - Model architecture:
     - Convolutional layers.
     - Fully connected layers.
     - Function for calculating the size of input data.

4. **`trainer.py`**:
   - Trains the model:
     - Splits the data into training and testing sets.
     - Sets up the optimizer and loss function.
     - Logs the training process.
   - Evaluates the model:
     - Metrics: precision, recall, F1-score.

5. **`predict.py`**:
   - Predicts classes for new data.
   - Processes EDF files:
     - Segments them.
     - Classifies using the trained model.
     - Outputs results in tabular format.

6. **`ui.py`**:
   - Provides graphical user interface (GUI):
     - Buttons for loading data, training, prediction.
     - Displays status and progress.
     - Visualizes training results (loss and accuracy graphs).

7. **`controller.py`**:
   - Manages application logic:
     - Interacts between the GUI and other modules.
     - Uses multithreading for long-running operations (data loading, training, prediction).

8. **`EEGApp.py`**:
   - Acts as the application's entry point.
   - Initializes the GUI and controller.

---

## 🏃‍♂️ Usage

### Running the Program

1. Configure settings in `config.py`:
   - Specify paths to data, models, and other parameters.

2. Launch the app:
   ```bash
   python EEGApp.py
   ```

3. Use the GUI to:
   - Load data.
   - Train the model.
   - Predict classes for new data.

---

## 📖 Examples

### Data Loading
- In `config.py`, specify the path to the directory containing EDF files and the CSV label file.

### Model Training
- Click the "Train" button in the GUI.
- Training results will appear in logs and on the graph.

### Prediction
- Click the "Predict" button in the GUI.
- Prediction results will display in a table.

---

## 📜 License

This project is licensed under the MIT License. For details, see the LICENSE file.

---

## 👨‍💻 Author

Timur Petrenko  
📧 Email: psy66@narod.ru

---

## 📚 Citation

If you use this tool in your research, please consider citing it as follows:

```
Petrenko, Timur. EEG Classification Software. 2025. Available on GitHub: https://github.com/Psy66/EEGA_Class.
```

---

## 🎉 Acknowledgments

- PyTorch for its powerful machine learning tools.
- MNE-Python for easy handling of EEG data.

---

### 📢 Important Note

This application is intended for educational and research purposes only. Use it at your own risk. The author does not take any responsibility for potential issues or damages caused by the use of this software.

---