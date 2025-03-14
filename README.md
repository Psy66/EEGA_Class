## EEG Classification Software

This project is designed to classify EEG signals using a convolutional neural network (CNN). The program uses the PyTorch library to train the model on data recorded in EDF files. The model is trained on signal segments that are filtered, normalized, and split into parts of a specified length. Classification is performed based on the labels specified in the CSV file.

---

## Key Features

- **Data Loading**: Support for EDF files (the extension must be in uppercase, for example, `.EDF`).
- **Preprocessing**:
- Signal filtering (lower and upper thresholds).
- Amplitude normalization.
- Splitting into segments of a specified length.
- **Model Training**: Using CNN to classify EEG segments.
- **Prediction**: Classifying new data using the trained model.
- **GPU support**: Ability to use a graphics card to speed up training (requires PyTorch with CUDA support).

---

## Requirements

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

## Installation

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

## Project structure

### Modules

1. **`config.py`**:
- Storing and managing the program configuration.
- Settings for data paths, model parameters, training hyperparameters.
- Ability to save and load the configuration to/from a file.

2. **`data_processor.py`**:
- Loading and preprocessing data:
- Reading EDF files.
- Filtering and normalizing signals.
- Splitting into segments.
- Forming tensors for training.

3. **`model.py`**:
- Implementation of CNN for EEG classification.
- Model architecture:
- Convolutional layers.
- Fully connected layers.
- Function for calculating the size of input data.

4. **`trainer.py`**:
- Training the model:
- Splitting the data into training and testing sets.
- Setting up the optimizer and loss function.
- Logging the training process.
- Model evaluation:
- Metrics: precision, recall, F1-score.

5. **`predict.py`**:
- Predicting classes for new data.
- EDF file processing:
- Segmentation.
- Classification using the trained model.
- Tabular output.

6. **`ui.py`**:
- Graphical user interface (GUI):
- Buttons for loading data, training, prediction.
- Displaying status and progress.
- Visualization of training results (loss and accuracy graphs).

7. **`controller.py`**:
- Application logic management:
- Interaction between the GUI and other modules.
- Multithreading for long-running operations (loading data, training, prediction).

8. **`EEGApp.py`**:
- Application entry point.
- Initialization of the GUI and controller.

---

## Usage

### Running the program

1. Set up the configuration in `config.py`:
- Specify the paths to the data, model, and other parameters.

2. Run the application:
```bash
python EEGApp.py
```

3. Use the GUI to:
- Load data.
- Train the model.
- Predict classes for new data.

### Command line example

To train the model:
```bash
python main.py
```

---

## Examples

### Load data
- Specify the path to the directory with the EDF files and the CSV file with labels in `config.py`.

### Train the model
- Click the "Train" button in the GUI.
- The training results will be displayed in the logs and on the graph.

### Prediction
- Click the "Predict" button in the GUI.
- The prediction results will be displayed in the table.

---

## License

This project is distributed under the MIT license. For details, see the [LICENSE](LICENSE) file.

---

## Authors

- [Psy66](https://github.com/psy66)

---

## Acknowledgments

- PyTorch for providing a powerful tool for machine learning.
- MNE-Python for convenient work with EEG data.

---

## Программа для классификации ЭЭГ

Этот проект предназначен для классификации сигналов ЭЭГ с использованием сверточной нейронной сети (CNN). Программа использует библиотеку PyTorch для обучения модели на данных, записанных в файлах формата EDF. Модель обучается на сегментах сигналов, которые фильтруются, нормализуются и разбиваются на части указанной длины. Классификация осуществляется на основе меток, указанных в CSV-файле.

---

## Основные возможности

- **Загрузка данных**: Поддержка файлов EDF (расширение должно быть в верхнем регистре, например, `.EDF`).
- **Препроцессинг**:
  - Фильтрация сигналов (нижний и верхний пороги).
  - Нормализация амплитуды.
  - Разбиение на сегменты указанной длины.
- **Обучение модели**: Использование CNN для классификации сегментов ЭЭГ.
- **Предсказание**: Классификация новых данных с использованием обученной модели.
- **Поддержка GPU**: Возможность использования видеокарты для ускорения обучения (требуется установка PyTorch с поддержкой CUDA).

---

## Требования

1. **Данные**:
   - Файлы EDF с сигналами ЭЭГ.
   - CSV-файл с метками классов. Формат:
     ```
     Filename,key
     file1.EDF,1
     file2.EDF,2
     ```
     - `Filename`: Полное имя файла (регистр имеет значение).
     - `key`: Код классификации (число).

2. **Окружение**:
   - Python 3.8 или выше.
   - Установленные зависимости (см. `requirements.txt`).

3. **Аппаратное обеспечение**:
   - Для использования GPU: видеокарта с поддержкой CUDA (рекомендуется).

---

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/psy66/EEG_Class.git
   cd EEG_Class
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Установите PyTorch с поддержкой CUDA (если требуется):
   - Следуйте инструкциям на [официальном сайте PyTorch](https://pytorch.org/get-started/locally/).

---

## Структура проекта

### Модули

1. **`config.py`**:
   - Хранение и управление конфигурацией программы.
   - Настройки путей к данным, параметров модели, гиперпараметров обучения.
   - Возможность сохранения и загрузки конфигурации в/из файла.

2. **`data_processor.py`**:
   - Загрузка и предобработка данных:
     - Чтение файлов EDF.
     - Фильтрация и нормализация сигналов.
     - Разбиение на сегменты.
   - Формирование тензоров для обучения.

3. **`model.py`**:
   - Реализация CNN для классификации ЭЭГ.
   - Архитектура модели:
     - Сверточные слои.
     - Полносвязные слои.
     - Функция для вычисления размера входных данных.

4. **`trainer.py`**:
   - Обучение модели:
     - Разделение данных на обучающую и тестовую выборки.
     - Настройка оптимизатора и функции потерь.
     - Логирование процесса обучения.
   - Оценка модели:
     - Метрики: точность, полнота, F1-мера.

5. **`predict.py`**:
   - Предсказание классов для новых данных.
   - Обработка файлов EDF:
     - Разбиение на сегменты.
     - Классификация с использованием обученной модели.
   - Вывод результатов в табличном формате.

6. **`ui.py`**:
   - Графический интерфейс пользователя (GUI):
     - Кнопки для загрузки данных, обучения, предсказания.
     - Отображение статуса и прогресса.
     - Визуализация результатов обучения (графики потерь и точности).

7. **`controller.py`**:
   - Управление логикой приложения:
     - Взаимодействие между GUI и остальными модулями.
     - Многопоточность для выполнения длительных операций (загрузка данных, обучение, предсказание).

8. **`EEGApp.py`**:
   - Точка входа в приложение.
   - Инициализация GUI и контроллера.

---

## Использование

### Запуск программы

1. Настройте конфигурацию в `config.py`:
   - Укажите пути к данным, модели и другим параметрам.

2. Запустите приложение:
   ```bash
   python EEGApp.py
   ```

3. Используйте графический интерфейс для:
   - Загрузки данных.
   - Обучения модели.
   - Предсказания классов для новых данных.

### Пример командной строки

Для обучения модели:
```bash
python main.py
```

---

## Примеры

### Загрузка данных
- Укажите путь к директории с файлами EDF и CSV-файлу с метками в `config.py`.

### Обучение модели
- Нажмите кнопку "Train" в GUI.
- Результаты обучения будут отображены в логах и на графике.

### Предсказание
- Нажмите кнопку "Predict" в GUI.
- Результаты предсказания будут отображены в таблице.

---

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

---

## Авторы

- [Psy66](https://github.com/psy66)

---

## Благодарности

- PyTorch за предоставление мощного инструмента для машинного обучения.
- MNE-Python за удобную работу с данными ЭЭГ.

---
