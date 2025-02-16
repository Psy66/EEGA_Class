# main.py
import logging
import torch
from torchinfo import summary
from config import Config
from data_processor import EEGProcessor
from model import CNN
from predict import predict_new_data
from trainer import EEGTrainer

def main() -> None:
    """
    Основная функция для запуска обучения и предсказания модели EEG.

    Этапы:
    1. Загрузка и обработка данных.
    2. Обучение модели.
    3. Оценка модели.
    4. Сохранение модели.
    5. Предсказание на новых данных.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используемое устройство: {device}")

    try:
        # Загрузка и обработка данных
        input("Нажмите Enter для загрузки данных и формирования датасета...")
        x, y, n_cls, n_chan, _, _, file_info = EEGProcessor.load_and_process_data(
            Config.DATA_PATH, Config.LABELS_PATH, Config.SEGMENT_LENGTH
        )
        logging.info("Данные успешно загружены и обработаны.")
        logging.info(file_info)

        # Обучение модели
        input("Нажмите Enter для продолжения к обучению модели...")
        model = CNN(n_cls, n_chan).to(device)
        summary(model, input_size=(Config.BATCH_SIZE, 1, n_chan, Config.SEGMENT_LENGTH))

        trainer = EEGTrainer(model, device)
        train_loader, x_test_tensor, y_test_tensor = trainer.create_data_loaders(x, y)

        trainer.train(train_loader, Config.EPOCHS, Config.LEARNING_RATE)

        # Оценка модели
        metrics = trainer.evaluate(x_test_tensor, y_test_tensor)
        logging.info(f'Precision: {metrics["precision"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}, '
                     f'Recall: {metrics["recall"]:.4f}, F1 Score: {metrics["f1"]:.4f}')

        # Сохранение модели
        torch.save(model.state_dict(), Config.MODEL_PATH)
        logging.info(f"Модель сохранена в: {Config.MODEL_PATH}")

        # Предсказание на новых данных
        input("Нажмите Enter для предсказаний новых данных...")
        labels_df = EEGProcessor.load_labels(Config.LABELS_PATH)
        class_labels = labels_df['key'].dropna().unique().tolist()

        predict_new_data(Config.MODEL_PATH, Config.PRED_PATH, Config.SEGMENT_LENGTH, n_cls, n_chan, class_labels)

    except Exception as e:
        logging.error(f"Ошибка в основном цикле: {e}")
        raise

if __name__ == "__main__":
    main()