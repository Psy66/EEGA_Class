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
    Main function for running EEG model training and prediction.

    Steps:
    1. Load and process data.
    2. Train the model.
    3. Evaluate the model.
    4. Save the model.
    5. Predict on new data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load and process data
        input("Press Enter to load data and create the dataset...")
        x, y, n_cls, n_chan, _, _, file_info = EEGProcessor.load_and_process_data(
            Config.DATA_PATH, Config.LABELS_PATH, Config.SEGMENT_LENGTH
        )
        logging.info("Data successfully loaded and processed.")
        logging.info(file_info)

        # Train the model
        input("Press Enter to proceed to model training...")
        model = CNN(n_cls, n_chan).to(device)
        summary(model, input_size=(Config.BATCH_SIZE, 1, n_chan, Config.SEGMENT_LENGTH))

        trainer = EEGTrainer(model, device)
        train_loader, x_test_tensor, y_test_tensor = trainer.create_data_loaders(x, y)

        trainer.train(train_loader, Config.EPOCHS, Config.LEARNING_RATE)

        # Evaluate the model
        metrics = trainer.evaluate(x_test_tensor, y_test_tensor)
        logging.info(f'Precision: {metrics["precision"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}, '
                     f'Recall: {metrics["recall"]:.4f}, F1 Score: {metrics["f1"]:.4f}')

        # Save the model
        torch.save(model.state_dict(), Config.MODEL_PATH)
        logging.info(f"Model saved to: {Config.MODEL_PATH}")

        # Predict on new data
        input("Press Enter to predict new data...")
        labels_df = EEGProcessor.load_labels(Config.LABELS_PATH)
        class_labels = labels_df['key'].dropna().unique().tolist()

        predict_new_data(Config.MODEL_PATH, Config.PRED_PATH, Config.SEGMENT_LENGTH, n_cls, n_chan, class_labels)

    except Exception as e:
        logging.error(f"Error in the main loop: {e}")
        raise

if __name__ == "__main__":
    main()