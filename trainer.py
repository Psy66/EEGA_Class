# trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import Precision, Recall, F1Score, Accuracy
import logging
from typing import Dict, List, Tuple
from config import Config

class EEGTrainer:
    """
    Class for training and evaluating the EEG model.

    Attributes:
        model (nn.Module): Model to train.
        device (torch.device): Device for computation (CPU/GPU).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): Model to train.
            device (torch.device): Device for computation.
        """
        self.model = model
        self.device = device

    def create_data_loaders(self, features: np.ndarray, labels: np.ndarray) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        """
        Creates DataLoader for training and test tensors.

        Args:
            features (np.ndarray): Array of features.
            labels (np.ndarray): Array of labels.

        Returns:
            Tuple[DataLoader, torch.Tensor, torch.Tensor]:
                - train_loader: DataLoader for training.
                - x_test_tensor: Test data.
                - y_test_tensor: Test labels.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=Config.TEST_SIZE, random_state=42)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device).squeeze()
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=self.device).squeeze()
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=self.device)
        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        return train_loader, x_test_tensor, y_test_tensor

    def train(self, train_loader: DataLoader, epochs: int, learning_rate: float) -> List[List[str]]:
        """
        Trains the model.

        Args:
            train_loader (DataLoader): DataLoader for training.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.

        Returns:
            List[List[str]]: Training results for each epoch.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Config.LEARNING_RATE_DECAY)
        epoch_results = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()
                inputs = inputs.unsqueeze(1).to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = correct / total

            current_lr = scheduler.get_last_lr()[0]
            epoch_results.append([epoch + 1, f"{avg_loss:.4f}", f"{accuracy:.4f}", f"{(current_lr / Config.LEARNING_RATE) * 100:.2f}%"])

            scheduler.step()

        logging.info("\n" + tabulate(epoch_results, headers=["Epoch", "Average Loss", "Accuracy", "Learning Rate"], tablefmt="pretty"))

        return epoch_results

    def evaluate(self, x_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates the model on test data.

        Args:
            x_test_tensor (torch.Tensor): Test data.
            y_test_tensor (torch.Tensor): Test labels.

        Returns:
            Dict[str, float]: Evaluation metrics (precision, recall, f1, accuracy).
        """
        self.model.eval()
        with torch.no_grad():
            pred = torch.max(self.model(x_test_tensor.unsqueeze(1).to(self.device)), dim=1)[1]
            num_classes = len(np.unique(y_test_tensor.cpu().numpy()))
            precision_metric = Precision(task="multiclass", num_classes=num_classes).to(self.device)
            recall_metric = Recall(task="multiclass", num_classes=num_classes).to(self.device)
            f1_metric = F1Score(task="multiclass", num_classes=num_classes).to(self.device)
            accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            precision = precision_metric(pred, y_test_tensor)
            recall = recall_metric(pred, y_test_tensor)
            f1 = f1_metric(pred, y_test_tensor)
            accuracy = accuracy_metric(pred, y_test_tensor)
            return {
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'accuracy': accuracy.item(),
            }