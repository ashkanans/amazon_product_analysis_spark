import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    @staticmethod
    def train_model(train_features, train_labels, input_dim, epochs=10, lr=0.001):
        model = NeuralNetworkModel(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        return model

    @staticmethod
    def evaluate_model(model, test_features, test_labels):
        model.eval()
        with torch.no_grad():
            # Get predictions
            outputs = model(test_features).view(-1)
            predictions = (outputs > 0.5).float()  # Binary predictions (threshold at 0.5)

            # Calculate metrics
            auc = roc_auc_score(test_labels.numpy(), outputs.numpy())
            accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
            precision = precision_score(test_labels.numpy(), predictions.numpy(), zero_division=0)
            recall = recall_score(test_labels.numpy(), predictions.numpy(), zero_division=0)
            f1 = f1_score(test_labels.numpy(), predictions.numpy(), zero_division=0)

            # Print metrics
            print(f"Model Evaluation Metrics:")
            print(f"AUC: {auc}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1-score: {f1}")

        return {
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }
