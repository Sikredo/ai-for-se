import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from transformers import get_linear_schedule_with_warmup, AdamW
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="ai-for-sevulnerability-detection") #log ROC curve to wandb to also see it when executing on server

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = FocalLoss(alpha=0.25, gamma=2)

# Load pre-computed embeddings and labels
train_embeddings = torch.load('train_embeddings.pt')
train_labels = torch.load('train_labels.pt')
test_embeddings = torch.load('test_embeddings.pt')
test_labels = torch.load('test_labels.pt')

# Oversample vul in training set
train_embeddings_np = train_embeddings.cpu().numpy()
train_labels_np = train_labels.cpu().numpy()
ros = RandomOverSampler()
train_embeddings_resampled, train_labels_resampled = ros.fit_resample(train_embeddings_np, train_labels_np)
train_embeddings_resampled = torch.tensor(train_embeddings_resampled).to(device)
train_labels_resampled = torch.tensor(train_labels_resampled).to(device)

train_dataset = TensorDataset(train_embeddings_resampled, train_labels_resampled)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(test_embeddings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8)


class LSTMVulnerabilityClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.1):
        super(LSTMVulnerabilityClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last hidden state
        x = self.dropout(x)
        x = self.fc(x)
        return x

input_dim = 768
hidden_dim = 512
num_layers = 2
num_classes = 2
classifier = LSTMVulnerabilityClassifier(input_dim, hidden_dim, num_layers, num_classes, 0.1).to(device)

# Defining optimizer and scaler
optimizer = AdamW(classifier.parameters(), lr=1e-4)
#scaler = torch.cuda.amp.GradScaler()

num_epochs = 30
number_of_training_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=number_of_training_steps)
for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(train_loader, desc="Training"):
        embeddings, labels = batch
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(embeddings.unsqueeze(1))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation on test set
classifier.eval()
all_predictions = []
true_labels = []
all_probabilities = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        embeddings, labels = batch
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = classifier(embeddings.unsqueeze(1))
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted_labels = torch.max(outputs, 1)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, all_predictions)
print("\nEvaluation: \nAccuracy=", accuracy)
print(classification_report(true_labels, all_predictions, target_names=["non-vul", "vul"]))

mcc = matthews_corrcoef(true_labels, all_predictions)
print("MCC: ", mcc)

fpr, tpr, _ = roc_curve(true_labels, all_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
wandb.log({"roc_curve": wandb.Image(plt)})
plt.show()

# Checking the distribution of the labels
unique, counts = np.unique(train_labels.numpy(), return_counts=True)
print(f"Training Labels Distribution: {dict(zip(unique, counts))}")

unique, counts = np.unique(test_labels.numpy(), return_counts=True)
print(f"Test Labels Distribution: {dict(zip(unique, counts))}")
