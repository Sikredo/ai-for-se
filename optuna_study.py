import numpy as np
import optuna
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#wandb.init(project="vulnerability-detection") #log ROC curve to wandb to also see it when executing on server

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

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_embeddings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define a simple classifier
class SimpleExtractionClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob):
        super(SimpleExtractionClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
# Initialize the classifier
input_dim = train_embeddings.size(1)
num_classes = 2
# Define the objective function
def objective(trial):
    # Hyperparameters to be optimized
    hidden_dim = trial.suggest_int('hidden_dim', 128, 1024)
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)


    classifier = SimpleExtractionClassifier(input_dim, hidden_dim, num_classes, dropout_prob).to(device)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    # Evaluation on validation set (here, we use the test set as validation for simplicity)
    classifier.eval()
    all_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = classifier(embeddings)
            _, predicted_labels = torch.max(outputs, 1)
            all_predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, all_predictions)
    return accuracy

# Create the study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Use the best hyperparameters to train the final model
best_params = study.best_params
classifier = SimpleExtractionClassifier(input_dim, best_params['hidden_dim'], num_classes, best_params['dropout_prob']).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=best_params['learning_rate'])

# Training loop for the final model
num_epochs = 15
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
        outputs = classifier(embeddings)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

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

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        embeddings, labels = batch
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = classifier(embeddings)
        _, predicted_labels = torch.max(outputs, 1)
        all_predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, all_predictions)
print("\nEvaluation: \nAccuracy=", accuracy)
print(classification_report(true_labels, all_predictions, target_names=["non-vul", "vul"]))

mcc = matthews_corrcoef(true_labels, all_predictions)
print("MCC: ", mcc)

fpr, tpr, _ = roc_curve(true_labels, all_predictions)
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
#wandb.log({"roc_curve": wandb.Image(plt)})
plt.show()

# Checking the distribution of the labels
unique, counts = np.unique(train_labels.numpy(), return_counts=True)
print(f"Training Labels Distribution: {dict(zip(unique, counts))}")

unique, counts = np.unique(test_labels.numpy(), return_counts=True)
print(f"Test Labels Distribution: {dict(zip(unique, counts))}")