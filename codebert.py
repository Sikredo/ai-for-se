import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from transformers import AutoTokenizer, AutoModel
import torch
from CDataLoader import CDataLoader
from JavaDataLoader import JsonDataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, matthews_corrcoef
import copy
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

datafiles = [
    "./data/original_method.json",
    "./data/rename_only.json",
    "./data/code_structure_change_only.json",
    "./data/full_transformation.json"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_training_and_test_data_per_function(input_json):
    output = []
    for index, vul in enumerate(input_json):
        function_id = vul['vulName']
        code_lines = [{'line': line['code'], 'vulnerable': line['vulnerable']} for line in vul['vulData']]
        output.append({'function': function_id, 'code': code_lines})
    train, test = train_test_split(output, test_size=0.2, random_state=42)
    return train, test

"""
computes the average portion of vulnerable lines per function in the training set and based on that
calculates the optimal class weights for balancing the training set
"""
def get_optimal_label_weights(training_data):
    vulnerable_portions = []

    for func in training_data:
        code_lines = func['code']
        total_lines = len(code_lines)
        vulnerable_lines = sum(1 for line in code_lines if line['vulnerable'])

        if total_lines > 0:
            vulnerable_portion = vulnerable_lines / total_lines
            vulnerable_portions.append(vulnerable_portion)

    if vulnerable_portions:
        average_portion = sum(vulnerable_portions) / len(vulnerable_portions)
    else:
        average_portion = 0.0

    optimal_class_weights = [1.0 / (1 - average_portion), 1.0 / average_portion]

    return optimal_class_weights

vulnerabilities = list()
dataloader = CDataLoader("./bigvul-data/data.json")
vulnerabilities_ = dataloader.get_prepared_data()
#vulnerabilities.extend(vulnerabilities_[0:250])
vulnerabilities.extend(vulnerabilities_[-500:])
training, test = get_training_and_test_data_per_function(vulnerabilities)
label_weights = get_optimal_label_weights(training)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

class BERTClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_of_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_of_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        raw_scores = self.classifier(pooled_output)
        return raw_scores

classifier = BERTClassifier(model, num_of_classes=2)
classifier.to(device)

def prepare_input_data(data, max_length):
    input_ids = []
    attention_masks = []
    labels = []
    total_functions = len(data)
    for idx, function in enumerate(data, 1):
        code_lines = [line["line"] for line in function["code"]]
        tokens = tokenizer(code_lines, add_special_tokens=True, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        labels.extend([int(line['vulnerable']) for line in function["code"]])
        print(f"\rProcessing {idx}/{total_functions} functions...", end='')
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    print("\nAll functions processed.")
    return input_ids, attention_masks, labels


max_length = max(max(len(line["line"]) for function in training for line in function["code"]),
                max(len(line["line"]) for function in test for line in function["code"]))

print("Start preparing training data...")
train_input_ids, train_attention_masks, train_labels = prepare_input_data(training, 512 )
print("Start preparing test data...")
test_input_ids, test_attention_masks, test_labels = prepare_input_data(test, 512 )

classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Include class weights in loss function
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

class VulnerabilityDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx]
        }

# Create the dataset
train_dataset = VulnerabilityDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = VulnerabilityDataset(test_input_ids, test_attention_masks, test_labels)

sampling_weights = [label_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sampling_weights, len(sampling_weights))

# Create DataLoader with sampler
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=8)

# optimizer for model parameters by computing gradient descent
optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)

classifier.train()
num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch", epoch + 1, "/", num_epochs)
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        raw_scores = classifier(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(raw_scores, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted_labels = torch.max(raw_scores, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss /= len(train_loader)
    accuracy = correct_predictions / total_predictions

    print(f"Training: Loss={epoch_loss:.4f} | Accuracy={accuracy:.4f}\n")
print("\n")

classifier.eval()
all_predictions = []
true_labels = []
all_probabilities = []

for batch in tqdm(test_loader, desc="Testing"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    with torch.no_grad():
        raw_scores = classifier(input_ids=input_ids, attention_mask=attention_mask)

    _, predicted_labels = torch.max(raw_scores, 1)
    all_predictions.extend(predicted_labels.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())
    probabilities = torch.softmax(raw_scores, dim=1).cpu().numpy()
    all_probabilities.extend(probabilities[:, 1])

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
plt.show()

# Checking the distribution of the labels
unique, counts = np.unique(train_labels.numpy(), return_counts=True)
print(f"Training Labels Distribution: {dict(zip(unique, counts))}")

unique, counts = np.unique(test_labels.numpy(), return_counts=True)
print(f"Test Labels Distribution: {dict(zip(unique, counts))}")