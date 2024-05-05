from transformers import AutoTokenizer, AutoModel
import torch
from DataLoader import JsonDataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import copy
import random

datafiles = [
    "./data/original_method.json",
    "./data/rename_only.json",
    "./data/code_structure_change_only.json",
    "./data/full_transformation.json"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vulnerabilities = list()
for datafile in datafiles:
    dataloader = JsonDataLoader(datafile)
    vulnerabilities.extend(dataloader.get_prepared_data())

# TODO:
#  #remove lines that do not contain statements from the dataset, so that less truncation is necessary


"""
For each function in the training set, iterate through the lines and duplicate vulnerable lines
to match the amount of non-vulnerable lines
"""
def balance_data(training_set):
    balanced_training_set = []
    for function in training_set:
        non_vul_lines = []
        vul_lines = []
        for line in function["code"]:
            if line["vulnerable"]:
                vul_lines.append(line)
            else:
                non_vul_lines.append(line)

        num_of_duplicatoins = len(non_vul_lines) - len(vul_lines)

        if num_of_duplicatoins > 0:
            new_vul_lines = copy.deepcopy(vul_lines) * (num_of_duplicatoins // len(vul_lines))
            balanced_lines = non_vul_lines + new_vul_lines
        else:
            balanced_lines = non_vul_lines + vul_lines

        random.shuffle(balanced_lines)

        balanced_function = copy.deepcopy(function)
        balanced_function["code"] = balanced_lines
        balanced_training_set.append(balanced_function)

    return balanced_training_set


################################################
# using hard-coded example balanced trainings and test set
# TODO: use real data from dataset
################################################

training = [
    {"function": "1",
     "code": [
         {"line": "public static void checkDirectoryTraversal(@NonNull Path parentPath, @NonNull Path pathToCheck) {",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "if (pathToCheck.startsWith(parentPath.normalize())) {",
          "vulnerable": True},
         {"line": "return;",
          "vulnerable": False},
         {"line": "}",
          "vulnerable": False},
         {
             "line": "throw new ForbiddenException(\"You do not have permission to access \" + pathToCheck).setErrorData(pathToCheck);",
             "vulnerable": False},
         {"line": "}",
          "vulnerable": False}

     ]
     },
    {"function": "2",
     "code": [
         {"line": "public static void checkDirectoryTraversal(@NonNull Path parentPath, @NonNull Path pathToCheck) {",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "if (pathToCheck.startsWith(parentPath.normalize())) {",
          "vulnerable": True},
         {"line": "return;",
          "vulnerable": False},
         {"line": "}",
          "vulnerable": False},
         {
             "line": "throw new ForbiddenException(\"You do not have permission to access \" + pathToCheck).setErrorData(pathToCheck);",
             "vulnerable": False},
         {"line": "}",
          "vulnerable": False}

     ]
     }
]

test = [
    {"function": "1",
     "code": [
         {"line": "public static void checkDirectoryTraversal(@NonNull Path parentPath, @NonNull Path pathToCheck) {",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "if (pathToCheck.startsWith(parentPath.normalize())) {",
          "vulnerable": True},
         {"line": "return;",
          "vulnerable": False},
         {"line": "}",
          "vulnerable": False},
         {
             "line": "throw new ForbiddenException(\"You do not have permission to access \" + pathToCheck).setErrorData(pathToCheck);",
             "vulnerable": False},
         {"line": "}",
          "vulnerable": False}

     ]
     },
    {"function": "2",
     "code": [
         {"line": "public static void checkDirectoryTraversal(@NonNull Path parentPath, @NonNull Path pathToCheck) {",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(parentPath, \"Parent path must not be null\");",
          "vulnerable": False},
         {"line": "Assert.notNull(pathToCheck, \"Path to check must not be null\");",
          "vulnerable": False},
         {"line": "if (pathToCheck.startsWith(parentPath.normalize())) {",
          "vulnerable": True},
         {"line": "return;",
          "vulnerable": False},
         {"line": "}",
          "vulnerable": False},
         {
             "line": "throw new ForbiddenException(\"You do not have permission to access \" + pathToCheck).setErrorData(pathToCheck);",
             "vulnerable": False},
         {"line": "}",
          "vulnerable": False}
     ]
     }
]

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)


"""
Custom PyTorch Neural Network Module using the pre-trained BERT model
-> Add extra linear layer to classify lines into vul and non-vul using output of BERT
"""
class BERTClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_of_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)  # drop random 10% of the data to prevent overfitting
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_of_classes)  # add linear layer

    def forward(self, input_ids,
                attention_mask):  # attention mask specifies which of the tokens from input_ids should be taken into account
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # get output from BERT model
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        raw_scores = self.classifier(
            pooled_output)  # pass output from BERT to classifier to get line vulnerability predictions
        return raw_scores


classifier = BERTClassifier(model, num_of_classes=2)
classifier.to(device)


"""
Prepare data to put it into the model:
Tokenize code lines
Extract labels
Generate attention masks (to indicate whether token is an actual token or just padding)

IMPORTANT: context embeddings are not computed, because the BERT model operates directly on the input ids!
-> context embeddings are calculated directly in the model
-> self attention is used to weigh the importance of each token in the context of a sequence
"""
def prepare_input_data(data):
    input_ids = []
    attention_masks = []
    labels = []
    for function in data:
        code_lines = [line["line"] for line in function["code"]]
        tokens = tokenizer(code_lines, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        labels.extend([int(line['vulnerable']) for line in function["code"]])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels


balanced_training = balance_data(training)
train_input_ids, train_attention_masks, train_labels = prepare_input_data(balanced_training)
test_input_ids, test_attention_masks, test_labels = prepare_input_data(test)

# optimizer for model parameters by computing gradient descent
optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)  # lr=learning rate -> TODO: tune!

# loss function for getting probability from raw output (softmax function)
loss_fn = torch.nn.CrossEntropyLoss()


"""
Training the Model:
-> One epoch is an iteration where the whole training set is passed through the NN
-> Multiple iterations for updating weights and biases to optimize them
"""
classifier.train()
num_epochs = 10  # TODO: tune!
for epoch in range(num_epochs):
    print("Epoch", epoch + 1, "/", num_epochs)
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # train model by a) feeding the input ids and attention mask for each function example to the model
    # then b) evaluate the predictions by comparing with the actual label using the cross entropy loss function
    # then c) update model parameters accordingly
    # -> minimize loss (=difference between predicted and actual label) by adapting parameters through backpropagation and optimization
    for i in tqdm(range(len(train_input_ids)), desc="Training"):
        input_ids = train_input_ids[i].unsqueeze(0).to(device)
        attention_mask = train_attention_masks[i].unsqueeze(0).to(device)
        label = train_labels[i].unsqueeze(0).to(device)

        optimizer.zero_grad()
        raw_scores = classifier(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(raw_scores, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted_label = torch.max(raw_scores, 1)
        correct_predictions += (predicted_label == label).item()
        total_predictions += 1

    epoch_loss /= len(train_input_ids)
    accuracy = correct_predictions / total_predictions

    print(f"Training: Loss={epoch_loss:.4f} | Accuracy={accuracy:.4f}\n")
print("\n")

"""
Testing the Model & Evaluation:
"""
classifier.eval()
all_predictions = []
true_labels = []

for i in tqdm(range(len(test_input_ids)), desc="Testing"):
    input_ids = test_input_ids[i].unsqueeze(0).to(device)
    attention_mask = test_attention_masks[i].unsqueeze(0).to(device)
    label = test_labels[i].unsqueeze(0).to(device)

    with torch.no_grad():
        raw_scores = classifier(input_ids=input_ids, attention_mask=attention_mask)

    # exctract predicted and actual labels for all examples
    _, predicted_label = torch.max(raw_scores, 1)
    all_predictions.extend(predicted_label.cpu().numpy())
    true_labels.append(label.item())

accuracy = accuracy_score(true_labels, all_predictions)
print("\nEvaluation: \nAccuracy=", accuracy)
print(classification_report(true_labels, all_predictions, target_names=["non-vul", "vul"]))
