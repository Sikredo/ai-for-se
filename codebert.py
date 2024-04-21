from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from DataLoader import JsonDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

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


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 context_embeddings,
                 label,
                 ):
        self.context_embeddings = context_embeddings
        self.label = label


def convert_line_to_features(java_line, tokenizer, model):
    code_tokens = tokenizer.tokenize(java_line["code"])
    tokens=[tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    #print("Context Embeddings")
    #print(context_embeddings)
    #print(context_embeddings.shape)
    context_embeddings_with_equal_length = torch.mean(context_embeddings, dim=1)
    #print("After applying mean to get same length:")
    #print(context_embeddings_with_equal_length)
    #print(context_embeddings_with_equal_length.shape)
    return InputFeatures(context_embeddings_with_equal_length, java_line['vulnerable'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, model):
        self.examples = []
        self.vul_lines = 0
        for vulnerability in vulnerabilities: #TODO: Think if a double foreach is needed or if we can get rid of the vulnerability level
            for java_line in vulnerability.get("vulData"):
                if java_line["vulnerable"]:
                    self.vul_lines += 1
                self.examples.append(convert_line_to_features(java_line, tokenizer, model))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].context_embeddings, self.examples[i].label


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# BertTokenizer.from_pretrained('bert-base-uncased') ---- Could also be used
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

print("Started Feature Engineering...")
dataset = TextDataset(tokenizer, model)
print(f"  Num examples(lines) = {len(dataset)}")
print(f"    - Vulnerable lines = {dataset.vul_lines}")
print(f"    - NonVulnerable lines = {len(dataset) - dataset.vul_lines}")
print("Finished Feature Engineering...")


def split_into_train_and_test(total_data):
    X = [] # 2D array where each row is a feature vector for one line
    y = [] # 1D array where each label belongs to one feature vector

    # store feature vectors and labels in arrays
    for i in range(len(total_data)):
        feature_vector, label = total_data[i]
        X.append(feature_vector.detach().numpy())
        y.append(int(label))

    X = np.array(X)
    y = np.array(y)

    # split into training and test set: 80/20 and randomly -> TODO: later use K-fold Cross Validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # change random_state for different seed

    X_train = np.squeeze(X_train, axis=1)
    X_test = np.squeeze(X_test, axis=1)

    # balance the training set
    randomUnderSampler = RandomUnderSampler(random_state=0)
    X_train_balanced, y_train_balanced = randomUnderSampler.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced, X_test, y_test


print("Preparing training and test set...")
X_train_balanced, y_train_balanced, X_test, y_test = split_into_train_and_test(dataset)

print("Training classifier...")
classifier = RandomForestClassifier(random_state=0, n_estimators=100) #change random_state for different seed; n_estimators is amount of trees
classifier.fit(X_train_balanced, y_train_balanced)

print("Testing classifier...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Result Metrics:")
print("Accuracy: ", accuracy)
print(classification_report(y_test, y_pred, target_names=["non-vulnerable", "vulnerable"]))

