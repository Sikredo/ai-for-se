import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from CDataLoader import CDataLoader

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

vulnerabilities = list()
dataloader = CDataLoader("./bigvul-data/data.json")
vulnerabilities_ = dataloader.get_prepared_data()
#vulnerabilities.extend(vulnerabilities_[0:2500])
vulnerabilities.extend(vulnerabilities_[-9000:])
training, test = get_training_and_test_data_per_function(vulnerabilities)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

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


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state #last hidden state for getting embeddings for all tokens

            #get mean embeddings for each line
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = attention_mask_expanded.sum(1)
            mean_embeddings = sum_embeddings / sum_mask

            embeddings.append(mean_embeddings.cpu())
            labels.append(label.cpu())

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    return embeddings, labels

# Create the dataset
train_dataset = VulnerabilityDataset(train_input_ids, train_attention_masks, train_labels)
# Compute class weights
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
sample_weights = [class_weights[label] for label in train_labels.numpy()]
# Create sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset,sampler=sampler, batch_size=16)
train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
# Save embeddings and labels
torch.save(train_embeddings, 'train_embeddings.pt')
torch.save(train_labels, 'train_labels.pt')
#
# Repeat the process for the test set
test_dataset = VulnerabilityDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)
test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
torch.save(test_embeddings, 'test_embeddings.pt')
torch.save(test_labels, 'test_labels.pt')