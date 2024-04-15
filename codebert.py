from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from DataLoader import JsonDataLoader

dataloader = JsonDataLoader("./data/original_method.json")
vulnerabilities = dataloader.get_prepared_data()


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
        for vulnerability in vulnerabilities: #TODO: Think if a double foreach is needed or if we can get rid of the vulnerability level
            for java_line in vulnerability.get("vulData"):
                self.examples.append(convert_line_to_features(java_line, tokenizer, model))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].context_embeddings, torch.tensor(self.examples[i].label)


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# BertTokenizer.from_pretrained('bert-base-uncased') ---- Could also be used
model = AutoModel.from_pretrained("microsoft/codebert-base")

dataset = TextDataset(tokenizer, model)
print("  Num examples = %d", len(dataset))
for x in range(4):
    print(dataset.__getitem__(x))


# TODO:
# Split Dataset into Test and Training (balanced!)
# Train Model -> e.g. using RandomForestClassifier from scikit-learn