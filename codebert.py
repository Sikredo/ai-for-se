import json

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch

java_per_line = [
    {
        "code": "public SecureRandom getObject() throws Exception {",
        "vulnerable": False
    },
    {
        "code": "SecureRandom rnd = SecureRandom.getInstance(algorithm);",
        "vulnerable": True
    },
    {
        "code": "return rnd;",
        "vulnerable": False
    },
    {
        "code": "}",
        "vulnerable": False
    }
]
class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(java_line, tokenizer):
    # source
    code_tokens = tokenizer.tokenize(java_line["code"], padding="max_length", max_length=256)
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = -1 - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, java_line['vulnerable'])


class TextDataset(Dataset):
    def __init__(self, tokenizer):
        self.examples = []
        for line in java_per_line:
            self.examples.append(convert_examples_to_features(line, tokenizer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# BertTokenizer.from_pretrained('bert-base-uncased') ---- Could also be used
#model = AutoModel.from_pretrained("microsoft/codebert-base")
nl_tokens=tokenizer.tokenize("return maximum value")

dataset = TextDataset(tokenizer)
print("  Num examples = %d", len(dataset))
for x in range(4):
    print(dataset.__getitem__(x))


#tensored_data = []
#for line in java_per_line:
    #code_tokens_per_line=tokenizer.tokenize(line["code"], padding='max_length')
    #print(code_tokens_per_line)

    #tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens_per_line+[tokenizer.eos_token]

    #tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    #labels = torch.tensor([1]).unsqueeze(0)
    #print(labels)
    #input_feature = convert_examples_to_features(line, tokenizer)
   # tensor = torch.tensor(input_feature)

   # context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
 #  print(context_embeddings)
#print(tensored_data)

#InputFeatures with assigned Label
