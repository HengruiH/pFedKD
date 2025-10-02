from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset

class AG_news(Dataset):
    def __init__(self, split, max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        dataset = load_dataset("ag_news", split=split)
        self.text = dataset['text']
        self.targets = dataset['label']
        self.max_length = max_length
        # tokenize(self.text,self.tokenizer)
        # self.tokenize(dataset['text'],self.tokenizer)

    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                            max_length=self.max_length, padding="max_length", truncation=True)
        return (inputs['input_ids'][0], inputs['attention_mask'][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.tokenize(self.text[index])
        target = self.targets[index]
        return input, target



dataset = AG_news(split="train")

print(len(dataset))

sample = dataset[119999][0][0]
len(sample)