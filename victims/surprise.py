import OpenAttack
import numpy

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch

from utils.data_mappings import SEPARATOR

BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
MAX_BATCHES = -1
pretrained_model = "roberta-base"


def trim(text, tokenizer):
    offsets = tokenizer(text, truncation=True, max_length=MAX_LEN + 10, return_offsets_mapping=True)['offset_mapping']
    limit = len(text)
    if len(offsets) > MAX_LEN:
        limit = offsets[512][1]
    return text[:limit]


def roberta_readfromfile_generator(subset, dir, with_pairs=False, trim_text=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    for line in open(dir / (subset + '.tsv')):
        parts = line.split('\t')
        label = int(parts[0])
        if not with_pairs:
            text = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            if trim_text:
                text = trim(text, tokenizer)
            yield {'fake': label, 'text': text}
        else:
            text1 = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            text2 = parts[3].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            if trim_text:
                text1 = trim(text1, tokenizer)
                text2 = trim(text2, tokenizer)
            yield {'fake': label, 'text1': text1, 'text2': text2}


def eval_loop(model, eval_dataloader, device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(eval_dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    for i, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        # print(logits)
        # a = input()
        pred = torch.argmax(logits, dim=-1).detach().to(torch.device('cpu')).numpy()
        Y = batch["labels"].to(torch.device('cpu')).numpy()
        eq = numpy.equal(Y, pred)
        size += len(eq)
        correct += sum(eq)
        TPs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 1.0)))
        FPs += sum(numpy.logical_and(numpy.equal(Y, 0.0), numpy.equal(pred, 1.0)))
        FNs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 0.0)))
        progress_bar.update(1)
        
        # print(Y)
        # print(pred)
        # a = input()
        
        if i == MAX_BATCHES:
            break
    print('Accuracy: ' + str(correct / size))
    print('F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))
    print(correct, size, TPs, FPs, FNs)
    
    results = {
        'Accuracy': correct / size,
        'F1': 2 * TPs / (2 * TPs + FPs + FNs)
    }
    return results


class VictimRoBERTa(OpenAttack.Classifier):
    def __init__(self, path, task, device=torch.device('cpu')):
        self.device = device
        config = AutoConfig.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.with_pairs = (task == 'FC' or task == 'C19')
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    
    def get_prob(self, input_):
        try:
            probs = None
            # print(len(input_), input_)
            
            batched = [input_[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in
                       range((len(input_) + BATCH_SIZE - 1) // BATCH_SIZE)]
            for batched_input in batched:
                if not self.with_pairs:
                    tokenised = self.tokenizer(batched_input, truncation=True, padding=True, max_length=MAX_LEN,
                                               return_tensors="pt")
                else:
                    parts = [x.split(SEPARATOR) for x in batched_input]
                    tokenised = self.tokenizer([x[0] for x in parts], [(x[1] if len(x) == 2 else '') for x in parts],
                                               truncation=True, padding=True,
                                               max_length=MAX_LEN,
                                               return_tensors="pt")
                with torch.no_grad():
                    tokenised = {k: v.to(self.device) for k, v in tokenised.items()}
                    outputs = self.model(**tokenised)
                probs_here = torch.nn.functional.softmax(outputs.logits, dim=-1).to(torch.device('cpu')).numpy()
                if probs is not None:
                    probs = numpy.concatenate((probs, probs_here))
                else:
                    probs = probs_here
            return probs
        except Exception as e:
            # Used for debugging
            raise
