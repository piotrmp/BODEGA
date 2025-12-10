import shutil

import OpenAttack
import huggingface_hub
import torch
import numpy

from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from utils.data_mappings import SEPARATOR

MAX_LEN = 512
EPOCHS = 5
MAX_BATCHES = -1
access_token = 'HF_ACCESS_TOKEN_HERE'
PRETRAINED_BERT = "bert-base-uncased"
PRETRAINED_GEMMA_2B = "google/gemma-2b"
PRETRAINED_GEMMA_7B = "google/gemma-7b"
batch_size = {PRETRAINED_BERT: 16, PRETRAINED_GEMMA_2B: 4, PRETRAINED_GEMMA_7B: 4}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def trim(text, tokenizer):
    offsets = tokenizer(text, truncation=True, max_length=MAX_LEN + 10, return_offsets_mapping=True)['offset_mapping']
    limit = len(text)
    if len(offsets) > MAX_LEN:
        limit = offsets[512][1]
    return text[:limit]


def readfromfile_generator(subset, dir, pretrained_model, with_pairs=False, trim_text=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
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


def prepare_dataloaders_training(dir, pretrained_model, with_pairs=False, just_codes=False, tokenizer=None):
    train_dataset = Dataset.from_generator(readfromfile_generator,
                                           gen_kwargs={'subset': 'train', 'dir': dir,
                                                       'pretrained_model': pretrained_model, 'with_pairs': with_pairs})
    test_dataset = Dataset.from_generator(readfromfile_generator,
                                          gen_kwargs={'subset': 'attack', 'dir': dir,
                                                      'pretrained_model': pretrained_model, 'with_pairs': with_pairs})
    dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
    
    accepts_token_type_ids = (pretrained_model == PRETRAINED_BERT)
    
    def tokenize_function(example):
        result = tokenizer(example["text"], truncation=True, max_length=MAX_LEN,
                           return_token_type_ids=accepts_token_type_ids and not just_codes,
                           return_attention_mask=not just_codes)
        return result
    
    def tokenize_function_pairs(example):
        result = tokenizer(example["text1"], example["text2"], truncation=True, max_length=MAX_LEN,
                           return_token_type_ids=accepts_token_type_ids and not just_codes,
                           return_attention_mask=not just_codes)
        return result
    
    if not with_pairs:
        tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    else:
        tokenized_datasets = dataset_dict.map(tokenize_function_pairs, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text1", "text2"])
    tokenized_datasets = tokenized_datasets.rename_column("fake", "labels")
    tokenized_datasets.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size[pretrained_model], collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=batch_size[pretrained_model], collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader


def eval_loop(model, eval_dataloader, device, skip_visual=False, print_path=None):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(eval_dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    all_preds = []
    for i, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).detach().to(torch.device('cpu')).numpy()
        Y = batch["labels"].to(torch.device('cpu')).numpy()
        if print_path:
            all_preds.extend(pred.tolist())
        eq = numpy.equal(Y, pred)
        size += len(eq)
        correct += sum(eq)
        TPs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 1.0)))
        FPs += sum(numpy.logical_and(numpy.equal(Y, 0.0), numpy.equal(pred, 1.0)))
        FNs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 0.0)))
        progress_bar.update(1)
        if i == MAX_BATCHES:
            break
    print('Accuracy: ' + str(correct / size))
    print('F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))
    if print_path:
        with open(print_path, 'w') as f:
            for pred in all_preds:
                f.write(str(pred)+'\n')


def train_loop(model, train_dataloader, device, optimizer, lr_scheduler, skip_visual=False):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(train_dataloader)), ascii=True, disable=skip_visual)
    losses = []
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.detach().to(torch.device('cpu')).numpy())
        progress_bar.update(1)
        if i == MAX_BATCHES:
            break
    print('Train loss: ' + str(numpy.mean(losses)))


def train_and_save(data_path, out_path, device, task, pretrained_model, using_peft, skip_visual):
    with_pairs = (task == 'FC' or task == 'C19')
    train_dataloader, eval_dataloader = prepare_dataloaders_training(data_path, pretrained_model, with_pairs=with_pairs)
    
    if using_peft:
        print("Adding PEFT through LoRA configuration...")
        lora_config = LoraConfig(r=8, inference_mode=False,
                                 target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
                                                 "down_proj"], task_type=TaskType.SEQ_CLS, )
        if device.type == 'cuda':
            print('GPU available, reducing model precision for QLoRA... ')
            quantization_config = bnb_config
        else:
            print('No GPU available, using full precision (expect high memory usage)...')
            quantization_config = None
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                                   quantization_config=quantization_config,
                                                                   token=access_token, num_labels=2)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, token=access_token, num_labels=2)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    model.to(device)
    
    eval_loop(model, eval_dataloader, device, skip_visual)
    for epoch in range(EPOCHS):
        print("EPOCH " + str(epoch + 1))
        train_loop(model, train_dataloader, device, optimizer, lr_scheduler, skip_visual)
        eval_loop(model, eval_dataloader, device, skip_visual)
    
    model.to(torch.device('cpu'))
    
    if not using_peft:
        torch.save(model.state_dict(), out_path)
    else:
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir()
        huggingface_hub.login(token=access_token)
        model.save_pretrained(out_path)
        huggingface_hub.logout()


class VictimTransformer(OpenAttack.Classifier):
    def __init__(self, path, task, pretrained_model, using_peft, device):
        self.device = device
        if not using_peft:
            config = AutoConfig.from_pretrained(pretrained_model, token=access_token)
            self.model = AutoModelForSequenceClassification.from_config(config)
            self.model.load_state_dict(torch.load(path), strict=False)
        else:
            if device.type == 'cuda':
                print('GPU available, loading low-precision model trained with QLoRA... ')
                quantization_config = bnb_config
            else:
                print(
                    'No GPU available, loading low-precision model using full precision (expect slightly different results)...')
                quantization_config = None
            config = PeftConfig.from_pretrained(path)
            assert config.base_model_name_or_path == pretrained_model
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                                            quantization_config=quantization_config,
                                                                            token=access_token)
            self.model = PeftModel.from_pretrained(self.model, path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
        self.with_pairs = (task == 'FC' or task == 'C19')
        self.batch_size = batch_size[pretrained_model]
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    
    def get_prob(self, input_):
        try:
            probs = None
            batched = [input_[i * self.batch_size:(i + 1) * self.batch_size] for i in
                       range((len(input_) + self.batch_size - 1) // self.batch_size)]
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
