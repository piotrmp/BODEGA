import OpenAttack
import numpy
import torch
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss
from torch.optim import Adam
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.data_mappings import SEPARATOR
from victims.bert import prepare_dataloaders_training, MAX_LEN

MAX_BATCHES = -1
BATCH_SIZE = 32
EPOCHS = 10
TOKENISER_MODEL = "bert-base-uncased"


def train_loop(dataloader, model, optimizer, device, skip_visual=False):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    losses = []
    for i, XY in enumerate(dataloader):
        X = XY['input_ids'].to(device)
        Y = XY['labels'].to(device)
        pred = model(X)
        loss = model.compute_loss(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().to(torch.device('cpu')).numpy())
        progress_bar.update(1)
        if i == MAX_BATCHES:
            break
    print('Train loss: ' + str(numpy.mean(losses)))


def eval_loop(dataloader, model, device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    with torch.no_grad():
        for i, XY in enumerate(dataloader):
            X = XY['input_ids'].to(device)
            Y = XY['labels']
            pred = model.postprocessing(model(X))
            Y = Y.to(torch.device('cpu')).numpy()
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


class BiLSTM(Module):
    
    def __init__(self, tokenizer):
        super(BiLSTM, self).__init__()
        self.embedding_layer = Embedding(len(tokenizer.vocab), 32, padding_idx=tokenizer.pad_token_id)
        self.lstm_layer = LSTM(input_size=self.embedding_layer.embedding_dim, hidden_size=128,
                               batch_first=True,
                               bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, 2)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
    
    def forward(self, x):
        embedded = self.embedding_layer(x)
        _, (hidden_state, _) = self.lstm_layer(embedded)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        scores = self.linear_layer(reshaped)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output
    
    @staticmethod
    def postprocessing(Y):
        decisions = Y.argmax(1).to(torch.device('cpu')).numpy()
        return decisions


def train_and_save(data_path, out_path, device, task, skip_visual=False):
    with_pairs = (task == 'FC' or task == 'C19')
    train_dataloader, eval_dataloader = prepare_dataloaders_training(data_path, with_pairs=with_pairs, just_codes=True)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENISER_MODEL)
    model = BiLSTM(tokenizer)
    
    print("Preparing training")
    model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    eval_loop(eval_dataloader, model, device, skip_visual)
    for epoch in range(EPOCHS):
        print("EPOCH " + str(epoch + 1))
        train_loop(train_dataloader, model, optimizer, device, skip_visual)
        eval_loop(eval_dataloader, model, device, skip_visual)
    
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), out_path)


class VictimBiLSTM(OpenAttack.Classifier):
    def __init__(self, path, task, device=torch.device('cpu')):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENISER_MODEL)
        self.model = BiLSTM(self.tokenizer)
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        self.model.eval()
        self.with_pairs = (task == 'FC' or task == 'C19')
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    
    def get_prob(self, input_):
        try:
            if not self.with_pairs:
                tokenised = self.tokenizer(input_, truncation=True, padding=True, max_length=MAX_LEN,
                                           return_token_type_ids=False,
                                           return_attention_mask=False,
                                           return_tensors="pt")
            else:
                parts = [x.split(SEPARATOR) for x in input_]
                tokenised = self.tokenizer([x[0] for x in parts], [(x[1] if len(x) == 2 else '') for x in parts],
                                           truncation=True, padding=True,
                                           max_length=MAX_LEN, return_token_type_ids=False,
                                           return_attention_mask=False,
                                           return_tensors="pt")
            with torch.no_grad():
                X = tokenised['input_ids'].to(self.device)
                logprobs = self.model(X).to(torch.device('cpu')).detach().numpy()
            probs = numpy.exp(logprobs)
            return probs
        except Exception as e:
            # Used for debugging
            raise
