import pathlib

import torch

import victims.bert as bert
import victims.bilstm as bilstm

task = 'RD'
victim = 'BERT'

data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
out_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim + '-512.pth')

pc_device = 'cpu' if victim == 'BiLSTM' else 'mps'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(pc_device)

if victim == 'BiLSTM':
    model = bilstm.VictimBiLSTM(out_path, task, device).model
    with_pairs = (task == 'FC')
    _, eval_dataloader = bert.prepare_dataloaders_training(data_path, with_pairs=with_pairs, just_codes=True)
    bilstm.eval_loop(eval_dataloader, model, device, skip_visual=False)
elif victim == 'BERT':
    model = bert.VictimBERT(out_path, task, device).model
    with_pairs = (task == 'FC')
    _, eval_dataloader = bert.prepare_dataloaders_training(data_path, with_pairs=with_pairs)
    bert.eval_loop(model, eval_dataloader, device, skip_visual=False)
