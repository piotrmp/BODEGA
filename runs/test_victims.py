import pathlib
import sys
import torch

import victims.transformer as transformer
import victims.surprise as surprise
import victims.bilstm as bilstm
from victims.transformer import PRETRAINED_BERT, PRETRAINED_GEMMA_2B, PRETRAINED_GEMMA_7B

task = sys.argv[1]
victim = sys.argv[2]
print_path = None
if len(sys.argv) > 3:
    print_path = sys.argv[3]

data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
out_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim + '-512.pth')

pc_device = 'cpu'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(pc_device)

if victim == 'BiLSTM':
    model = bilstm.VictimBiLSTM(out_path, task, device).model
    print("Parameters: "+str(sum(p.numel() for p in model.parameters())))
    with_pairs = (task == 'FC' or task == 'C19')
    _, eval_dataloader = transformer.prepare_dataloaders_training(data_path, PRETRAINED_BERT, with_pairs=with_pairs, just_codes=True)
    bilstm.eval_loop(eval_dataloader, model, device, skip_visual=False, print_path=print_path)
elif victim == 'BERT':
    model = transformer.VictimTransformer(out_path, task, PRETRAINED_BERT, False, device).model
    with_pairs = (task == 'FC' or task == 'C19')
    _, eval_dataloader = transformer.prepare_dataloaders_training(data_path, PRETRAINED_BERT, with_pairs=with_pairs)
    transformer.eval_loop(model, eval_dataloader, device, skip_visual=False, print_path=print_path)
elif victim == 'GEMMA':
    model = transformer.VictimTransformer(out_path, task, PRETRAINED_GEMMA_2B, True, device).model
    with_pairs = (task == 'FC' or task == 'C19')
    _, eval_dataloader = transformer.prepare_dataloaders_training(data_path, PRETRAINED_GEMMA_2B, with_pairs=with_pairs)
    transformer.eval_loop(model, eval_dataloader, device, skip_visual=False, print_path=print_path)
elif victim == 'GEMMA7B':
    model = transformer.VictimTransformer(out_path, task, PRETRAINED_GEMMA_7B, True, device).model
    with_pairs = (task == 'FC' or task == 'C19')
    _, eval_dataloader = transformer.prepare_dataloaders_training(data_path, PRETRAINED_GEMMA_7B, with_pairs=with_pairs)
    transformer.eval_loop(model, eval_dataloader, device, skip_visual=False, print_path=print_path)
elif victim == 'surprise':
    model = surprise.VictimRoBERTa(out_path, task, device).model
    with_pairs = (task == 'FC' or task == 'C19')
    _, eval_dataloader = transformer.prepare_dataloaders_training(data_path, PRETRAINED_GEMMA_7B, with_pairs=with_pairs)
    transformer.eval_loop(model, eval_dataloader, device, skip_visual=False)
