import pathlib, torch, sys

import victims.transformer
import victims.bilstm
from victims.transformer import PRETRAINED_BERT, PRETRAINED_GEMMA_2B, PRETRAINED_GEMMA_7B

# task = 'HN'
# victim = 'BiLSTM'
# data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
# out_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim + '-512.pth')

task = sys.argv[1]
victim = sys.argv[2]
data_path = pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])

pc_device = 'cpu'  # if victim == 'BiLSTM' else 'mps'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(pc_device)

if victim == 'BERT':
    victims.transformer.train_and_save(data_path, out_path, device, task, PRETRAINED_BERT, using_peft=False,
                                       skip_visual=False)
elif victim == 'GEMMA':
    victims.transformer.train_and_save(data_path, out_path, device, task, PRETRAINED_GEMMA_2B, using_peft=True,
                                       skip_visual=False)
elif victim == 'GEMMA7B':
    victims.transformer.train_and_save(data_path, out_path, device, task, PRETRAINED_GEMMA_7B, using_peft=True,
                                       skip_visual=False)
elif victim == 'BiLSTM':
    victims.bilstm.train_and_save(data_path, out_path, device, task, skip_visual=False)
