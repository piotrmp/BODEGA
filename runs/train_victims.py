import pathlib, torch, sys

import victims.bert
import victims.bilstm

#task = 'HN'
#victim = 'BiLSTM'
#data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
#out_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim + '-512.pth')

task = sys.argv[1]
victim = sys.argv[2]
data_path= pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])

pc_device = 'cpu'# if victim == 'BiLSTM' else 'mps'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(pc_device)

if victim == 'BERT':
    victims.bert.train_and_save(data_path, out_path, device, task, skip_visual=False)
elif victim == 'BiLSTM':
    victims.bilstm.train_and_save(data_path, out_path, device, task, skip_visual=False)
