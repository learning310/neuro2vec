import os
import numpy as np
import torch
import argparse
from models.neuro2vec import neuro2vec
from misc.dataset import Load_Dataset

home_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_description', default='MAE', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='test', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--bs', default=128, type=int,
                    help='Project home directory')
args = parser.parse_args()

####### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

device = torch.device(args.device)
experiment_description = args.experiment_description
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
experiment_log_dir = os.path.join(
    logs_save_dir, experiment_description, run_description, f"seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

train_dataset = torch.load(os.path.join('./data/train.pt'))
train_dataset = Load_Dataset(train_dataset)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)

model = neuro2vec().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-2)

mask_ratio = 0.1
interval = 50
model.train()
for i in range(1, 600+1):
    total_loss = []
    total_acc = []
    for _, (data, label) in enumerate(train_loader):
        data, label = data.float().to(device), label.long().to(device)
        optimizer.zero_grad()
        loss, pred, mask = model(data, mask_ratio)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Training->Epoch:{:0>2d}, Loss:{:.3f}".format(i,torch.tensor(total_loss).mean()))
    if i % interval == 0:
        chkpoint = {'model': model.state_dict()} 
        torch.save(chkpoint, os.path.join(home_dir, str('epoch'+str(i)+'_chkpoint.pt')))