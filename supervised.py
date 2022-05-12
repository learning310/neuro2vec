import os
import numpy as np
import torch 
import torch.nn as nn
import argparse
from models.tit import TimeTransformer
from misc.dataset import Load_Dataset
from misc.metrics import accuracy_metric, _calc_metrics

home_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_description', default='TiT', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='withPE20', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
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
    dataset=train_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
model = TimeTransformer().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-2)
for i in range(40):
    total_loss = []
    total_acc = []
    model.train()
    for _, (data, label) in enumerate(train_loader):
        data, label = data.float().to(device), label.long().to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, label)
        total_loss.append(loss.item())
        total_acc.append(accuracy_metric(pred, label))
        loss.backward()
        optimizer.step()
    print("Training->Epoch:{:0>2d}, Loss:{:.3f}, Acc:{:.3f}.".format(i,
        torch.tensor(total_loss).mean(), torch.tensor(total_acc).mean()))

test_dataset = torch.load(os.path.join('./data/test.pt'))
test_dataset = Load_Dataset(test_dataset)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

model.eval()
with torch.no_grad():
    total_loss = []
    outs = np.array([])
    trgs = np.array([])
    for data, label in test_loader:
        data, label = data.float().to(device), label.long().to(device)
        pred= model(data)
        loss = criterion(pred, label)
        total_loss.append(loss.item())
        pred = pred.max(1, keepdim=True)[1]  
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, label.data.cpu().numpy())
    _calc_metrics(outs, trgs, experiment_log_dir, args.home_path)