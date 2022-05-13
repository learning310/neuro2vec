import os
import numpy as np
import torch 
from torch import nn
import argparse
from models.tit import TimeTransformer
from misc.dataset import Load_Dataset

home_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--device', default='cpu', type=str,
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
criterion = nn.CrossEntropyLoss()
model = TimeTransformer().to(device)

model_dict = model.state_dict()
pretrained_dict = torch.load(os.path.abspath("./supervised.pt"))['model']
del_list = ['pos_embed', 'mask_token','temporal_pred', ]
pretrained_dict_copy = pretrained_dict.copy()
for i in pretrained_dict_copy.keys():
    for j in del_list:
        if j in i:
            del pretrained_dict[i]
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

test_dataset = torch.load(os.path.join('./data/test.pt'))
test_dataset = Load_Dataset(test_dataset)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

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