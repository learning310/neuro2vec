import os
import numpy as np
import torch 
from torch import nn
import argparse
from models.tit import TimeTransformer
from misc.dataset import Load_Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

W = []
N1 = []
N2 = []
N3 = []
R = []
sample = 1000
model.eval()
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.float().to(device), label.long().to(device)
        pred = model(data)
        pred = F.softmax(pred)
        for i in range(len(pred)):
            if label[i] == 0 and len(W) <= sample :
                if torch.argmax(pred[i]) != 0 and pred[i].max() <= 0.6:
                    W.append(pred[i])
            if label[i] == 1 and len(N1) <= sample:
                if torch.argmax(pred[i]) != 1 and pred[i].max() <= 0.6:
                    N1.append(pred[i])
            if label[i] == 2 and len(N2) <= sample:
                if torch.argmax(pred[i]) != 2 and pred[i].max() <= 0.6:
                    N2.append(pred[i])
            if label[i] == 3 and len(N3) <= sample:
                if torch.argmax(pred[i]) != 3 and pred[i].max() <= 0.6:
                    N3.append(pred[i])
            if label[i] == 4 and len(R) <= sample:
                if torch.argmax(pred[i]) != 4 and pred[i].max() <= 0.6:
                    R.append(pred[i])
                    
for i in range(len(W)):
    plt.subplot(5,1,1)
    plt.ylabel("Wake")
    plt.plot(W[i])
for i in range(len(N1)):
    plt.subplot(5,1,2)
    plt.ylabel("N1")
    plt.plot(N1[i])
for i in range(len(N2)):
    plt.subplot(5,1,3)
    plt.ylabel("N2")
    plt.plot(N2[i])
for i in range(len(N3)):
    plt.subplot(5,1,4)
    plt.ylabel("N3")
    plt.plot(N3[i])
for i in range(len(R)):
    plt.subplot(5,1,5)
    plt.ylabel("REM")
    plt.plot(R[i])
plt.show()

# import matplotlib.pyplot as plt
# att_map = torch.load('./map1.pt')
# fig, ax = plt.subplots(2,2)

# ax[0][0].imshow(att_map[0], interpolation='nearest', cmap='Reds')
# ax[0][1].imshow(att_map[1], interpolation='nearest', cmap='Reds')
# ax[1][0].imshow(att_map[2], interpolation='nearest', cmap='Reds')
# ax[1][1].imshow(att_map[3], interpolation='nearest', cmap='Reds')

# plt.show()