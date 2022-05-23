import torch
import os
from models.neuro2vec import neuro2vec
import matplotlib.pyplot as plt

masking_ratio = 0.5
patch_size = 30

def run_one_epoch(signal, model):
    signal = signal.squeeze()
    mean = torch.mean(signal)
    std = torch.std(signal)
    signal = (signal - mean) / std

    # make it a batch-like
    x = signal.unsqueeze(dim=0)

    # run MAE
    loss, y, mask, fourier_pred = model(x, masking_ratio)

    # visual
    plt.subplot(4, 1, 1)
    plt.xticks([])
    plt.plot(signal)
    plt.ylabel("Original")

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size)
    mask = mask.reshape(-1)
    im_masked = signal * (1-mask)
    plt.subplot(4, 1, 2)
    plt.xticks([])
    plt.plot(im_masked)
    plt.ylabel("Masked")
    
    y = y.reshape((-1)).detach()
    y = y*(1)
    plt.subplot(4, 1, 3)
    plt.xticks([])
    plt.plot(y)
    plt.ylabel("Temmporal_Reconstruction")

    fourier_pred = fourier_pred.squeeze().detach()
    fourier_pred = fourier_pred*(1)
    plt.subplot(4, 1, 4)
    plt.xticks([])
    plt.plot(fourier_pred)
    plt.ylabel("Fourier_Reconstruction")

    plt.show()

device = torch.device('cpu')
model = neuro2vec().to(device)
chkpoint = torch.load(os.path.abspath("./epoch200_chkpoint.pt"))
model.load_state_dict(chkpoint['model'])

test_dataset = torch.load(os.path.join('./data/test.pt'))['samples']
run_one_epoch(test_dataset[1], model)
