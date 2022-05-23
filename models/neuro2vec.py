import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from models.transformer import Transformer
from misc.utils import get_1d_sincos_pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )
    
    def forward(self, x):
        return self.proj(x)

class neuro2vec(nn.Module):
    def __init__(self, signal_length=3000, patch_size=30, in_chans=1, embed_dim=128, 
                depth=4, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size

        # encoder specifics
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # heads
        self.temporal_pred = nn.Linear(embed_dim, patch_size, bias=True)
        self.amplitude_pred = nn.Linear(embed_dim, 1500 // self.num_patches)
        self.phase_pred = nn.Linear(embed_dim, 1500 // self.num_patches)
       
        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, inputs, mask_ratio):
        # standardization (per-epoch)
        mean = torch.mean(inputs, dim=-1)
        mean = mean.unsqueeze(-1).repeat(1, 1, inputs.shape[-1])
        std = torch.std(inputs, dim=-1)
        std = std.unsqueeze(-1).repeat(1, 1, inputs.shape[-1])
        inputs = (inputs - mean) / std

        # embed patches
        x = self.patch_embed(inputs)

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1) 
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)

        # temporal_loss 
        tempral_pred = self.temporal_pred(x)

        target = inputs.reshape((inputs.shape[0], self.num_patches, -1))
        temporal_loss = (tempral_pred - target) ** 2
        temporal_loss = temporal_loss.mean()

        # amplitude_loss
        amplitude_pred = self.amplitude_pred(x)
        phase_pred = self.phase_pred(x)

        amplitude = amplitude_pred.reshape((inputs.shape[0], -1))
        phase = phase_pred.reshape((inputs.shape[0], -1))
        img = torch.tensor(1j, device=x.device)
        l_fourier = amplitude + img*phase 
        r_fourier = torch.flip(amplitude[:, 1:], [-1]) + (-img)*torch.flip(phase[:, 1:], [-1])
        r_fourier = torch.concat(((amplitude[:, 0] + (-img)*phase[:, 0]).unsqueeze(1), r_fourier), dim=-1)
        fourier = torch.concat((l_fourier, r_fourier), dim=-1)
        fourier_pred = torch.fft.ifft(fourier, dim=-1, norm='forward').real
        target = inputs.reshape((inputs.shape[0], -1))
        fourier_loss = (fourier_pred - target) ** 2
        fourier_loss = fourier_loss.mean()
        
        # total loss
        loss = temporal_loss + fourier_loss

        return loss, tempral_pred, mask, fourier_pred
