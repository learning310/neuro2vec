import torch
import torch.nn as nn
from models.transformer import Transformer
from misc.utils import get_1d_sincos_pos_embed
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )
    
    def forward(self, x):
        return self.proj(x)


class FreqEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.length = 200
        self.nfft = 256
        self.proj = nn.Linear(self.nfft//2+1, 128)

    def forward(self, x):
        with torch.no_grad():
            x = x.reshape(x.shape[0], x.shape[-1])
            win = torch.hamming_window(self.length).to(x.device)
            x = torch.stft(x, self.nfft, self.length//2, self.length, win, return_complex=True)
            x = torch.abs(x)
            x = torch.tensor(20, device=x.device) * torch.log10(x)
            x = torch.permute(x, (0, 2, 1))
        x = self.proj(x)
        return x


class TimeTransformer(nn.Module):
    def __init__(self, signal_length=3000, patch_size=30, pool='cls', in_chans=1, embed_dim=128,
                depth=4, num_heads=4, mlp_ratio=4, dropout=0.2, num_classes=5):
        super().__init__()

        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.pool == 'cls':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches), cls_token=True if self.pool == 'cls' else False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token
        if self.pool == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.dropout(x)
        x = self.blocks(x)

        # using token feature as classification head or avgerage pooling for all feature
        x = self.avg_pool(x.permute((0, 2, 1))).squeeze() if self.pool == 'mean' else x[:, 0]

        # classify
        pred = self.classifier(x)

        return pred
