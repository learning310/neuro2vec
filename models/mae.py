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
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c l -> b l c')
        )
    
    def forward(self, x):
        return self.proj(x)

class MaskedAutoencoder(nn.Module):
    def __init__(self, signal_length=3000, patch_size=30, in_chans=1, embed_dim=128, 
                depth=4, num_heads=4, mlp_ratio=4, dropout=0.1, 
                decoder_embed_dim=64, decoder_depth=2):
        super().__init__()

        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False) 
        self.blocks = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) 
        # --------------------------------------------------------------------------

        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
    
    def patchify(self, series):
        x = series.reshape(shape=(series.shape[0], self.num_patches, -1))
        return x
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence 
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # # apply Transformer blocks
        x = self.decoder_blocks(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, x, pred, mask):
        target = self.patchify(x)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio):
        # mean = torch.mean(x, dim=-1)
        # mean = mean.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # std = torch.std(x, dim=-1)
        # std = std.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # x = (x - mean) / std

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

       

'''
NOTE:
1. 位置编码1000的选取
2. 预测生成的是做完卷积之后的patch, 还是指定patch位置的原始信号
3. 关于MSA的heads
4. dropout是feedforward中的部分
5. 如何应用在下游任务
'''