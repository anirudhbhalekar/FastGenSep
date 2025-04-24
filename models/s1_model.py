import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from json import encoder
import logging

import torch
import torch.nn as nn
import itertools

#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from .s1_utils.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from .s1_utils.patch_embed import PatchEmbed_org

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TransformerResidualBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        
        self.transform1 = Block(
            dim=embed_dim, 
            num_heads=8, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer= nn.LayerNorm
        )
        self.relu = nn.ReLU()
        self.transform2 = Block(
            dim=embed_dim, 
            num_heads=8, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer= nn.LayerNorm 
        )
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        x = self.transform1(x)
        x = self.relu(x)
        x = self.transform2(x)
        x = x + identity

        return x


class VitEncoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=1,
                 embed_dim=192, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 audio_exp=False, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, pos_trainable=False, use_nce=False, beta=4.0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, num_speakers = 2, permute = True, temp = 0.07, **kwargs,
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.num_speakers = num_speakers
        self.permute = permute
        self.temp = temp 
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            TransformerResidualBlock(embed_dim)
            for i in range(depth//2)])
        self.norm = norm_layer(embed_dim)
        self.final_linear = nn.Linear(num_patches + 1, self.num_speakers)
        # --------------------------------------------------------------------------
        # Removed decoder
        # --------------------------------------------------------------------------

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch
        # --------------------------------------------------------------------------

        self.avgpool = nn.AvgPool1d(kernel_size=int((num_patches + 1)/num_speakers), stride=int((num_patches + 1)/num_speakers))
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)

        log.debug("Initialised pos embed")
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                #h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
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


    def forward_encoder(self, x):

        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks

        for i, blk in enumerate(self.blocks):
            x = blk(x)
             
        x = self.norm(x)
        
        
        # Permuting and pooling to go from shape (num_batches, num_patches + 1, embed_dim) to (num_batches, num_speaker, embed_dim)
        x = x.permute(0, 2, 1) # (N, L, D) -> (N, D, L) so num patches is last dimension
        x = self.avgpool(x)
        x = x.permute(0, 2, 1) # (N, D, L) -> (N, L, D) 

        return x


    def forward(self, x, mask_ratio=0):
        emb_enc = self.forward_encoder(x)
        # emb_enc: [N, L, D] - a batch of a stack of 1D vectors

        return emb_enc
    

def cl_encoder_2spk(**kwargs): 
    model = VitEncoder(patch_size=16, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_speakers=2, **kwargs)
    return model 

cl_encoder_vit_base_2spk = cl_encoder_2spk
