# 2023 (c) LINE Corporation
# MIT License
import copy

import torch
import torchaudio
import logging
from hydra.utils import instantiate
from functools import partial
from .s1_model import VitEncoder 

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

"Want to integrate S1 results into this"

class ScoreModelNCSNpp(torch.nn.Module):
    def __init__(
        self,
        num_sources,
        stft_args,
        backbone_args,
        s1_model_args,
        fs, 
        s1_checkpoint = None,
        transform="exponent",
        spec_abs_exponent=0.5,
        spec_factor=3.0,
        spec_trans_learnable=False,
        **kwargs
    ):
        super().__init__()

        # infer input output channels of backbone from number of sources
        backbone_args.update(
            num_channels_in=2 * num_sources + 2, num_channels_out=2 * num_sources
        )

        s1_model_args.update(
            in_chans=2 * num_sources,
        )

        log.debug("Instantiating backbone...")
        self.backbone = instantiate(backbone_args, recursive=False)
        log.debug("Backbone instantiated.")

        self.stft_args = stft_args
        self.fs = fs
        self.stft = torchaudio.transforms.Spectrogram(power=None, **stft_args)
        self.stft_inv = torchaudio.transforms.InverseSpectrogram(**stft_args)

        #self.s1_model = instantiate(s1_model_args, recursive=False)

        self.s1_target_length = s1_model_args["img_size"][0]
        self.s1_num_freq_bins = s1_model_args["img_size"][1]

        if s1_checkpoint is not None:
            self.s1_model.load_state_dict(torch.load(s1_checkpoint))
         
        self.transform = transform
        self.spec_abs_exponent = spec_abs_exponent
        self.spec_factor = spec_factor
        if spec_trans_learnable:
            self.spec_abs_exponent = torch.nn.Parameter(
                torch.tensor(self.spec_abs_exponent)
            )
            self.spec_factor = torch.nn.Parameter(torch.tensor(spec_factor))

    def transform_forward(self, spec):
        if self.transform == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = abs(self.spec_abs_exponent)
                spec = spec.abs() ** abs(e) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform == "log":
            spec = torch.log1p(spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * abs(self.spec_factor)
        elif self.transform == "none":
            spec = spec
        else:
            raise ValueError("transform must be one of 'exponent'|'log'|'none'")

        return spec

    def transform_backward(self, spec):
        if self.transform == "exponent":
            spec = spec / abs(self.spec_factor)
            if self.spec_abs_exponent != 1:
                e = abs(self.spec_abs_exponent)
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform == "log":
            spec = spec / abs(self.spec_factor)
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform == "none":
            spec = spec
        return spec

    def complex_to_real(self, x):
        # x: (batch, chan, freq, frames)
        x = torch.stack((x.real, x.imag), dim=1)  # (batch, 2, chan, freq, frames)
        x = x.flatten(start_dim=1, end_dim=2)  # (batch, 2 * chan, freq, frames)
        return x

    def real_to_complex(self, x):
        x = x.reshape((x.shape[0], 2, -1) + x.shape[2:])
        x = torch.view_as_complex(x.moveaxis(1, -1).contiguous())
        return x

    def pad(self, x):
        n_frames = x.shape[-1]
        rem = n_frames % 64
        if rem == 0:
            return x, 0
        else:
            pad = 64 - rem
            x = torch.nn.functional.pad(x, (0, pad))
            return x, pad

    def unpad(self, x, pad):
        if pad == 0:
            return x
        else:
            return x[..., :-pad]

    def adjust_length(self, x, n_samples):
        if x.shape[-1] < n_samples:
            return torch.nn.functional.pad(x, (0, n_samples - x.shape[-1]))
        elif x.shape[-1] > n_samples:
            return x[..., :n_samples]
        else:
            return x

    def pre_process(self, x):
        n_samples = x.shape[-1]
        x = torch.nn.functional.pad(
            x, (0, self.stft_args["n_fft"] - self.stft_args["hop_length"])
        )
        x = self.stft(x)
        x = self.transform_forward(x)
        x = self.complex_to_real(x)
        x, n_pad = self.pad(x)
        return x, n_samples, n_pad

    def post_process(self, x, n_samples, n_pad):
        x = self.unpad(x, n_pad)
        x = self.real_to_complex(x)
        x = self.transform_backward(x)
        x = self.stft_inv(x)
        x = self.adjust_length(x, n_samples)
        return x
    
    def s1_pre_process(self, stft_img):
        """
        stft_img shape is [N, C, F, T] 
        
        return shape is [N, C, F, self.s1_target_length]
        """

        T = stft_img.shape[-1]
        p = self.s1_target_length

        result_img = stft_img

        if T < p:
            result_img = torch.nn.functional.pad(stft_img, pad=(0, p-T, 0, 0, 0, 0, 0, 0), mode='constant', value=0)

        elif T > p:
            rand_start = torch.randint(low=0, high=T-p, size=(1,))[0]
            result_img = stft_img[..., rand_start:rand_start+p]
        
        return result_img

    
    def forward(self, xt, mix, time_cond, dt):
        """
        Args:
            x: (batch, channels, time)
            mix: (batch, channels, time)    
            time_cond: (batch,)
        Returns:
            x: (batch, channels, time) same size as input
        """
        
        x = torch.cat((xt, mix), dim=1)
        x, n_samples, n_pad = self.pre_process(x)
        
        s1_output = None
        #s1_input = self.s1_pre_process(x)
        #s1_output = self.s1_model(s1_input)

        x = self.backbone(x, time_cond, dt, s1_output)
        x = self.post_process(x, n_samples, n_pad)
        return x
