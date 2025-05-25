#2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import datetime
import itertools
import json
import logging
import torchaudio
import math
import os
from collections import defaultdict
from pathlib import Path
import itertools

import fast_bss_eval
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf.omegaconf import open_dict
from scipy.optimize import linear_sum_assignment
from torch_ema import ExponentialMovingAverage
from models.score_models import ScoreModelNCSNpp
from torch import autograd

from prodict import Prodict
import utils

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def normalize_batch(batch):
    mix, tgt = batch[0], batch[1]
    mean = mix.mean(dim=(1, 2), keepdim=True)
    std = mix.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    mix = (mix - mean) / std
    if tgt is not None:
        tgt = (tgt - mean) / std
    return (mix, tgt), mean, std

def denormalize_batch(x, mean, std):
    return x * std + mean


class FastGenSep(pl.LightningModule):
    def __init__(self, config):
        # init superclass
        super().__init__()

        self.save_hyperparameters()

        # the config and all hyperparameters are saved by hydra to the experiment dir
        self.config = config
        self.config = Prodict.from_dict(self.config)
        
        os.environ["HYDRA_FULL_ERROR"] = "1"
        self.model = instantiate(self.config.model.main_model, _recursive_=False)

        self.sm_config = self.config.datamodule.shortcut_target_args

        self.valid_max_sep_batches = getattr(
            self.config.model, "valid_max_sep_batches", 1
        )
        
        # for moving average of weights
        self.ema_decay = getattr(self.config.model, "ema_decay", 0.0)
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.normalize_batch = normalize_batch
        self.denormalize_batch = denormalize_batch

        self.lr_warmup = getattr(config.model, "lr_warmup", None)
        self.lr_original = self.config.model.optimizer.lr

        self.loss = instantiate(self.config.model.loss)
        self.val_losses = {}
        for name, loss_args in self.config.model.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)


        self.si_sdr_loss = self.val_losses["val/si_sdr"]

        self.num_spks = self.config.model.n_speakers
        self.num_denoising_steps = int(self.config.datamodule.shortcut_target_args.denoise_timesteps)

    def on_train_epoch_start(self):
        pass
    
    def compute_pit_loss(self, v_target, v_pred):
        """
        Compute the PIT loss between the target and the prediction
        """

        # v_target is of shape [num_spks, timesteps]
        # v_pred is of shape [num_spks, timesteps]

        minloss = torch.tensor(float('inf')).to(v_target.device) 
        for perm in itertools.permutations(range(v_target.shape[0])):
            perm = list(perm)
            v_target_perm = v_target[perm]
            # want to preserve batch losses
            loss = torch.nn.MSELoss(reduction='none')(v_target_perm, v_pred).mean(dim=(-1))
            minloss = torch.min(minloss, loss)

        return minloss

    def compute_sm_loss(self, v_target, v_pred, t):
        """
        Compute the loss function in two parts:
        - the consistency loss
        - the flow matching loss

        each with permuation invariance

        This is intrinsically done by get targets function, so we do 
        PIT MSE loss here.

        """

        # We want to do PIT ONLY when t=0. Otherwise just do MSE loss    
        # v_target is of shape [batch, num_spks, timesteps]
        # v_pred is of shape [batch, num_spks, timesteps]

        mse_loss = torch.nn.MSELoss(reduction='none')(v_target, v_pred).mean(dim=(-1))

        pit_indices = (t == 0).nonzero(as_tuple=True)[0]
        if len(pit_indices) > 0:
            pit_loss = torch.stack([self.compute_pit_loss(v_target[i], v_pred[i]) for i in pit_indices])
            mse_loss[pit_indices] = pit_loss
        
        mean_loss = mse_loss.mean()
        return mean_loss
 
    def compute_separation_loss(self, target, estimate, title = None): 
        """
        target is of shape [batch, n_spks, time]
        estimate is of shape [batch, n_spks, time]

        We compute the permutation invariant SI-SDR and PESQ loss between the target and the estimate 
        on the num_spks axis.
        """
       
        # Compute the SI-SDR
        si_sdr_loss = self.si_sdr_loss(target, estimate)
        self.log(title, si_sdr_loss, on_epoch=True, sync_dist=True)

        return si_sdr_loss
        

    def do_lr_warmup(self):
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original

    def separate(self, mix, num_steps): 
        """
        Separation procedure with jump dt the sources from the mixture
        """

        # mix is of shape [batch, 1, time]
        # num_steps is of shape [batch, 1]

        # num_steps is in the range [1, 2, 4, 8, 16, 32]

        batch, *stats = self.normalize_batch((mix, None))
        mix = batch[0]
        
        x_t = torch.randn_like(mix).to(mix.device)
        x_t = torch.cat([x_t] * self.num_spks, dim=1)
        
        # We will condition on log(1/dt)

        dt_base = torch.log2(num_steps).to(mix.device)

        # We will run the model for num_denoising_steps 

        dt = 1/num_steps
        dt = dt.to(mix.device)
        t = torch.zeros_like(dt).to(mix.device)

        with torch.no_grad():
            while t < 1: 
                v_pred = self(x_t, mix, t, dt_base)
                x_t = x_t + dt[:, None, None] * v_pred
                x_t = torch.clip(x_t, -4, 4)
                t = t + dt
        
        x_t = self.denormalize_batch(x_t, *stats)

        return x_t
    
    def training_step(self, batch, batch_idx):
        """
        Need to rework this, keeping in mind that score models will handle get targets now
        """
        loss = None
        batch, *stats = self.normalize_batch(batch)
        mix, target = batch

        # mix is of shape [batch, 1, time]
        # target is of shape [batch, num_spks, time]

        # get the targets
        x_t, v_t, t, dt = self.get_targets(self.sm_config, mix, target)
        v_pred = self(x_t, mix, t, dt) # [batch, 2, timesteps]

        # compute mse loss of v_pred and v_t
        loss = self.compute_sm_loss(v_pred, v_t, t)

        # every 10 steps, we log stuff
        cur_step = self.trainer.global_step
        self.last_step = getattr(self, "last_step", 0)
        if cur_step > self.last_step and cur_step % 10 == 0:
            self.last_step = cur_step
            log.info(f"Training step {batch_idx} loss {loss}")
            # log the classification metrics
            self.logger.log_metrics(
                {"train_sm_loss": loss},
                step=cur_step,
            )

        self.do_lr_warmup()
        return loss
        
    
    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0

    def validation_step(self, batch, batch_idx, dataset_i=0):
        """
        Validation FM+SC loss 
        Then reconstruct seperation and computer 1,2,4,8,16,32 step SI-SDRs

        """
        torch.cuda.empty_cache() # Clear the cache to avoid memory issues
        
        batch, *stats = self.normalize_batch(batch)
        mix, target = batch

        # mix is of shape [batch, 1, time]
        # target is of shape [batch, num_spks, time]

        t, dt = torch.Tensor([0]*mix.shape[0]).to(mix.device), torch.Tensor([self.num_denoising_steps]*mix.shape[0]).to(mix.device)
        x_0 = torch.concat([mix] * self.num_spks, dim=1)
        v_pred = self(x_0, mix, t, dt)
        v_t = target - x_0

        # compute mse loss of v_pred and v_t
        loss = self.compute_sm_loss(v_pred, v_t, t)
        self.log("val_sm_loss", loss, on_epoch=True, sync_dist=True)

        steps_arr = torch.Tensor([1, 2, 4, 8, 16, 32]).to(mix.device)
        # validation separation losses
        if self.trainer.testing or self.n_batches_est_done < self.valid_max_sep_batches:
            self.n_batches_est_done += 1
            for num_steps in steps_arr:
                num_steps = num_steps.unsqueeze(0)
                est = self.separate(mix, num_steps)
                if num_steps == 32: 
                    self.compute_separation_loss(target, est, title=f"val/si_sdr")
                else: 
                    self.compute_separation_loss(target, est, title=f"val/si_sdr_{num_steps}")
        

    def on_validation_epoch_end(self, outputs = None):
        pass

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.on_validation_epoch_end(outputs)

    def configure_optimizers(self):
        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.config.model.optimizer}")

        opt_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(
            {**{"params": opt_params}, **self.config.model.optimizer}
        )

        if getattr(self.config.model, "scheduler", None) is not None:
            scheduler = instantiate(
                {**self.config.model.scheduler, **{"optimizer": optimizer}}
            )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.config.model.grad_clipper)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.model.main_val_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            log.warn("EMA state_dict not found in checkpoint!")


    def train(self, mode=True, no_ema=False): 
        
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                self.ema.store(self.parameters())  # store current params in EMA
                self.ema.copy_to(
                    self.parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.parameters()
                    )  # restore the EMA weights (if stored)
    
        return res
    
    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def forward(self, xt, mix, t, dt):
        return self.model(xt, mix, t, dt)
    
    # Load the config for flow-based target generation in time domain
    @torch.no_grad()
    def get_targets(self, cfg, mix, tgts, force_t=-1, force_dt=-1): 
        # ensure that BOTH mix and tgt are of shape [batch_size, num_spks, timesteps]
        
        # So mix here is [batch_size, 1, timesteps]
        # tgts is [batch_size, num_spks, timesteps]

        x_0 = torch.concat([mix] * tgts.shape[1], dim=1).to(mix.device)
        batch_size = mix.shape[0]
        device = mix.device 

        # Since we are GPU poor, we cannot have a high batch size
        # We will sample a uniform rv, and if it is less than 1/bootstrap_every, we will generate bootstrap targets only 
        # Otherwise, we will generate flow-matching targets only

        u = torch.rand(1).item()
        p_0 = torch.rand(1).item()
        
        t_init = False
        if p_0 < cfg.p_0: 
            t_init = True 
            

        bootstrap_batchsize = 0
        if u < 1/int(cfg.bootstrap_every):
            bootstrap_batchsize = batch_size

            # 1) =========== Sample dt. ============

            log2_sections = int(np.log2(int(cfg.denoise_timesteps)))
          
            dt_base = torch.randint(low = 0, high = log2_sections, size=(bootstrap_batchsize,)).to(device)

            #force_dt_vec = (torch.ones(bootstrap_batchsize, dtype=torch.float32) * force_dt).to(device)
            #dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
            dt = 1 / (2 ** (dt_base)).to(device) # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
            dt_base_bootstrap = (dt_base + 1).to(device)
            dt_bootstrap = dt / 2


            # 2) =========== Sample t. ============

            dt_sections = torch.pow(2, dt_base).to(device) # [1, 2, 4, 8, 16, 32]
            t = torch.randint(low = 0, high = int(dt_sections), size=(bootstrap_batchsize,)).to(device)
            if t_init: 
                t = torch.zeros_like(t).to(device)
            t = t / dt_sections # Between 0 and 1.
    
            #force_t_vec = (torch.ones(bootstrap_batchsize, dtype=torch.float32) * force_t).to(device)
            #t = torch.where(force_t_vec != -1, force_t_vec, t)
            t_full = t[:, None, None].to(device)

            # 3) =========== Generate Bootstrap Targets ============

            # tgts will be of shape (batch_size, num_spks, stft_bins, timesteps)
            # mix will be of shape (batch_size, num_spks, stft_bins, timesteps)

            # x_0 takes the value of mix, x_1 takes the value of tgts and x_t interpolates between the two
            # No injection of Gaussian noise - though this could be added in the future if needed

            # we want flow matching targets after this step

            x_1 = tgts[:bootstrap_batchsize].to(device)
            x_0 = torch.randn_like(x_1).to(device) # Random initialisation

            x_t = ((1 - (1 - cfg.sigma_min)*t_full) * x_0 + t_full * x_1).to(device)

            # Enforce bootstrap targets (consistency metric)
            # We want to run x_t through the model itself to get the velocity

            self.model.eval()
            v_b1 = self(x_t, mix, t, dt_base_bootstrap)
            t2 = (t + dt_bootstrap).to(device)
            x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
            x_t2 = torch.clip(x_t2, -4, 4) # For stablity (since we work with normalised data)
            v_b2 = self(x_t2, mix, t2, dt_base_bootstrap)
           
            v_target = (v_b1 + v_b2) / 2
            v_target = torch.clip(v_target, -4, 4) # For stablity (since we work with normalised data)

            self.logger.log_metrics(
                {'v_target_magnitude': torch.sqrt(torch.mean(v_target**2)).item()
                },
                step=self.trainer.global_step
            )
            self.model.train()

            return x_t, v_target, t, dt_base
        

        # 4) =========== Generate Flow-Matching Targets ============

        # Sample t.
        t = torch.randint(low = 0, high=int(cfg.denoise_timesteps), size=(mix.shape[0],)).to(device)
        if t_init: 
            t = torch.zeros_like(t).to(device)

        t = t.float() / int(cfg.denoise_timesteps)
        t = t.to(device)

        #force_t_vec = (torch.ones(mix.shape[0], dtype=torch.float32)* force_t).to(device) 
        #t = torch.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
        t_full = t[:, None, None] # [batch, 1, 1, 1]

        # Sample flow pairs x_t, v_t.
        x_1 = tgts[:batch_size - bootstrap_batchsize].to(device)
        x_0 = torch.randn_like(x_1).to(device) # Random initialisation

        x_t = ((1 - (1 - cfg.sigma_min)*t_full) * x_0 + t_full * x_1).to(device)
        v_t = (x_1 - x_0).to(device)   
        dt_flow = np.log2(cfg.denoise_timesteps)
        dt_base = torch.ones(mix.shape[0], dtype=torch.int32) * dt_flow

        # 5) ==== Merge Flow+Bootstrap ====

        x_t = x_t.to(device)
        v_t = v_t.to(device)
        t = t.to(device)
        dt_base = dt_base.to(device)

        self.logger.log_metrics(
            {'v_target_magnitude': torch.sqrt(torch.mean(v_t**2)).item()            
            },
            step=self.trainer.global_step
        )

        return x_t, v_t, t, dt_base
