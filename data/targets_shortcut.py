import numpy as np
import torch 
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

"""
config additions: 

- bootstrap_every: int
- denoise_timesteps: int
- bootstrap_dt_bias: float
- sigma_min: float

"""

# Load the config for flow-based target generation
def get_targets(cfg, model, mix, tgts, force_t=-1, force_dt=-1): 
    # ensure that BOTH mix and tgt are of shape [batch_size, num_spks, stft_bins, timesteps]
    
    batch_size = mix.shape[0]
    device = mix.device


    # Since we are GPU poor, we cannot have a high batch size
    # We will sample a uniform rv, and if it is less than 1/bootstrap_every, we will generate bootstrap targets only 
    # Otherwise, we will generate flow-matching targets only

    u = torch.rand(1).item()
    bootstrap_batchsize = 0
    if u < 1/int(cfg.bootstrap_every):
        log.debug("Generating bootstrap targets")
        bootstrap_batchsize = batch_size

        # 1) =========== Sample dt. ============

        log2_sections = int(np.log2(int(cfg.denoise_timesteps)))

        if cfg.bootstrap_dt_bias == 0: 
            dt_base = torch.Tensor.repeat(int(log2_sections) - 1 - torch.arange(log2_sections), bootstrap_batchsize // log2_sections).to(device)
            dt_base = torch.concatenate([dt_base, torch.zeros(bootstrap_batchsize-dt_base.shape[0],).to(device)]).to(device)

        else: 
            dt_base = torch.Tensor.repeat(log2_sections - 1 - torch.arange(log2_sections-2), (bootstrap_batchsize // 2) // log2_sections).to(device)
            dt_base = torch.concatenate([dt_base, torch.ones(bootstrap_batchsize // 4), torch.zeros(bootstrap_batchsize // 4)]).to(device)
            dt_base = torch.concatenate([dt_base, torch.zeros(bootstrap_batchsize-dt_base.shape[0],)]).to(device)

        force_dt_vec = (torch.ones(bootstrap_batchsize, dtype=torch.float32) * force_dt).to(device)
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
        dt = 1 / (2 ** (dt_base)).to(device) # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = (dt_base + 1).to(device)
        dt_bootstrap = dt / 2

        # 2) =========== Sample t. ============

        dt_sections = torch.pow(2, dt_base).to(device) # [1, 2, 4, 8, 16, 32]
        t = torch.randint(low = 0, high = int(dt_sections), size=(bootstrap_batchsize,)).to(device)
        t = t / dt_sections # Between 0 and 1.
        force_t_vec = (torch.ones(bootstrap_batchsize, dtype=torch.float32) * force_t).to(device)
        t = torch.where(force_t_vec != -1, force_t_vec, t)
        t_full = t[:, None, None, None].to(device)

        # 3) =========== Generate Bootstrap Targets ============

        # tgts will be of shape (batch_size, num_spks, stft_bins, timesteps)
        # mix will be of shape (batch_size, num_spks, stft_bins, timesteps)

        # x_0 takes the value of mix, x_1 takes the value of tgts and x_t interpolates between the two
        # No injection of Gaussian noise - though this could be added in the future if needed

        # we want flow matching targets after this step

        x_1 = tgts[:bootstrap_batchsize].to(device)
        x_0 = mix[:bootstrap_batchsize].to(device)

        x_t = ((1 - (1 - cfg.sigma_min)*t_full) * x_0 + t_full * x_1).to(device)

        # Enforce bootstrap targets (consistency metric)
        v_b1 = model(x_t, t, dt_base_bootstrap)
        t2 = (t + dt_bootstrap).to(device)
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1

        v_b2 = model(x_t2, t2, dt_base_bootstrap)
        v_target = (v_b1 + v_b2) / 2

        return x_t, v_target, t, dt_base
    

    # 4) =========== Generate Flow-Matching Targets ============

    # Sample t.
    t = torch.randint(low = 0, high=int(cfg.denoise_timesteps), size=(mix.shape[0],)).to(device)
    t = t.float() / int(cfg.denoise_timesteps)
    t = t.to(device)

    force_t_vec = (torch.ones(mix.shape[0], dtype=torch.float32)* force_t).to(device) 
    t = torch.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]

    # Sample flow pairs x_t, v_t.
    x_0 = mix[:batch_size - bootstrap_batchsize].to(device)
    x_1 = tgts[:batch_size - bootstrap_batchsize].to(device)

    x_t = ((1 - (1 - cfg.sigma_min)*t_full) * x_0 + t_full * x_1).to(device)
    v_t = (x_1 - (1 - cfg.sigma_min) * x_0).to(device)   
    dt_flow = np.log2(cfg.denoise_timesteps)
    dt_base = torch.ones(mix.shape[0], dtype=torch.int32) * dt_flow


    # 5) ==== Merge Flow+Bootstrap ====

    x_t = x_t.to(device)
    v_t = v_t.to(device)
    t = t.to(device)
    dt_base = dt_base.to(device)

    return x_t, v_t, t, dt_base


def plot_xts(x_t, t): 
    # x_t is of shape [batch_size, num_spks, stft_bins, timesteps]   
    x_t = x_t.detach().cpu().numpy()

    for i in range(x_t.shape[0]):
        plt.figure()
        plt.imshow(x_t[i, 0, :, :], label='x_t1')
        plt.imshow(x_t[i, 0, :, :], label='x_t2')
        plt.legend()
        plt.title(f"t: {t[i]}")
        plt.show()
    
    return None

