from model import FastGenSep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import hydra
from data import LibriMix_Module
from omegaconf import OmegaConf
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging
import matplotlib.pyplot as plt
# Want to get the dataloader and then visualise a couple of examples of x_0, x_1, x_t 
# Create dataloaders for the training and validation sets
# Load the model

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(config_path="/research/milsrg1/user_workspace/ab2810/IIB_Proj/SM_GenSep/config", config_name="config")
def main(cfg):
    dm = LibriMix_Module(cfg)
    model = FastGenSep(cfg)

    log.info("Model loaded")
    log.info("Data loaded")
    num_speakers = 2
    # Get the dataloaders
    dm.setup()
    train_loader = dm.train_dataloader()

    log.info("Dataloaders loaded")

    # Get a random batch of data
    for batch in train_loader:
        
        batch, *stats = model.normalize_batch(batch)
        mix, target = batch
        mix = torch.concat([mix]*target.shape[1], dim=1)
        
        # at a random point we break the loop
        if torch.rand(1) < 0.1:
            break
    
    ############################################
    mix = torch.rand_like(target)
    ############################################
    
    log.info("Batch loaded")
    # Get the get_targets_function from the model

    t_arr = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to(mix.device)
    x_arr = torch.tensor([])
    x_arr = torch.concat([x_arr, mix], dim=0)
    for t in t_arr: 
        t = torch.Tensor([t]).to(mix.device).unsqueeze(0).unsqueeze(0)
        x_t = (torch.ones_like(t)-t)*mix + t*target
        x_arr = torch.concat([x_arr, x_t], dim=0)
   
    x_arr = torch.concat([x_arr, target], dim=0)

    # Now print shapes of x_arr
    print("Shape of x_arr")
    print(x_arr.shape)

    # We want to reduce the 1st dimension from 4 to 2, so we will square and add 0th and 1st elements, and 2nd and 3rd elements
   
    # Save this image as a dumped numpy array
    print("Saving numpy array")
    np.save("/research/milsrg1/user_workspace/ab2810/IIB_Proj/SM_GenSep/visualize.npy", x_arr.cpu().detach().numpy())
    print("Numpy array saved")
    """
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(5):
        axs[0, i].imshow(imgs[0, i].cpu().detach().numpy())
        axs[1, i].imshow(imgs[1, i].cpu().detach().numpy())
    """
   
   

if __name__ == "__main__":
    main()