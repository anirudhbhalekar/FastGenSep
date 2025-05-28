# 2024 (c) LY Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf

# from sdes.sdes import MixSDE
from model import FastGenSep

import logging 


def str_or_int(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x


def get_model(args):
    if not args.model.exists():
        # assume this is a HF model
        path = hf_hub_download(repo_id=str(args.model), filename="checkpoint.pt")
    else:
        path = args.model

    # load model
    model = FastGenSep.load_from_checkpoint(str(path))

    # transfer to GPU
    model = model.to(args.device)
    model.eval()

    return model


def scale_output(mix, sep):
    # project mix onto separated signal
    num = (mix * sep).sum(dim=-1, keepdim=True)
    denom = (sep * sep + 1e-10).sum(dim=-1, keepdim=True)
    alpha = num / denom
    return alpha * sep


def separate(mix, model: FastGenSep, device, num_steps=32):
    mix = mix.to(device)
    mix = mix[None]  # add batch dim

    # run model
    with torch.no_grad():
        sep = model.separate(mix, num_steps=torch.Tensor([num_steps]).to(device))
    sep = sep.cpu()  # move back to CPU

    return sep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate all the wav files in a specified folder"
    )
    parser.add_argument("--input_dir", type=Path, help="Path to the input folder")
    parser.add_argument("--output_dir", type=Path, help="Path to the output folder")
    parser.add_argument("--max_count", type=int, default=1000, help="Max number of files to process")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to model or Huggingface model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str_or_int,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument("-N", type=int, default=None, help="Number of steps")
    parser.add_argument(
        "--snr", type=float, default=None, help="Step size of corrector"
    )
    parser.add_argument(
        "--corrector-steps", type=int, default=None, help="Number of corrector steps"
    )
    parser.add_argument(
        "--denoise", type=bool, default=True, help="Use denoising in solver"
    )
    parser.add_argument(
        "-s", "--schedule", type=str, help="Pick a different schedule for the inference"
    )
    parser.add_argument(
        "--num_steps", type=int, default=32, help="Number of steps for separation"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"
        print("No CUDA, fall back to CPU")

    model = get_model(args)
    model_sr = model.config.model.fs

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif args.output_dir.is_file():
        raise ValueError("Output directory is a file")

    count = 0
    for wavpath in tqdm(args.input_dir.glob("*.wav"), desc="Separating wav files"):
        if count >= args.max_count:
            break
        count += 1
        waveform, sr = torchaudio.load(wavpath)

        if sr != model_sr:
            print(
                f"Skipping {wavpath.stem} due to mismatched sample rate. "
                f"This model expects {model_sr} Hz, but the file is {sr} Hz."
            )
        sep = separate(waveform, model, args.device, num_steps=args.num_steps)
        for i in range(sep.shape[1]):
            spkr_dir = args.output_dir / f"s{i}"
            spkr_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                spkr_dir / f"{wavpath.stem}.wav", sep[:, i, :], sr, format="wav"
            )

    print(f"Processed {count} files. Output saved to {args.output_dir}.")
    print("Done.")