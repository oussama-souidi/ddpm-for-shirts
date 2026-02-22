# Conditional DDPM for Fashion Image Generation

This project implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** in PyTorch for generating 128×128 fashion product images.

## Features
- Custom implementation of forward and reverse diffusion processes
- Conditional U-Net with residual blocks
- Sinusoidal time embeddings
- Class conditioning via learned embeddings
- GPU-safe training with registered buffers

## Architecture
- Multi-scale U-Net encoder–decoder
- Residual blocks + GroupNorm
- 1000-step linear beta schedule

## Tech Stack
- Python
- PyTorch
- CUDA (GPU acceleration)

## Usage
Train and sample images using the configurable `Config` class.  
Pretrained weights can be loaded using:

```python
ddpm.load_state_dict(torch.load("ddpm_fashion.pth"))
