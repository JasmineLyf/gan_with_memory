# config.py

import os
import torch

data_root = "./data/mnist"
image_save_dir = "./images"
model_save_dir = "./checkpoints"

os.makedirs(data_root, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)


class Config:
    # Training
    batch_size = 64
    n_epochs = 30
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999

    # Image
    img_size = 32
    channels = 1

    # Latent space
    latent_dim = 100

    # MemoryBank
    mem_key_dim = 128
    mem_val_dim = 128
    mem_capacity = 512

    # GNN
    gnn_hidden_dim = 128
    gnn_k = 5

    # Others
    sample_interval = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
