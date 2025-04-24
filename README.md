# 🔬 WGAN-GP with Memory, GNN, GRU Discriminator

This project explores **Wasserstein GAN with Gradient Penalty (WGAN-GP)** under various discriminator designs, focusing on **memory-enhanced, graph-structured, and recurrent discriminators**, and compares them to a baseline DCGAN-style WGAN-GP.

## 📌 Features

- 🧠 **Modular Discriminator** with:
  - GRU Memory Module
  - MemoryBank for high-level semantic memory
  - GNN-based structural embedding
- 📊 **WGAN-GP loss** with spectral norm and gradient penalty
- 🧪 **Ablation Control**: Toggle GRU, MemoryBank, GNN via CLI
- 🛡️ **Baseline Discriminator** for structure comparison
- 📈 **Early Stopping** based on Wasserstein gap
- 🎨 Output: MNIST real/fake scores and visual samples
- 📁 Organized results & checkpoints auto-saving

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train full model (GRU + Memory + GNN)

```bash
python train_wgan_gp.py --use_gru --use_memory --use_gnn
```

### 3. Train baseline WGAN-GP

```bash
python train_baseline.py
```

### 4. Run inference

```bash
python inference_wgan.py         # for full model
python inference_baseline.py    # for baseline
```

## 📂 Directory Structure

```
.
├── model/
│   ├── generator.py
│   ├── wgan_gp_discriminator.py     # with GRU + GNN + Memory
│   ├── baseline_discriminator.py
│   ├── memory_bank.py, gnn_encoder.py
├── train_wgan_gp.py
├── train_baseline.py
├── inference_wgan.py
├── inference_baseline.py
├── checkpoints/
├── images_wgan/
├── images_baseline/
└── requirements.txt
```

## 🤪 Results & Evaluation

You can visually and numerically compare:

- Real vs Fake image scores
- Generator diversity
- Wasserstein gap
- Visual quality collapse vs success

## 📜 License

MIT License

