import matplotlib.pyplot as plt
import json


def plot_losses(g_losses, d_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved loss plot to {save_path}")


# 可选：如果将 loss 存为 JSON 或 CSV，可以用以下函数读取

def load_losses(json_file):
    with open(json_file, 'r') as f:
        losses = json.load(f)
    return losses['G'], losses['D']


g_loss, d_loss = load_losses("../loss_log.json")
plot_losses(g_loss, d_loss)