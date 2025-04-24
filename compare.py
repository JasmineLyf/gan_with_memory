from utils.visualize_results import load_losses, plot_losses
g, d = load_losses("loss_log.json")
g_base, d_base = load_losses("loss_log_baseline.json")
plot_losses(g, d)  # 增强版
plot_losses(g_base, d_base)  # 原始版
