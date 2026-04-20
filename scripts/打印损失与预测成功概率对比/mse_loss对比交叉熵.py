import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 参数设置
# -------------------------------
num_bins = 201
v_min, v_max = -1.0, 0.0
Rt = -0.5  # 假设真实连续值在 [-1,0]

# MSE 放大倍数，可自由修改
mse_scale = 10.0  # 例如 1.0, 5.0, 10.0

# 生成 201 个 bin 对应连续值
bin_values = torch.linspace(v_min, v_max, num_bins)

# -------------------------------
# 绘制曲线：预测成功概率 vs CE loss & MSE loss
# -------------------------------
p_correct_list = np.linspace(0.01, 1.0, 1000)  # 0% ~ 100%，避免 log(0)

ce_losses = []
mse_losses = []

for p in p_correct_list:
    # 交叉熵 CE
    ce_loss = -np.log(p)
    ce_losses.append(ce_loss)
    
    # MSE：期望值 loss
    pred_value = p * Rt + (1 - p) * v_min
    mse_loss = (pred_value - Rt)**2 * mse_scale
    mse_losses.append(mse_loss)

ce_losses = np.array(ce_losses)
mse_losses = np.array(mse_losses)

# -------------------------------
# 绘图
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(p_correct_list*100, ce_losses, label="Cross-Entropy Loss", color='red')
plt.plot(p_correct_list*100, mse_losses, label=f"Expected Value MSE Loss x{mse_scale}", color='blue')
plt.xlabel("Predicted Probability of Correct Bin (%)")
plt.ylabel("Loss")
plt.title(f"201-bin CE vs MSE x{mse_scale} Loss vs Predicted Success Rate (Rt={Rt})")
plt.grid(True)
plt.legend()
plt.savefig("ce_vs_mse_scaled_loss_curve.png")
plt.close()
print(f"曲线图已保存为 ce_vs_mse_scaled_loss_curve.png（MSE放大倍数={mse_scale}）")

# -------------------------------
# 输出每 10% 的预测成功率对应的 CE & MSE loss
# -------------------------------
print("预测成功率\tCE Loss\t\tMSE Loss")
for target_prob in [0.1 * i for i in range(1, 10)]:  # 10% ~ 90%
    ce_loss = -np.log(target_prob)
    pred_value = target_prob * Rt + (1 - target_prob) * v_min
    mse_loss = (pred_value - Rt)**2 * mse_scale
    print(f"{target_prob*100:.0f}%\t\t{ce_loss:.4f}\t\t{mse_loss:.4f}")