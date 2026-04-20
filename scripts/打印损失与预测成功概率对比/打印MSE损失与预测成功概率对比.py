import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 参数设置
# -------------------------------
num_bins = 201
v_min, v_max = -1.0, 0.0
Rt = -0.5  # 假设真实连续值在 [-1,0]

# 生成 201 个 bin 对应连续值
bin_values = torch.linspace(v_min, v_max, num_bins)  # 201-bin 对应值

# -------------------------------
# 绘制曲线：预测成功概率 vs MSE loss
# -------------------------------
# 假设预测期望值 = p_correct * Rt + (1 - p_correct) * v_min
p_correct_list = np.linspace(0, 1, 1000)  # 0% ~ 100%
losses = []

for p in p_correct_list:
    pred_value = p * Rt + (1 - p) * v_min
    loss = (pred_value - Rt)**2
    losses.append(loss)

losses = np.array(losses)

# 绘图
plt.figure(figsize=(8,5))
plt.plot(p_correct_list*100, losses, label="MSE Loss")
plt.xlabel("Predicted Probability of Correct Bin (%)")
plt.ylabel("MSE Loss")
plt.title(f"201-bin Expected Value MSE Loss vs Predicted Success Rate (Rt={Rt})")
plt.grid(True)
plt.legend()
plt.savefig("mse_loss_vs_prob_curve.png")
plt.close()
print("曲线图已保存为 mse_loss_vs_prob_curve.png")

# -------------------------------
# 输出每 10% 的预测成功率对应的 loss
# -------------------------------
print("预测成功率\tMSE Loss")
for target_prob in [0.1 * i for i in range(1, 10)]:  # 10% ~ 90%
    pred_value = target_prob * Rt + (1 - target_prob) * v_min
    loss = (pred_value - Rt)**2
    print(f"{target_prob*100:.0f}%\t\t{loss:.4f}")