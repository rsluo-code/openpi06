# 预测正确概率 0% 对应的 loss ≈ 27.6310
# 预测正确概率 10% 对应的 loss ≈ 2.3026
# 预测正确概率 20% 对应的 loss ≈ 1.6094
# 预测正确概率 30% 对应的 loss ≈ 1.2040
# 预测正确概率 40% 对应的 loss ≈ 0.9163
# 预测正确概率 50% 对应的 loss ≈ 0.6931
# 预测正确概率 60% 对应的 loss ≈ 0.5108
# 预测正确概率 70% 对应的 loss ≈ 0.3567
# 预测正确概率 80% 对应的 loss ≈ 0.2231
# 预测正确概率 90% 对应的 loss ≈ 0.1054

import numpy as np
import matplotlib.pyplot as plt

# 分类数
num_classes = 201

# loss 范围
losses = np.linspace(0, np.log(num_classes), 1000)
probs = np.exp(-losses)

# 绘图
plt.figure(figsize=(8,5))
plt.plot(losses, probs, label="p_correct = exp(-loss)")
plt.xlabel("Cross-Entropy Loss")
plt.ylabel("Predicted Correct Class Probability")
plt.title(f"201-class Classification: Loss vs Correct Class Probability")
plt.grid(True)
plt.legend()

# 保存到文件
plt.savefig("cross_entropy_vs_probability.png")
plt.close()  # 关闭绘图，释放内存
print("图已保存为 cross_entropy_vs_probability.png")

# 输出特定概率对应的loss
epsilon = 1e-12
for target_prob in [0.1*i for i in range(10)]:
    loss_value = -np.log(max(target_prob, epsilon))
    print(f"预测正确概率 {target_prob*100:.0f}% 对应的 loss ≈ {loss_value:.4f}")