import matplotlib.pyplot as plt

# Validation error 数值
errors = [
    0.022804,
    0.0274614,
    0.029708,
    0.0286907,
    0.04349307,
    0.0354931,
    0.0609125
]

# 对应的 prompt（用英文/拼音，避免服务器字体问题）
prompts = [
    "banana_seen",
    "kiwi_seen",
    "water_seen",
    "tissue_seen",
    "bowl_seen",
    "blindbox_seen",
    "plush_unseen"
]

# 创建图像
plt.figure(figsize=(8, 4))
plt.plot(prompts, errors, marker='o')

# 坐标轴与标题
plt.xlabel("Prompt")
plt.ylabel("Validation Error")
plt.title("Validation Error per Prompt")

# 旋转 x 轴标签，防止重叠
plt.xticks(rotation=30)

# 自适应布局
plt.tight_layout()

# 保存图片（服务器环境）
output_path = "validation_error_curve.png"
plt.savefig(output_path)

# 关闭图像，释放资源
plt.close()

print(f"Saved figure to: {output_path}")
