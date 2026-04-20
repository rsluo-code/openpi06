import numpy as np
import matplotlib.pyplot as plt

# X-axis labels (English)
labels = [
    "Banana (seen)",
    "Kiwi (seen)",
    "Mineral Water (seen)",
    "Tissue (seen)",
    "Bowl (seen)",
    "Blind Box (seen)",
    "Plush Toy (unseen)"
]

# Validation one error
one_error = np.array([
    0.03437583,
    0.03132854,
    0.04670927,
    0.02883125,
    0.03921515,
    0.03472927,
    0.05811696
])

# Validation zero error
zero_error = np.array([
    0.03531723,
    0.03338658,
    0.04038435,
    0.02853605,
    0.0371604,
    0.029609,
    0.05664988
])

x = np.arange(len(labels))

plt.figure(figsize=(10, 5))
plt.plot(x, one_error, marker='o', linewidth=2, label='Validation One Error')
plt.plot(x, zero_error, marker='s', linewidth=2, label='Validation Zero Error')

plt.xticks(x, labels, rotation=30, ha='right')
plt.xlabel("Item Category")
plt.ylabel("Error")
plt.title("Validation Error Comparison (Seen vs Unseen)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("validation_error.png", dpi=300)
plt.close()

print("Saved figure as validation_error.png")
