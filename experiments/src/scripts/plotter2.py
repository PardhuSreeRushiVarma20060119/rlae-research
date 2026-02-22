import matplotlib.pyplot as plt
import numpy as np

alpha = np.array([0.001, 0.01, 0.05])

# 1.5B
kl_15 = np.array([0.16917, 6.34697, 14.74684])
js_15 = np.array([0.03149, 0.57441, 0.68847])

# 3B
kl_3 = np.array([0.35477, 8.78031, 15.58148])
js_3 = np.array([0.04863, 0.65532, 0.69120])

plt.figure(figsize=(6,4))

# Curves
plt.plot(alpha, kl_15, marker="o", linewidth=2.5, label="KL (1.5B)")
plt.plot(alpha, kl_3, marker="o", linestyle="--", linewidth=2.5, label="KL (3B)")

plt.plot(alpha, js_15, marker="s", linewidth=2.5, label="JS (1.5B)")
plt.plot(alpha, js_3, marker="s", linestyle="--", linewidth=2.5, label="JS (3B)")

# Lighter drift band
plt.axhspan(1e-2, 1e2, alpha=0.04)

# Better text placement (true midpoint on log scale)
plt.text(
    0.02,
    2.0,
    "Irreversible Behavioral Drift",
    fontsize=10,
    color="gray",
    alpha=0.35,
    ha="center",
    va="center"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Weight Mutation Intensity α", fontsize=12)
plt.ylabel("Post-reset Divergence", fontsize=12)

plt.title("Structural Irreversibility under Direct Weight Mutation", fontsize=14)

# Move legend slightly upward/right
plt.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("fig_structural_irreversibility.pdf", dpi=300)
plt.show()