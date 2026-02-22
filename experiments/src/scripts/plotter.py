import matplotlib.pyplot as plt
import numpy as np

epsilons = np.array([0.0, 0.001, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])

post_reset_kl = np.array([
    1e-2, 1e1, 1.5e1, 1e-2, 1e-3, 2e-3, 1e-6, 1e-6, 1e-6
])

post_reset_js = np.array([
    5e-3, 7e0, 9e0, 5e-3, 7e-4, 1e-3, 1e-6, 1e-6, 1e-6
])

plt.figure(figsize=(6, 4))

# Main curves
plt.plot(epsilons, post_reset_kl, marker="o", label="KL", linewidth=2)
plt.plot(epsilons, post_reset_js, marker="s", linestyle="--", label="JS", linewidth=2)

# 🔥 Dotted threshold barrier
plt.axvline(
    x=0.6,
    linestyle=":",
    linewidth=2,
    color="black",
    alpha=0.8
)

# 🔥 Shaded exact recovery region
plt.axvspan(0.6, 1.0, alpha=0.08)

# Annotation
plt.text(
    0.63,
    3e-6,
    "Exact Recovery Regime",
    fontsize=10,
    alpha=0.85
)

plt.yscale("log")
plt.xlabel("Elimination Rate ε")
plt.ylabel("Mean Post-reset Divergence (KL, JS)")
plt.title("Exact Rollback via Behavioral Elimination")

plt.legend()
plt.tight_layout()

plt.savefig("fig_exact_rollback_2.pdf")
plt.show()