
import matplotlib.pyplot as plt
from pathlib import Path

# Recorded experimental improvements
modes = [
    "Mode 1\n(Normal Movement)",
    "Mode 2\n(Sudden Direction Changes)"
]

error_reduction = [30.6, 28.4]

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(modes, error_reduction)

ax.set_title(
    "Predictive Tracking: Angular Error Reduction",
    fontsize=15,
    fontweight="bold"
)
ax.set_ylabel("Mean Angular Error Reduction (%)")
ax.set_xlabel("Movement Condition")
ax.set_ylim(0, 40)

ax.bar_label(bars, fmt="%.1f%%", padding=5)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent
    / "predictive_error_reduction.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Graph saved successfully: {output_path}")

plt.show()
