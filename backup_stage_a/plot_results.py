
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Recorded mean angular errors from the experiments
modes = ["Mode 1\n(Normal Movement)", "Mode 2\n(Sudden Direction Changes)"]

reactive_errors = [12.16, 12.06]
predictive_errors = [8.44, 8.63]

x = np.arange(len(modes))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 6))

reactive_bars = ax.bar(
    x - width / 2,
    reactive_errors,
    width,
    label="Reactive Tracking"
)

predictive_bars = ax.bar(
    x + width / 2,
    predictive_errors,
    width,
    label="Predictive Tracking"
)

ax.set_title(
    "Reactive vs Predictive Tracking Performance",
    fontsize=15,
    fontweight="bold"
)
ax.set_ylabel("Mean Angular Error (degrees)")
ax.set_xlabel("Movement Condition")
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.bar_label(reactive_bars, fmt="%.2f°", padding=4)
ax.bar_label(predictive_bars, fmt="%.2f°", padding=4)

ax.set_ylim(0, max(reactive_errors) * 1.25)

plt.tight_layout()

# Save the graph in this script's own folder
output_path = Path(__file__).resolve().parent / "mean_angular_error_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Graph saved successfully: {output_path}")

plt.show()
