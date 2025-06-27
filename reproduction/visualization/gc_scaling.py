import json
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path('reproduction/out/gc_overhead.jsonl')
gc_stride = 15 # GC was sampled every 15 decoding steps
show_ratio_panel = True 

# Load the JSON
with open(file_path) as f:
    data = json.load(f)

pass_time = data["pass_time"][1:]
gc_time   = data["gc_time"][1:]

# Build the x-axes
x_pass = list(range(len(pass_time)))
x_gc   = [i * gc_stride for i in range(len(gc_time))]

fig, ax_pass = plt.subplots(
    1, 2 if show_ratio_panel else 1,
    figsize=(14 if show_ratio_panel else 7, 4),
    squeeze=False
)
ax_pass = ax_pass[0]  # turn 2-D array into 1-D row

# Panel 0: raw times
ax_pass[0].plot(x_pass, pass_time, label='Pass time', marker='o', ms=4)
ax_pass[0].plot(x_gc  , gc_time  , label='GC time'  , marker='x', ms=5, ls='--')
ax_pass[0].set_xlabel('Decoding step')
ax_pass[0].set_ylabel('Time (s)')
ax_pass[0].set_title('Pass time vs. GC time')
ax_pass[0].grid(alpha=.3)
ax_pass[0].legend()

# Panel 1: GC / Pass ratio (optional)
if show_ratio_panel:
    ratio = [g / pass_time[s] for g, s in zip(gc_time, x_gc)]
    ax_pass[1].stem(
        x_gc, ratio,
        linefmt='tab:red', markerfmt='ro', basefmt='k-',
    )
    ax_pass[1].set_xlabel('Decoding step')
    ax_pass[1].set_ylabel('GC / Pass ratio')
    ax_pass[1].set_title('Relative GC overhead')
    ax_pass[1].set_ylim(bottom=0)
    ax_pass[1].grid(alpha=.3)

out_path = "reproduction/figs/gc_overhead.png"
fig.tight_layout()
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"GC overhead figure saved to {out_path}")

"""
Example usage:
python -m reproduction.visualization.gc_scaling
"""