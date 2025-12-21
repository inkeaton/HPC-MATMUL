import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Measured Hardware Limits ---
# From likwid-bench_gflops_amd.txt
P_peak = 217.98   # GFLOPS
# From likwid-bench_band_amd.txt
B_peak = 58.68    # GB/s

# --- 2. Define Application Performance ---
# From likwid-perfctr logs
App_GFLOPS = 37.65
App_BW = 4.04
App_AI = App_GFLOPS / App_BW  # ~9.32 FLOPs/Byte

# --- 3. Generate Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# X-Axis: Arithmetic Intensity (Log Scale 0.1 to 100)
x = np.logspace(-1, 2, 100)

# Roofline Function: Min(Peak Compute, Peak Bandwidth * Intensity)
y = np.minimum(P_peak, B_peak * x)

# Plot the Rooflines
ax.loglog(x, y, 'k-', linewidth=2, label='Hardware Roofline')
ax.axhline(P_peak, color='r', linestyle='--', label=f'Peak Compute ({P_peak:.0f} GFLOPS)')

# Plot the Bandwidth Slope
# Start from 0.1 up to the ridge point (P_peak / B_peak)
ridge_point = P_peak / B_peak
x_bw = np.array([0.1, ridge_point])
y_bw = B_peak * x_bw
ax.loglog(x_bw, y_bw, 'b--', label=f'Peak Bandwidth ({B_peak:.0f} GB/s)')

# Plot Your Application "Dot"
ax.plot(App_AI, App_GFLOPS, 'ro', markersize=12, label='Your Code (MatMul)', zorder=10)

# Labels and Formatting
ax.set_xlabel('Arithmetic Intensity (FLOPs / Byte)')
ax.set_ylabel('Performance (GFLOPS)')
ax.set_title('Empirical Roofline Model: AMD Ryzen 9 7900X')
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend()
ax.set_xlim(0.1, 100)
ax.set_ylim(1, 1000)

plt.show()