import numpy as np
import matplotlib.pyplot as plt

# Define given statistics
average = 3.7597
std_dev = 0.0253
max_deviation = 0.0889

# Generate a synthetic dataset with 25 data points around the average with the given std deviation
np.random.seed(0)  # For reproducibility
data = np.clip(np.random.normal(average, std_dev, 25), average - max_deviation, average + max_deviation)

# Set up the figure with broken y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
fig.subplots_adjust(hspace=0.1)  # Adjust space between plots

# Plot data on both axes
ax1.boxplot(data, patch_artist=True, showfliers=False)
ax2.boxplot(data, patch_artist=True, showfliers=False)

# Set y-axis limits for the "cut" effect
ax1.set_ylim(3.5, 4.0)  # Show range from 3.5mm to 4.0mm
ax2.set_ylim(0, 0.5)    # Show range from 0mm to 0.5mm

# Hide the spines between the broken axes
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Create a wavy line to indicate the break
def draw_wave(ax, position, amplitude=0.02, frequency=3):
    """Draw a sinusoidal line on the specified axis."""
    x = np.linspace(-amplitude, 1 + amplitude, 500)
    y = np.sin(frequency * np.pi * x) * amplitude + position
    ax.plot(x, y, transform=ax.transAxes, color='k', clip_on=False)

# Draw waves on both the top of the lower plot and the bottom of the upper plot
draw_wave(ax1, position=0, amplitude=0.02, frequency=3)
draw_wave(ax2, position=1, amplitude=0.02, frequency=3)

# Set y-axis label in the middle
fig.suptitle("Pressing Displacement for BaRiFlex", fontname="Times New Roman")
fig.text(0.04, 0.5, "Pressing Displacement (mm)", va='center', rotation='vertical', fontname="Times New Roman")

# Plot average, std deviation, and max deviation as bold, smaller points in Times New Roman font
ax1.text(1.1, 3.80, f"Average: {average:.4f}mm", fontsize=8, fontweight='bold', fontname="Times New Roman", verticalalignment='center')
ax1.text(1.1, 3.77, f"Std Deviation: {std_dev:.4f}mm", fontsize=8, fontweight='bold', fontname="Times New Roman", verticalalignment='center')
ax1.text(1.1, 3.74, f"Max Deviation: {max_deviation:.4f}mm", fontsize=8, fontweight='bold', fontname="Times New Roman", verticalalignment='center')

# Customize x-axis to display "BaRiFlex" without numbers or label
ax2.set_xticks([1])                # Position the label in the center
ax2.set_xticklabels(["BaRiFlex"], fontname="Times New Roman")  # Label the tick as "BaRiFlex"
ax1.xaxis.set_visible(False)       # Hide x-axis on the upper subplot

# Display the plot
plt.show()