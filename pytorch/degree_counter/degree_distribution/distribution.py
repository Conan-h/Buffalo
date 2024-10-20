# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.ticker import ScalarFormatter
# import pickle
# import math
# # Data for the full batch
# degrees = np.arange(1, 11)  # Equivalent to MATLAB 1:10
# full_batch_nodes = [13428, 11706, 9277, 7320, 6222, 4868, 4045, 3472, 2976, 27627]

# # Data for micro-batch 0
# micro_batch_0_nodes = [6768, 6015, 4981, 4022, 3397, 2719, 2296, 1987, 1669, 12610]

# # Data for micro-batch 1
# micro_batch_1_nodes = [6660, 5691, 4296, 3298, 2825, 2149, 1749, 1485, 1307, 15017]

# # Color maps for the bars
# colorMap1 = [
#     (0.0000, 0.1570, 0.3140),   # Deep blue
#     (0.0000, 0.2678, 0.5356),   # Dark blue
#     (0.0000, 0.3785, 0.7573),   # Medium blue
#     (0.1421, 0.4883, 0.9791),   # Light blue
#     (0.3930, 0.5682, 0.9564),   # Lighter blue
#     (0.6440, 0.6482, 0.9337)    # Lightest blue
# ]

# colorMap2 = [
#     (0.8500, 0.3250, 0.0980),   # Deep orange
#     (0.9000, 0.3800, 0.1100),   # Darker orange
#     (0.9500, 0.4350, 0.1220),   # Dark orange
#     (1.0000, 0.4900, 0.1340),   # Medium orange
#     (1.0000, 0.5450, 0.1460),   # Light orange
#     (1.0000, 0.6000, 0.1580),   # Lighter orange
#     (1.0000, 0.6550, 0.1700)    # Lightest orange
# ]

# # Create a figure and set the size
# plt.figure(figsize=(7, 2))
# # Set global grid customization
# plt.rcParams['grid.color'] = 'grey'
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.linewidth'] = 0.2
# # Subplot 1: Full Batch
# f,ax = plt.subplot(1, 2, 1)
# plt.bar(degrees, full_batch_nodes, width=0.4, color=colorMap1[3], edgecolor='none')
# plt.xlabel('# of degree', fontsize=14)
# plt.ylabel('# of nodes', fontsize=14)
# plt.legend(['Full batch'], fontsize=14)
# plt.ylim(0, 30000)
# formatter = ScalarFormatter(useMathText=True)  # Create a formatter object
# formatter.set_scientific(True)  # Force the use of scientific notation
# formatter.set_powerlimits((-1,1))  # Use scientific notation when exponent is larger than 1 or smaller than -1

# # ax.xaxis.set_major_formatter(formatter)  # Apply formatter to x-axis
# ax.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis
# plt.text(5, -11000, '(a)', fontsize=14, verticalalignment='top')
# plt.xticks(degrees)
# plt.grid(True)

# # Subplot 2: Micro-batch Comparison
# f,ax = plt.subplot(1, 2, 2)
# plt.bar(degrees - 0.2, micro_batch_0_nodes, width=0.4, color=colorMap1[4], edgecolor='none')
# plt.bar(degrees + 0.2, micro_batch_1_nodes, width=0.4, color=colorMap2[2], edgecolor='none')
# plt.xlabel('# of degree', fontsize=14)
# plt.ylabel('# of nodes', fontsize=14)
# plt.ylim(0, 30000)
# formatter = ScalarFormatter(useMathText=True)  # Create a formatter object
# formatter.set_scientific(True)  # Force the use of scientific notation
# formatter.set_powerlimits((-1,1))  # Use scientific notation when exponent is larger than 1 or smaller than -1

# # ax.xaxis.set_major_formatter(formatter)  # Apply formatter to x-axis
# ax.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis
# plt.legend(['Micro-batch 0', 'Micro-batch 1'], fontsize=14)
# plt.xticks(degrees)
# plt.grid(True)
# plt.text(5, -11000, '(b)', fontsize=14, verticalalignment='top')

# plt.tight_layout()
# plt.subplots_adjust(left=0.12, right=0.98, bottom=0.3, top=0.95) 
# plt.savefig('./products.pdf', format='pdf')

# plt.show()




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Data for the full batch and micro-batches
degrees = np.arange(1, 11)
full_batch_nodes = [13428, 11706, 9277, 7320, 6222, 4868, 4045, 3472, 2976, 27627]
micro_batch_0_nodes = [6768, 6015, 4981, 4022, 3397, 2719, 2296, 1987, 1669, 12610]
micro_batch_1_nodes = [6660, 5691, 4296, 3298, 2825, 2149, 1749, 1485, 1307, 15017]

# Colors for the bars
colorMap1 = [(0.0000, 0.1570, 0.3140), (0.3930, 0.5682, 0.9564)]
colorMap2 = [(0.9500, 0.4350, 0.1220)]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.2))

# Subplot 1: Full Batch
ax1.bar(degrees, full_batch_nodes, width=0.4, color=colorMap1[1], edgecolor='none')
ax1.set_xlabel('# degree', fontsize=14)
ax1.set_ylabel('# nodes', fontsize=14)
ax1.legend(['Full batch'], fontsize=14)
ax1.set_ylim(0, 30000)
ax1.set_xticks(degrees)
ax1.grid(True, linestyle='--', linewidth=0.5, color='grey')
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax1.yaxis.set_major_formatter(formatter)
ax1.text(5, -12000, '(a)', fontsize=14, verticalalignment='top')

# Subplot 2: Micro-batch Comparison
ax2.bar(degrees - 0.2, micro_batch_0_nodes, width=0.4, color=colorMap1[1], edgecolor='none')
ax2.bar(degrees + 0.2, micro_batch_1_nodes, width=0.4, color=colorMap2[0], edgecolor='none')
ax2.set_xlabel('# degree', fontsize=14)
ax2.set_ylabel('# nodes', fontsize=14)
ax2.set_ylim(0, 30000)
ax2.legend(['Micro-batch 0', 'Micro-batch 1'], fontsize=14)
ax2.set_xticks(degrees)
ax2.grid(True, linestyle='--', linewidth=0.5, color='grey')
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax2.yaxis.set_major_formatter(formatter)
ax2.text(5, -12000, '(b)', fontsize=14, verticalalignment='top')

plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.99, bottom=0.35, top=0.9)
plt.savefig('./products.pdf', format='pdf')
plt.show()
