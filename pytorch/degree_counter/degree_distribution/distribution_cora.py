import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
legend_font=12
font_properties = {'size': 11}
# Data for the full batch and micro-batches
degrees = np.arange(1, 11)
cora_full_batch = [485,583,553,389,281,131,82,57,25,122]
full_batch_nodes = [13428, 11706, 9277, 7320, 6222, 4868, 4045, 3472, 2976, 27627]
micro_batch_0_nodes = [6768, 6015, 4981, 4022, 3397, 2719, 2296, 1987, 1669, 12610]
micro_batch_1_nodes = [6660, 5691, 4296, 3298, 2825, 2149, 1749, 1485, 1307, 15017]

# Colors for the bars
colorMap1 = [(0.0000, 0.1570, 0.3140), (0.3930, 0.5682, 0.9564)]
colorMap2 = [(0.9500, 0.4350, 0.1220)]

# Create figure and subplots
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(6.1, 2.2))
# Subplot 0: cora Full Batch
ax0.bar(degrees, cora_full_batch, width=0.4, color=colorMap1[1], edgecolor='none',label='Batch')
ax0.set_xlabel('# degree', fontsize=14)
ax0.set_ylabel('# nodes', fontsize=14)
# ax0.legend(['full batch'], fontsize=legend_font)

# ax0.legend(['Batch'], fontsize=legend_font)
ax0.text(5.7, 1500, '(sampling ', fontsize=11, verticalalignment='top')
ax0.text(5.7, 1200, 'subgraph)', fontsize=11, verticalalignment='top')
ax0.set_ylim(0, 2000)
ax0.set_xticks(degrees)
# ax0.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax0.set_title('Cora', fontsize=12)
ax0.legend(frameon=False, prop=font_properties)
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax0.yaxis.set_major_formatter(formatter)
ax0.text(5, -840, '(a)', fontsize=14, verticalalignment='top')





# Subplot 1: Full Batch
ax1.bar(degrees, full_batch_nodes, width=0.4, color=colorMap1[1], edgecolor='none',label='Batch')
ax1.set_xlabel('# degree', fontsize=14)
# ax1.set_ylabel('# nodes', fontsize=14)
# ax1.legend(['Batch'], fontsize=legend_font)
ax1.text(2.3, 22600, '(sampling ', fontsize=11, verticalalignment='top')
ax1.text(2.3, 18600, 'subgraph)', fontsize=11, verticalalignment='top')
ax1.set_ylim(0, 30000)
ax1.set_xticks(degrees)
# ax1.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax1.set_title('OGBN-arxiv', fontsize=12)
ax1.legend(frameon=False,  prop=font_properties)


formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax1.yaxis.set_major_formatter(formatter)
ax1.text(5, -12800, '(b)', fontsize=14, verticalalignment='top')


# Subplot 2: Micro-batch Comparison
ax2.bar(degrees - 0.2, micro_batch_0_nodes, width=0.4, color=colorMap1[1], edgecolor='none', label='Micro-batch 0')
ax2.bar(degrees + 0.2, micro_batch_1_nodes, width=0.4, color=colorMap2[0], edgecolor='none', label='Micro-batch 1')
ax2.set_xlabel('# degree', fontsize=14)
# ax2.set_ylabel('# nodes', fontsize=14)
ax2.set_ylim(0, 30000)
# ax2.legend(['micro-batch 0', 'micro-batch 1'], fontsize=legend_font-1)
ax2.set_xticks(degrees)
# ax2.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax2.set_title('OGBN-arxiv', fontsize=12)
ax2.legend(frameon=False,  prop=font_properties)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax2.yaxis.set_major_formatter(formatter)
ax2.text(5, -12800, '(c)', fontsize=14, verticalalignment='top')


plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.995, bottom=0.35, top=0.85, wspace=0.15,)
plt.savefig('./cora_arxiv_distribution.pdf', format='pdf')
plt.show()
