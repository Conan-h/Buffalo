import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
legend_font=12
font_properties = {'size': 11}
# Data for the full batch and micro-batches


arxiv = [16.66,16.96,17.40,16.67]
batches_4 = np.arange(1, 5)

products = [20.38,20.78,20.73,20.83,20.78,20.35,20.77,20.62,20.59,20.60,20.53,20.48]
batches_12 = np.arange(1, 13)

papers100M = [58.60,61.71,61.10,61.23,60.43,60.25,57.94,59.42]
batches_8 = np.arange(1, 9)


# Colors for the bars
                # yellow,           green,                    blue,             orange,              pink
colorMap1 = [(0.9290, 0.6940, 0.1250),(0.4660, 0.6740, 0.1880),(0.0000, 0.1570, 0.3140), (0.3930, 0.5682, 0.9564),(0.9, 0.5470, 0.7410)]
colorMap2 = [(0.9500, 0.4350, 0.1220)]
color_yellow =(0.9290, 0.6940, 0.1250)

# Create figure and subplots
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(6.5, 2.3))
# Subplot 0: cora Full Batch
print(batches_4)
print(arxiv)

ax0.bar(batches_4, arxiv, width=0.4, color=colorMap1[1], edgecolor='none')
# ax0.set_xlabel('# micro-batches', fontsize=14)
ax0.set_ylabel('CUDA memory (GB)', fontsize=14)
# ax0.legend(['full batch'], fontsize=legend_font)

# ax0.legend(['Batch'], fontsize=legend_font)
# ax0.text(5.7, 1500, '(sampling ', fontsize=11, verticalalignment='top')
# ax0.text(5.7, 1200, 'subgraph)', fontsize=11, verticalalignment='top')
ax0.set_ylim(0, 24)
ax0.set_xticks(batches_4)
# ax0.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax0.set_title('OGBN-arxiv', fontsize=12)
# ax0.legend(frameon=False, prop=font_properties)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1))
# ax0.yaxis.set_major_formatter(formatter)
ax0.text(2, -6, '(a)', fontsize=14, verticalalignment='top')






ax1.bar(batches_12, products, width=0.4, color=color_yellow, edgecolor='none')
# ax1.set_xlabel('# micro-batches', fontsize=14)
# ax1.set_ylabel('# nodes', fontsize=14)

ax1.set_ylim(0, 24)
ax1.set_xticks(batches_12)
# ax1.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax1.set_title('OGBN-products', fontsize=12)
# ax1.legend(frameon=False,  prop=font_properties)
ax1.text(0, -11, 'Id of micro-batch ', fontsize=16, verticalalignment='top')
# ax0.text(5.7, 1200, 'subgraph)', fontsize=11, verticalalignment='top')

# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1))
# ax1.yaxis.set_major_formatter(formatter)
ax1.text(5, -6, '(b)', fontsize=14, verticalalignment='top')




ax2.bar(batches_8, papers100M, width=0.4, color=colorMap1[3], edgecolor='none', label='papers100M')
# ax2.set_xlabel('# micro-batches', fontsize=14)
# ax2.set_ylabel('# nodes', fontsize=14)
ax2.set_ylim(0, 80)
# ax2.legend(['micro-batch 0', 'micro-batch 1'], fontsize=legend_font-1)
ax2.set_xticks(batches_8)
# ax2.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax2.set_title('OGBN-papers100M', fontsize=12)
# ax2.legend(frameon=False,  prop=font_properties)

# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1))
# ax2.yaxis.set_major_formatter(formatter)
ax2.text(4, -19, '(c)', fontsize=14, verticalalignment='top')


plt.tight_layout()
plt.subplots_adjust(left=0.09, right=0.995, bottom=0.35, top=0.75, wspace=0.16,)
plt.savefig('./balanced_mem.pdf', format='pdf')
plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# # Create a figure
# fig = plt.figure(figsize=(10, 4))

# # Create a GridSpec object with 1 row and 6 columns
# gs = GridSpec(1, 6, figure=fig)

# # Create the first subplot with width 2
# ax1 = fig.add_subplot(gs[0, :1])
# ax1.plot([1, 2, 3], [1, 2, 3])
# ax1.set_title('Width 2')

# # Create the second subplot with width 3
# ax2 = fig.add_subplot(gs[0, 1:4])
# ax2.plot([1, 2, 3], [3, 2, 1])
# ax2.set_title('Width 3')

# ax3 = fig.add_subplot(gs[0, 4:])
# ax3.plot([1, 2, 3], [3, 2, 1])
# ax3.set_title('Width 2')

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.savefig('./balance.pdf', format='pdf')
# plt.show()
