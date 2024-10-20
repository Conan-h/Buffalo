# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import pickle
# import math

# # Load data for subplot (a)
# with open("products_degree_counts_data.pkl", "rb") as file:
#     counts = pickle.load(file)

# # Data for subplot (b)
# values = [173, 221, 374, 456, 445, 550, 655, 512, 576, 192609]
# labels = range(len(values))
# light_blue = (173/255, 216/255, 230/255)
# blue = (100/255, 116/255, 230/255)
# colors_rgb = [light_blue]*9 + [blue]

# # Create figure and axes for two subplots
# fig, (ax1, ax2a, ax2b) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 3, 1]})

# # Subplot (a)
# ax1.plot(range(1, len(counts)+1), counts, linestyle='-', color='b')
# ax1.set_title('In-degree Frequency of OGBN-products')
# ax1.set_xlabel('in-degree', fontsize=14)
# ax1.set_ylabel('Frequency', fontsize=14)
# ax1.set_xscale('log')
# x_ticks = [10**i for i in range(int(math.log10(1)), int(math.log10(len(counts)))+1)]
# ax1.set_xticks(x_ticks)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1,1))
# ax1.xaxis.set_major_formatter(formatter)
# ax1.yaxis.set_major_formatter(formatter)

# # Subplot (b) with broken y-axis
# ax2a.bar(labels, values, color=colors_rgb)
# ax2a.set_ylim(180000, 200000)  # Upper part for the outlier
# ax2b.bar(labels, values, color=colors_rgb)
# ax2b.set_ylim(0, 2000)  # Lower part for the rest

# # Hide the spines and ticks for the broken axis
# ax2a.spines['bottom'].set_visible(False)
# ax2b.spines['top'].set_visible(False)
# ax2a.xaxis.tick_top()
# ax2a.tick_params(labeltop=False)  # Hide top tick labels

# # Add diagonal lines to indicate the break
# d = .015  # Size of diagonal lines
# kwargs = dict(transform=ax2a.transAxes, color='k', clip_on=False)
# ax2a.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
# ax2a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

# kwargs.update(transform=ax2b.transAxes)  # Switch to the bottom axes
# ax2b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
# ax2b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# # Common properties for subplot (b)
# for ax in [ax2a, ax2b]:
#     ax.set_xlabel('Index', fontsize=14)
#     ax.set_ylabel('Values', fontsize=14)
#     ax.xaxis.set_tick_params(which='both', labelbottom=True)

# # Adjust subplot spacings
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.4)
# plt.savefig('./combined_plots.pdf', format='pdf')
# plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import pickle
# import math

# # Load data for subplot (a)
# with open("products_degree_counts_data.pkl", "rb") as file:
#     counts = pickle.load(file)

# # Data for subplot (b)
# values = [173, 221, 374, 456, 445, 550, 655, 512, 576, 192609]
# labels = range(len(values))
# light_blue = (173/255, 216/255, 230/255)
# blue = (100/255, 116/255, 230/255)
# colors_rgb = [light_blue]*9 + [blue]

# # Create figure and axes for two subplots
# fig, axs = plt.subplots(1, 3, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1, 0.05], 'hspace': 0.05})

# ax1 = axs[0]  # In-degree frequency plot
# ax2a = axs[1]  # Upper part for the broken y-axis
# ax2b = axs[2]  # Lower part for the broken y-axis

# # Subplot (a)
# ax1.plot(range(1, len(counts)+1), counts, linestyle='-', color='b')
# ax1.set_title('In-degree Frequency of OGBN-products')
# ax1.set_xlabel('in-degree', fontsize=14)
# ax1.set_ylabel('Frequency', fontsize=14)
# ax1.set_xscale('log')
# x_ticks = [10**i for i in range(int(math.log10(1)), int(math.log10(len(counts)))+1)]
# ax1.set_xticks(x_ticks)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1,1))
# ax1.xaxis.set_major_formatter(formatter)
# ax1.yaxis.set_major_formatter(formatter)

# # Subplot (b) with broken y-axis
# ax2a.bar(labels, values, color=colors_rgb)
# ax2a.set_ylim(180000, 200000)  # Upper part for the outlier
# ax2b.bar(labels, values, color=colors_rgb)
# ax2b.set_ylim(0, 2000)  # Lower part for the rest

# # Ensure ax2a shares x-axis with ax2b
# ax2b.sharex(ax2a)

# # Hide the spines and ticks for the broken axis
# ax2a.spines['bottom'].set_visible(False)
# ax2b.spines['top'].set_visible(False)
# ax2a.xaxis.tick_top()
# ax2a.tick_params(labeltop=False)  # Hide top tick labels
# ax2b.xaxis.tick_bottom()

# # Add diagonal lines to indicate the break
# d = .015  # Size of diagonal lines
# kwargs = dict(transform=ax2a.transAxes, color='k', clip_on=False)
# ax2a.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
# ax2a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

# kwargs.update(transform=ax2b.transAxes)  # Switch to the bottom axes
# ax2b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
# ax2b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# # Common properties for subplot (b)
# for ax in [ax2a, ax2b]:
#     ax.set_xlabel('Index', fontsize=14)
#     ax.set_ylabel('Values', fontsize=14)
#     ax.xaxis.set_tick_params(which='both', labelbottom=True)

# # Adjust subplot spacings
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.4)
# plt.savefig('./combined_plots.pdf', format='pdf')
# plt.show()


# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# # Create the main figure and subplot
# fig, ax = plt.subplots(figsize=(8, 6))

# # Data for the main plot
# x = range(10)
# y = [xi**2 for xi in x]
# ax.plot(x, y, label='Main Plot')
# ax.set_title('Main Plot with Embedded Subplot')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')

# # Add an inset (embedded subplot)
# ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')
# x_inset = range(1, 6)
# y_inset = [xi**3 for xi in x_inset]
# ax_inset.plot(x_inset, y_inset, color='r', label='Embedded Plot')
# ax_inset.set_title('Embedded Subplot')
# ax_inset.set_xlabel('X')
# ax_inset.set_ylabel('Y^3')
# plt.savefig('./combined_plots.pdf', format='pdf')
# # Show the plot
# plt.show()


import matplotlib.pyplot as plt

# Create main figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Data for the line plot (subplot 1)
x = range(10)
y = [xi**2 for xi in x]
ax1.plot(x, y)
ax1.set_title('Line Plot')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')

# Data for the broken bar chart (subplot 2)
values = [10, 50, 150, 600, 1100, 100, 80, 90, 300, 1200]
ax2b = ax2  # the lower part of the broken bar chart
ax2a = ax2.twinx()  # create twin y-axis for the upper part

# Lower part of the broken bar chart
ax2b.bar(x, values, color='blue')
ax2b.set_ylim(0, 200)  # Limit for smaller values
ax2b.set_ylabel('Lower Values')
ax2b.spines['top'].set_visible(False)  # Hide the top spine

# Upper part of the broken bar chart
ax2a.bar(x, values, color='red')
ax2a.set_ylim(1000, 1300)  # Limit for larger values
ax2a.set_ylabel('Upper Values')
ax2a.spines['bottom'].set_visible(False)  # Hide the bottom spine

# Adding diagonal lines to indicate the break
d = .015  # Size of diagonal lines
kwargs = dict(transform=ax2a.transAxes, color='k', clip_on=False)
ax2a.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
ax2a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

# Labels and titles for the broken bar chart
ax2.set_xlabel('X-axis for Both')
ax2.set_title('Broken Bar Chart')

# Adjust layout
plt.subplots_adjust(hspace=0.4)
plt.savefig('./combined_plots.pdf', format='pdf')
# Show the plot
plt.show()
