import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
# Data
values = [173, 221, 374, 456, 445, 550, 655, 512, 576, 192609]
labels = range(len(values))

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(2.7, 2.2))
light_blue = (173/255, 216/255, 230/255)
blue = (100/255, 116/255, 230/255)
colors_rgb = [light_blue]*9
colors_rgb.append(blue)
# Plot on each subplot
ax1.bar(labels, values, color=blue)
ax2.bar(labels, values, color=colors_rgb)

# Break the y-axis into two parts and set limits
ax1.set_ylim(180000, 200000)  # Upper plot for the large outlier
ax2.set_ylim(0, 2000)          # Lower plot for smaller values

# Hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add diagonal lines to indicate the break in the y-axis
d = .015  # Size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal
formatter = ScalarFormatter(useMathText=True)  # Create a formatter object
formatter.set_scientific(True)  # Force the use of scientific notation
formatter.set_powerlimits((-1,1))  # Use scientific notation when exponent is larger than 1 or smaller than -1

# ax1.xaxis.set_major_formatter(formatter)  # Apply formatter to x-axis
ax1.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis



kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal
formatter = ScalarFormatter(useMathText=True)  # Create a formatter object
formatter.set_scientific(True)  # Force the use of scientific notation
formatter.set_powerlimits((-1,1))  # Use scientific notation when exponent is larger than 1 or smaller than -1

# ax2.xaxis.set_major_formatter(formatter)  # Apply formatter to x-axis
ax2.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis

# Labels and title
plt.xlabel('# degree', fontsize=14)

ax1.set_ylabel('# nodes          ', fontsize=14)
# ax2.set_ylabel('Frequency (low values)')
x_ticks = range(1, 11)
str_list = [str(num) for num in x_ticks]
ax2.set_xticks(range(10))
ax2.set_xticklabels(str_list)
ax1.grid(True, color='grey', linestyle='-.', linewidth=0.2)
ax2.grid(True, color='grey', linestyle='-.', linewidth=0.2)
# fig.suptitle('Degree Frequency of OGBN-products \n after sampling')
fig.suptitle('          OGBN-products \n           after sampling')

plt.subplots_adjust(left=0.22, right=0.98, bottom=0.225, top=0.8) 
plt.savefig('./figure2.pdf', format='pdf')

plt.show()
