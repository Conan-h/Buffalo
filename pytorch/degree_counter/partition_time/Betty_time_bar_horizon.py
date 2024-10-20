import matplotlib.pyplot as plt
import numpy as np

# Data Setup
fontsize = 13
start_value = 10
step_size = 15
length_of_list = 3
micro_batches = np.linspace(start_value, start_value + (length_of_list - 1) * step_size, length_of_list)
categories = ['Pure train', 'Block generation', 'Random partition', 'Metis partition', 'Betty(REG) partition']

# Color map
colorMap = np.array([
    [0.9, 0.5470, 0.7410],    # 'pure train' - pink
    [0.4, 0.6470, 0.7410],    # 'block generation' - blue
    [0.4660, 0.6740, 0.1880], # 'random partition' - green
    [0.8500, 0.3250, 0.0980], # 'metis partition' - orange
    [0.9290, 0.6940, 0.1250]  # 'betty partition' - yellow
])

# Data for each category
data_random = np.array([
    [4.32, 6.79, 0.0064, 1e-10, 1e-10],
    [5.46, 9.02, 0.012, 1e-10, 1e-10],
    [7.89, 10.76, 0.014, 1e-10, 1e-10]
])

data_metis = np.array([
    [4.39, 8.3, 1e-10, 0.25, 1e-10],
    [5.46, 10.02, 1e-10, 0.31, 1e-10],
    [7.63, 11.63, 1e-10, 0.25, 1e-10]
])

data_betty = np.array([
    [4.13, 4.35, 1e-10, 1e-10, 14.24],
    [5.26, 5.21, 1e-10, 1e-10, 14.64],
    [7.51, 5.94, 1e-10, 1e-10, 14.57]
])

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

for idx, ax in enumerate(axes):
    data = [data_random, data_metis, data_betty][idx]
    for i, row in enumerate(data.T):
        ax.barh(micro_batches + i*0.2, row, height=0.1, color=colorMap[i], label=categories[i] if idx == 0 else "")
    ax.set_xlabel('Time (sec)', fontsize=fontsize)
    ax.set_title(['Random', 'Metis', 'Betty'][idx])
    ax.set_yticks(micro_batches)
    ax.set_yticklabels(['8', '16', '32'])
    ax.set_xlim(0, max(data.flatten()) + 5)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('./figure.pdf', format='pdf')
plt.show()
