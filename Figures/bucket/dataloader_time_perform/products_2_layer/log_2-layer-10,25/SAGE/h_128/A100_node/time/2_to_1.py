import matplotlib.pyplot as plt  
from matplotlib.ticker import ScalarFormatter  

# Data for the plots  
nb = [2, 4, 12, 18]  
iteration_time = [9.379, 11.022, 14.3771, 18.946]  
gpu_memory_cost = [72.64, 47.05, 21.27, 15.34]  

number_of_nodes = [98285, 48965, 16380, 10917]  
memory = [76.65, 47, 21.21, 15.34]  
number_of_micro_batch = [2, 4, 12, 18]  

# Create a figure  
fig, ax1 = plt.subplots(figsize=(10, 6))  

# Left y-axis: Average Number of Output Nodes per Bucket Group  
ax1.plot(memory, number_of_nodes, marker='o', color='blue', label='Average Number of Output Nodes')  
ax1.set_xlabel('Memory (GB)', fontsize=16)  
ax1.set_ylabel('Average Number of Output \n Nodes per Bucket Group', fontsize=16, color='blue')  
ax1.tick_params(axis='y', labelcolor='blue')  
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))  
ax1.grid(True)  

# Adding text annotations for each point in the left y-axis  
for i in range(len(memory)):  
    ax1.text(memory[i], number_of_nodes[i] + 1000, f'{number_of_micro_batch[i]}', fontsize=16, ha='right')  

# Right y-axis: Iteration Time (sec)  
ax2 = ax1.twinx()  
ax2.scatter(gpu_memory_cost, iteration_time, color='red', marker='o', label='Per-Iteration Time(s)')  
ax2.set_ylabel('Iteration Time (sec)', fontsize=16, color='red')  
ax2.tick_params(axis='y', labelcolor='red')  

fig.tight_layout()  # Adjust spacing to prevent overlap  

# Saving the figure  
plt.savefig('combined_plot_with_two_y_axes.pdf')  
# Show the plot  
plt.show()