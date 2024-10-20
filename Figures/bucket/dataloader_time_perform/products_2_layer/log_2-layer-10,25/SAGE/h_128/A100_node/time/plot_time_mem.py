# import matplotlib.pyplot as plt  

# # Data  
# nb = [2, 4, 12, 18]  
# iteration_time = [9.379, 11.022, 14.3771, 18.946]  
# gpu_memory_cost = [72.64, 47.05, 21.27, 15.34]  

# # Create a scatter plot  
# plt.figure(figsize=(10, 5))  
# plt.scatter(gpu_memory_cost, iteration_time, color='blue', marker='o')  

# # Adding labels and title  
# plt.xlabel('GPU Memory Cost (GB)')  
# plt.ylabel('Iteration Time (sec)')  
# plt.title('Iteration Time vs GPU Memory Cost')  

# # Adding grid for better readability  
# plt.grid()  
# # Saving the figure  
# plt.savefig('time-vs-mem.pdf')  
# # Show the plot  
# plt.show()

# import matplotlib.pyplot as plt  
# from matplotlib.ticker import ScalarFormatter  
# # Data  
# number_of_nodes = [98285, 48965, 16380, 10917]  
# memory = [76.65, 47, 21.21, 15.34]  
# number_of_micro_batch=[2,4,12,18]

# # Plotting  
# plt.figure(figsize=(9, 5.2))  
# plt.plot(memory, number_of_nodes, marker='o')  
# # plt.title('Memory vs. Number of Output Nodes', fontsize=16)  
# plt.xlabel('Memory (GB)', fontsize=16)  
# plt.ylabel('Average Number of Output \n Nodes per Bucket Group', fontsize=16)  
# plt.grid(True)  

# # Adjusting tick label sizes  
# plt.tick_params(axis='both', which='major', labelsize=14)  
# # Formatting y-axis to use scientific notation  
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
# plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))  


# # Adding text annotations for each point  
# for i in range(len(memory)):  
#     plt.text(memory[i], number_of_nodes[i]+1000, f'{number_of_micro_batch[i]}', fontsize=16, ha='right')  

# # Saving the figure  
# plt.savefig('mem-vs-number_of_output_nodes.pdf')  
# plt.show()

import matplotlib.pyplot as plt  
from matplotlib.ticker import ScalarFormatter  

# Data for the first plot  
nb = [2, 4, 12, 18]  
iteration_time = [9.379, 11.022, 14.3771, 18.946]  
gpu_memory_cost = [72.64, 47.05, 21.27, 15.34]  

# Data for the second plot  
number_of_nodes = [98285, 48965, 16380, 10917]  
memory = [76.65, 47, 21.21, 15.34]  
number_of_micro_batch = [2, 4, 12, 18]  

# Create a figure with two subplots (1 row, 2 columns)  
fig, axs = plt.subplots(1, 2, figsize=(15, 5))  

# First subplot: Line plot (now the second plot)  
axs[0].plot(memory, number_of_nodes, marker='o')  
axs[0].set_xlabel('Memory (GB)', fontsize=16)  
axs[0].set_ylabel('Average Number of Output \n Nodes per Bucket Group', fontsize=16)  
axs[0].grid(True)  
axs[0].tick_params(axis='both', which='major', labelsize=14)  
axs[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
axs[0].yaxis.get_major_formatter().set_powerlimits((0, 1))  

# Adding text annotations for each point in the first subplot  
for i in range(len(memory)):  
    axs[0].text(memory[i], number_of_nodes[i] + 1000, f'{number_of_micro_batch[i]}', fontsize=16, ha='right')  

# Second subplot: Scatter plot (now the first plot)  
axs[1].scatter(gpu_memory_cost, iteration_time, color='blue', marker='o')  
axs[1].set_xlabel('GPU Memory Cost (GB)', fontsize=16)  
axs[1].set_ylabel('Iteration Time (sec)', fontsize=16)  
axs[1].tick_params(axis='both', which='major', labelsize=14)  
axs[1].grid()  

# Saving the figure  
plt.tight_layout()  # Adjust spacing to prevent overlap  
plt.savefig('combined_plot.pdf')  
# Show the plot  
plt.show()