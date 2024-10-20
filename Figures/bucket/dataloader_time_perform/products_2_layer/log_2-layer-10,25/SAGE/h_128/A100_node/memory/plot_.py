import matplotlib.pyplot as plt  
from matplotlib.ticker import ScalarFormatter  
# Data  
number_of_nodes = [98285, 48965, 16380, 10917]  
memory = [76.65, 47, 21.21, 15.34]  
number_of_micro_batch=[2,4,12,18]

# Plotting  
plt.figure(figsize=(9, 5.2))  
plt.plot(memory, number_of_nodes, marker='o')  
# plt.title('Memory vs. Number of Output Nodes', fontsize=16)  
plt.xlabel('Memory (GB)', fontsize=16)  
plt.ylabel('Average Number of Output \n Nodes per Bucket Group', fontsize=16)  
plt.grid(True)  

# Adjusting tick label sizes  
plt.tick_params(axis='both', which='major', labelsize=14)  
# Formatting y-axis to use scientific notation  
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))  


# Adding text annotations for each point  
for i in range(len(memory)):  
    plt.text(memory[i], number_of_nodes[i]+1000, f'{number_of_micro_batch[i]}', fontsize=16, ha='right')  

# Saving the figure  
plt.savefig('mem-vs-number_of_output_nodes.pdf')  
plt.show()