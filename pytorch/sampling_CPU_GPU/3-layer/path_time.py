
def print_vals(val_list):
    # Print the selected values
    for value in val_list:
        print(value)


# Specify the file path
file_path = '100_epoch_CUDA_3-layers_h_128___.log'  # Replace with the path to your file
# file_path = '0,2.log'  # Replace with the path to your file
# file_path = '0,1.log'  # Replace with the path to your file
# file_path = '0,1,2.log'  # Replace with the path to your file

# Initialize a list to store the values
fw_values = []
bw_values = []
opt_values = []
g2c_values =[]
c2g_values =[]
bw_allreduce_values=[]
iteration_time_list=[]
max_mem=0
# Open the file and read lines starting from line 3918
with open(file_path, 'r') as file:
    lines = file.readlines()[1800:2100]  # Lines are 0-indexed, so 3917 corresponds to line 3918
    # print(lines)
    # Iterate through the lines
    for line in lines:
        if "Pure training time/epoch" in line:
            value_list = line.split(" ")[1:]
            print(value_list)
            iteration_time = float(value_list[0].split("elapsed time per iteration (ms): ")[1].strip())
            iteration_time_list.append(iteration_time)
        
        if "MaxMemAllocated=" in line:
            value_list = line.split(",")[5]
            max_mem = value_list.split("=")[1].strip()
            
        
            
            
# print('forward time')
# print_vals(fw_values)



avg_iteration=(sum(fw_values)+sum(bw_values)+sum(opt_values))/len(fw_values)
print('the avg iteration time (sec) calculated', avg_iteration)
print()


print("the avg iteration time (sec)   measured", sum(iteration_time_list)/len(iteration_time_list))
print()
print("forward time average  (sec) ", sum(fw_values)/len(fw_values))

print()

# print()
# # print("backward time")
# # print_vals(bw_values)
print("backward time average (sec) ", sum(bw_values)/len(bw_values))

print()

print("optimizer time average (sec)", sum(opt_values)/len(opt_values))

print('-'*60)

print("bwd allreduce time average  (sec) ", sum(bw_allreduce_values)/len(bw_allreduce_values))
print()

print("gpu-->cpu time average  (sec) ", sum(g2c_values)/len(g2c_values))
print()
print("cpu-->gpu time average  (sec) ", sum(c2g_values)/len(c2g_values))
print()


print('-='*40)
print("max cuda mem allocated :",max_mem)
print()

print("file path :",file_path)
print()