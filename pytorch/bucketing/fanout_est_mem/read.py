import json
def read_est_mem(filename):
	# Open the file in read mode

	with open(filename, 'r') as file:
		# Load the dictionary from the file
		data_str = file.read()
        # Evaluate the string as a python dictionary
		dict_data = eval(data_str)


	# Convert the dictionary values to a list
	list_data = list(dict_data.values())
	# Now list_data contains the dictionary values
	return list_data

print(read_est_mem('/home/cc/Betty_baseline/pytorch/bucketing/fanout_est_mem/fanout_800_est_mem.txt'))