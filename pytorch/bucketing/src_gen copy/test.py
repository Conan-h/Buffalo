import src_gen
import numpy as np

# Generate random data for testing
local_in_edges_tensor_list = [  [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)] for ii in range(3)]
global_batched_nids_list = [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)]
# print('local_in_edges_tensor_list ', local_in_edges_tensor_list)
# Generate mappings for induced_src and eids_global
# induced_src = {i: np.random.randint(0, 50) for i in range(10)}
# eids_global = {i: np.random.randint(0, 50) for i in range(10)}
induced_src =  [np.random.randint(0, 50) for i in range(10)]
eids_global =  [np.random.randint(0, 50) for i in range(10)]

# Call the src_gen function
eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)
print("Eids List: ", eids_list)
print("Src Long List: ", src_long_list)
# eids_list = list({k: eids_list[k] for k in sorted(eids_list)}.values())
# src_long_list = list({k: src_long_list[k] for k in sorted(src_long_list)}.values())
# # Print the output
# print("Eids List: ", eids_list)
# print("Src Long List: ", src_long_list)
