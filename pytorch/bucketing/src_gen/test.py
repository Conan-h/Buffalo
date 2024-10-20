# import src_gen
# import numpy as np

# # Generate random data for testing
# local_in_edges_tensor_list = [  [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)] for ii in range(3)]
# global_batched_nids_list = [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)]
# # print('local_in_edges_tensor_list ', local_in_edges_tensor_list)
# # Generate mappings for induced_src and eids_global
# # induced_src = {i: np.random.randint(0, 50) for i in range(10)}
# # eids_global = {i: np.random.randint(0, 50) for i in range(10)}
# induced_src =  [np.random.randint(0, 50) for i in range(10)]
# eids_global =  [np.random.randint(0, 50) for i in range(10)]

# # Call the src_gen function
# eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)
# print("Eids List: ", eids_list)
# print("Src Long List: ", src_long_list)
# # eids_list = list({k: eids_list[k] for k in sorted(eids_list)}.values())
# # src_long_list = list({k: src_long_list[k] for k in sorted(src_long_list)}.values())
# # # Print the output
# # print("Eids List: ", eids_list)
# # print("Src Long List: ", src_long_list)
import src_gen
import numpy as np
def test_gen_src():
    local_in_edges_tensor_list = [ [[5, 7, 8, 3, 7, 12], [12, 16, 11, 19, 15,14], [1, 2, 3, 4, 5,6]] ,[[5,5,7], [15,7,2], [9,7,8]]]
    global_batched_nids_list = [[12, 16, 11, 19, 14],[15,7,2]]
    print('global_batched_nids_list ', global_batched_nids_list)
    print('local_in_edges_tensor_list ', local_in_edges_tensor_list)
    # induced_src = [17,23,34,45,6,47,29,30,11,14]
    induced_src = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    eids_global = [10,11,12,13,14,15,16,17,18,19]
    

    result = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)

    eids_list, tails_list = result

    print("EIDS List: ", eids_list)
    print("Tails List: ", tails_list)

if __name__ == "__main__":
    test_gen_src()
