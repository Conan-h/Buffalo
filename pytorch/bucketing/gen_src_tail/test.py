import gen_src_tails
import numpy as np
def test_gen_src():
    local_in_edges_tensor_list = [ [[5, 7, 8, 3, 7, 12], [12, 16, 11, 19, 15,14], [1, 2, 3, 4, 5,6]] ,[[5,5,7], [15,7,2], [9,7,8]]]
    global_batched_nids_list = [[12, 16, 11, 19, 14],[15,7,2]]
    print('global_batched_nids_list ', global_batched_nids_list)
    print('local_in_edges_tensor_list ', local_in_edges_tensor_list)
    # induced_src = [17,23,34,45,6,47,29,30,11,14]
    induced_src = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    eids_global = [10,11,12,13,14,15,16,17,18,19]
    

    result = gen_src_tails.gen_src_tails(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)

    eids_list, tails_list = result

    print("EIDS List: ", eids_list)
    print("Tails List: ", tails_list)

if __name__ == "__main__":
    test_gen_src()
