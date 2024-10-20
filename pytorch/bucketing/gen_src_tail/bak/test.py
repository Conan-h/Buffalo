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






def gen_src_tails(local_in_edges_tensor_list, global_batched_nids_list, induced_src_vec, eids_global_vec):
    # Generate induced_src and eids_global mapping
    induced_src = {i: induced_src_vec[i] for i in range(len(induced_src_vec))}
    eids_global = {i: eids_global_vec[i] for i in range(len(eids_global_vec))}

    eids_list = {}
    src_long_list = {}

    for i in range(len(local_in_edges_tensor_list)):
        local_in_edges_tensor = local_in_edges_tensor_list[i]
        mini_batch_src_local = list(dict.fromkeys(local_in_edges_tensor[0]))

        mini_batch_src_global = []
        for local in mini_batch_src_local:
            mini_batch_src_global.append(induced_src[local])

        r_ = [global for global in mini_batch_src_global if global not in global_batched_nids_list[i]]
        eid_local_list = local_in_edges_tensor[2]

        global_eid_tensor = []
        for local in eid_local_list:
            global_eid_tensor.append(eids_global[local])

        eids_list[i] = global_eid_tensor
        src_long_list[i] = r_

    return eids_list, src_long_list
