# # from multiprocessing import Pool, Manager
# # from functools import partial

# # def process_data(shared_dict, shared_list, items):
# #     # items is a list. Modify the function to work with a list.
# #     print('current items ', items)
# #     for item in items:
# #         print(item)
# #         shared_dict[item] = item * 2
# #         print('shared_dict ', item * 2)
# #         shared_list.append(item * 3)
# #         print('shared_list ', item * 3)

# # if __name__ == "__main__":
    
# #     with Manager() as manager:
# #         shared_dict = manager.dict()
# #         shared_list = manager.list()
# #         items = [[j for j in range(i, i + 2)] for i in range(0, 10, 2)]  # A list of lists.
# #         print('items ',items)
# #         process_data_partial = partial(process_data, shared_dict, shared_list)

# #         with Pool() as p:
# #             p.map(process_data_partial, items)

# #         print(dict(shared_dict)) 
# #         print(sorted(list(set(shared_list))))  


# # from multiprocessing import Pool, Manager
# # from functools import partial

# # def process_data(shared_dict, shared_list, items):
# #     # items is a list. Modify the function to work with a list.
# #     for item in items:
# #         shared_dict[item] = item * 2
# #         shared_list.append(item * 3)
# #     # return the results for this batch of items.
# #     return (dict(shared_dict), sorted(list(set(shared_list))))

# # if __name__ == "__main__":
# #     with Manager() as manager:
# #         shared_dict = manager.dict()
# #         shared_list = manager.list()
# #         items = [[j for j in range(i, i + 2)] for i in range(0, 10, 2)]  # A list of lists.

# #         process_data_partial = partial(process_data, shared_dict, shared_list)

# #         with Pool() as p:
# #             results = p.map(process_data_partial, items)

# #         # Print the results from all processes.
# #         # for res in results:
# #         #     print('shared_dict', res[0])
# #         #     print('shared_list', res[1])
# #     for res in results:
# #             print('shared_dict______', res[0])
# #             print('shared_list______', res[1])
# #             print()



# # from multiprocessing import Pool, Manager
# # from functools import partial
# # import torch
# # import dgl

# # def process_data(shared_dict, shared_list, induced_src, eids_global, shared_dict_nid_2_local, shared_current_layer_block, items):
# #     # items is a list. Modify the function to work with a list.
# #     for item in items:
# #         shared_dict[item] = item * 2
# #         shared_list.append(item * 3)
# #     # return the results for this batch of items.
# #     return (dict(shared_dict), sorted(list(set(shared_list))))

# # if __name__ == "__main__":
# #     # Source nodes for edges (2, 1), (3, 2), (4, 3)
# #     src_ids = torch.tensor([2, 3, 4])
# #     # Destination nodes for edges (2, 1), (3, 2), (4, 3)
# #     dst_ids = torch.tensor([1, 2, 3])
# #     g = dgl.graph((src_ids, dst_ids))
    
# #     induced_src = torch.tensor([1,2,3])
# #     eids_global = torch.tensor([4,5,6,7])
# #     dict_nid_2_local = dict(zip(induced_src.tolist(), range(len(induced_src))))
# #     with Manager() as manager:
# #         induced_src.share_memory_()
# #         eids_global.share_memory_()
# #         shared_dict_nid_2_local = manager.dict(dict_nid_2_local)
# #         shared_current_layer_block = manager.Namespace()  # create a namespace to contain the shared object
# #         shared_current_layer_block.value = g
        
# #         shared_dict = manager.dict()
# #         shared_list = manager.list()
# #         items = [[j for j in range(i, i + 2)] for i in range(0, 10, 2)]  # A list of lists.

# #         process_data_partial = partial(process_data, shared_dict, shared_list, induced_src, eids_global, shared_dict_nid_2_local, shared_current_layer_block)

# #         with Pool() as p:
# #             results = p.map(process_data_partial, items)

# #         # Join the results from all processes.
# #         joined_dict = {}
# #         joined_list = []
# #         for res in results:
# #             joined_dict.update(res[0])  # join dictionaries
# #             joined_list += res[1]  # join lists

# #         print('joined_dict', joined_dict)
# #         print('joined_list', sorted(list(set(joined_list))))
	
	

# # from multiprocessing import Pool

# # # Define your global variable here
# # values_to_remove = {'banana'}

# # def remove_values(item):
# #     # Here you are saying that you want to use the global variable values_to_remove
# #     global values_to_remove
# #     return item not in values_to_remove

# # # Create your list here
# # long_list = ['apple', 'banana', 'cherry', 'banana', 'pear', 'apple', 'orange']

# # # Create your pool and map the function to the list
# # with Pool() as pool:
# #     new_list = pool.map(remove_values, long_list)

# # print(new_list)

# from multiprocessing import Pool

# # Define your global variable here
# values_to_remove = {'banana'}

# def remove_values(item):
#     # Here you are saying that you want to use the global variable values_to_remove
#     global values_to_remove
#     if item not in values_to_remove:
#         return item
#     else:
#         return None

# # Create your list here
# long_list = ['apple', 'banana', 'cherry', 'banana', 'pear', 'apple', 'orange']

# # Create your pool and map the function to the list
# with Pool() as pool:
#     new_list = pool.map(remove_values, long_list)

# # Filter out the None values
# new_list = [i for i in new_list if i]

# print(new_list)


def split_list(input_list, K):
    avg = len(input_list) / float(K)
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg

    return out
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
print(split_list(lst, 3))
