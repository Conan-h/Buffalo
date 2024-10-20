import numpy as np
import itertools
from math import ceil

def backpack_split(weights, values, capacity):
    '''
    multi-backpack algorithm
    return packs=[[]], we just the first pack result, packs[0]
    '''
    n = len(weights)
    
    # Create a table to store the maximum values for each pack and weight combination
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    # table [i-1] means contains the from 1 to i-1 items.
    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j: # weigths idx range[0, n-1], i range [1,n]'i-1' here means the range[0,n-1]
                # If the current weight can fit in the pack
                # Take the maximum of either including the item or excluding it
                table[i][j] = max(values[i - 1] + table[i - 1][j - weights[i - 1]], table[i - 1][j])
            else:
                # If the current weight cannot fit in the pack, exclude the item
                table[i][j] = table[i - 1][j]

    # Determine the items included in each pack
    packs = []
    i, j = n, capacity
    while i > 0 and j > 0:
        if table[i][j] != table[i - 1][j]:
            pack = []
            while i > 0 and j > 0 and table[i][j] != 0:
                if table[i][j] != table[i - 1][j]:
                    pack.append(i - 1)
                    j -= weights[i - 1]
                i -= 1
            packs.append(pack[::-1])
        i -= 1

    # Calculate the maximum value for each pack
    max_values = [table[n][capacity] for _ in range(len(packs))]

    return max_values, packs

def remove_items_by_indices(lst, indices):
    # print('indices ',indices)
    rst = [item for index, item in enumerate(lst) if index not in indices]
    # print('rst ', rst)
    return rst

def remove_bucket_ids(sorted_dict, original_bucket_ids):
    
    # print('original_bucket_ids ',original_bucket_ids)
    rst = {key: value for key, value in sorted_dict.items() if key not in original_bucket_ids}
    # rst = {key:wgt for key, wgt in (sorted_dict) if key not in original_bucket_ids}
    # print('left sorted_dict ', rst)
    
    return rst

def sort_with_original_index(weight_lst):
    '''
    sort the integer weight list in desending order
    '''
    # print(weight_lst)
    # weight list = [weight_0, weight_1, ...]
    indexed_dict = {index: value for index, value in enumerate(weight_lst)}
    #indexed_dict = {0: weight_0, 1: weight_1,...}
    sorted_dict = dict(sorted(indexed_dict.items(), key=lambda x: -x[1]))
    #sorted_dict = {a: weight_a, b: weight_b, ...},  weight_a > weight_b
    # print('indexed_dict ', indexed_dict)
    # print('sorted_dict ', sorted_dict)
    sorted_indices = list(sorted_dict.keys())
    sorted_values = list(sorted_dict.values())
    return sorted_indices, sorted_values, indexed_dict, sorted_dict

def get_index_by_value(dictionary, values_list):
    '''
    get the keys of the given values
    '''
    # some values are the same in the dictionary
    # e.g. dictionary = {0: 9, 1: 29, 2: 39, 3: 54, 4: 49, 5: 39, 6: 17, 7: 14, 8: 22, 9: 18, 10: 5, 11: 14, 12: 10, 13: 14, 14: 28}
    # there are degree bucket 7, 11, 13 share the same weights 14.
    n = len(values_list)
    values_list = list(values_list)
    keys = []
    # print('values_list ', values_list)
    # print('sorted dictionary ', dictionary)
    # print('keys ------')
    
    for va in values_list:
        m = values_list.count(va)
        idx = [key for key, value in dictionary.items() if value == va]
        if m == len(idx):
            if idx[0] not in keys:
                keys += idx
        elif m < len(idx):
            if idx[0] not in keys:
                keys += idx[:m]
        
    # print(keys)
    # print('keys ------end')
    return keys



def split_all(weights, values, capacity):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, my_dict, sorted_dict = sort_with_original_index(weights)
    # print('sorted_dict ', sorted_dict)
    # print()
    # print('weights after sort', sorted_values)
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>=1:
        if sum(weights)< capacity:

            original_index = get_index_by_value(sorted_dict, weights)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            break
        else:    # sum(weights)>= capacity
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 1.65) :
            # if len(packs[0]) > 3 or  (len(packs[0]) > 2 and max(res_tmp) > min(res_tmp)* 1.5) :
            # if len(packs[0]) > 2 and  max(res_tmp) > min(res_tmp)*1.5 :
                aa = min(packs[0]) # remove this aa from current group
                packs[0].remove(aa)
                res_tmp = np.array(weights)[packs[0]]
            GROUPS_weight.append(list(res_tmp))
            

            original_index = get_index_by_value(sorted_dict, res_tmp)
            GROUPS_bucket_idx.append(original_index)
            # print()
            # print("remove bucket_id: ",packs[0])
            # print('original bucket_id :, ', original_index)
            # print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
            # print()
            # print('before remove weights, ',weights)
            weights = remove_items_by_indices(weights, packs[0])
            sorted_dict = remove_bucket_ids(sorted_dict, original_index)
            # print('after remove pre pack weights, ', weights)
            values = weights
                
    if len(weights)==1 :
        if sum(weights)< capacity:
            print('the last batch value is ', weights[0])
            GROUPS_weight.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS_weight, GROUPS_bucket_idx

def split_all_products(weights, values, capacity):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, my_dict, sorted_dict = sort_with_original_index(weights)
    # print('sorted_dict ', sorted_dict)
    # print()
    # print('weights after sort', sorted_values)
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>=1:
        if sum(weights)< capacity:

            original_index = get_index_by_value(sorted_dict, weights)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            break
        else:    # sum(weights)>= capacity
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 1.5) :
            # if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 1.65) :
            # if len(packs[0]) > 3 or  (len(packs[0]) > 2 and max(res_tmp) > min(res_tmp)* 1.5) :
            # if len(packs[0]) > 2 and  max(res_tmp) > min(res_tmp)*1.5 :
                aa = min(packs[0]) # remove this aa from current group
                packs[0].remove(aa)
                res_tmp = np.array(weights)[packs[0]]
            GROUPS_weight.append(list(res_tmp))
            

            original_index = get_index_by_value(sorted_dict, res_tmp)
            GROUPS_bucket_idx.append(original_index)
            # print()
            # print("remove bucket_id: ",packs[0])
            # print('original bucket_id :, ', original_index)
            # print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
            # print()
            # print('before remove weights, ',weights)
            weights = remove_items_by_indices(weights, packs[0])
            sorted_dict = remove_bucket_ids(sorted_dict, original_index)
            # print('after remove pre pack weights, ', weights)
            values = weights
                
    if len(weights)==1 :
        if sum(weights)< capacity:
            print('the last batch value is ', weights[0])
            GROUPS_weight.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS_weight, GROUPS_bucket_idx

def grouping_fanout_products(adjust, weights, capacity):
	# print('the grouping_fanout_1 called successfully')
	# if len(weights) == 49:
	# 	weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
	# if len(weights) == 24:
	# 	weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	weights = [int(item * adjust) for item in weights]

	values =  weights 
	capacity = int(capacity * adjust)
	# print('capacity ', capacity)


	GROUPS_weight, GROUPS_bucket_idx = split_all_products(weights, values, capacity)

	return GROUPS_weight, GROUPS_bucket_idx

def split_list(input_list, K):
    avg = len(input_list) / float(K)
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg
    return out


def split_all_arxiv(weights, values, capacity, fanout, K):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, my_dict, sorted_dict = sort_with_original_index(weights)
    # print('sorted_dict ', sorted_dict)
    # print()
    # print('weights after sort', sorted_values)
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>=1:
        if sum(weights)< capacity:
            original_index = get_index_by_value(sorted_dict, weights)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            # if K == 1:
                # GROUPS_weight.append(weights)
                # GROUPS_bucket_idx.append(original_index)
            # else:
            #     # print('split_list(weights, K) ', split_list(weights, K))
            #     GROUPS_weight += split_list(weights, K)
            #     GROUPS_bucket_idx += split_list(original_index,K)
            break
        else:    # sum(weights)>= capacity
            # if 1.25 * np.mean(res_tmp) < max(res_tmp) and np.mean(res_tmp) < 1.25 * min(res_tmp):
            
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            # print('res_tmp ', res_tmp)
            
            # if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 2.5) :
            # if len(packs[0]) > 3 or  (len(packs[0]) > 2 and max(res_tmp) > min(res_tmp)* 1.5) :
            if len(packs[0]) > int(fanout/K)+1 and  max(res_tmp) > min(res_tmp)*1.5 :
                aa = min(packs[0]) # remove this aa from current group
                packs[0].remove(aa)
                res_tmp = np.array(weights)[packs[0]]
            
                
            GROUPS_weight.append(list(res_tmp))
            

            original_index = get_index_by_value(sorted_dict, res_tmp)
            GROUPS_bucket_idx.append(original_index)
            # print()
            # print("remove bucket_id: ",packs[0])
            # print('original bucket_id :, ', original_index)
            # print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
            # print()
            # print('before remove weights, ',weights)
            weights = remove_items_by_indices(weights, packs[0])
            sorted_dict = remove_bucket_ids(sorted_dict, original_index)
            # print('after remove pre pack weights, ', weights)
            values = weights
                
    if len(weights)==1 :
        if sum(weights)< capacity:
            # print('the last batch value is ', weights[0])
            GROUPS_weight.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS_weight, GROUPS_bucket_idx


def ffd_bin_packing(weights, capacity):
    # Combine weights with their indices
    weights = list(enumerate(weights))
    
    # Sort weights in decreasing order
    weights.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize bins and bin_sums
    bins = []
    bin_sums = []
    
    for index, weight in weights:
        # Try to fit weight into existing bins
        for i, bin_sum in enumerate(bin_sums):
            if bin_sum + weight <= capacity:
                bins[i].append((index, weight))
                bin_sums[i] += weight
                break
        else:
            # If weight can't fit into any existing bin, create a new bin
            bins.append([(index, weight)])
            bin_sums.append(weight)
            
    return bins

def grouping_fanout_arxiv(adjust, weights, capacity, fanout, K):
	# print('the grouping_fanout_arxiv called successfully')
	weights = [int(item * adjust) for item in weights]
	values = weights
	capacity = int(capacity * adjust)
	# print('capacity ', capacity)

	# print(' ')
	GROUPS_weight, GROUPS_bucket_idx = split_all_arxiv(weights, values, capacity, fanout, K) #####
	
	return GROUPS_weight, GROUPS_bucket_idx


def grouping_fanout_arxiv_new(adjust, weights, capacity):
	# print('the grouping_fanout_arxiv called successfully')
	weights = [int(item * adjust) for item in weights]

	capacity = int(capacity * adjust)
	# print('capacity ', capacity)

	# print(' ')
	bins = ffd_bin_packing(weights, capacity) #####
	GROUPS_weight =[]
	GROUPS_bucket_idx =[]
	for i, bin in enumerate(bins):
		GROUPS_weight.append(sum(weight for index, weight in bin))
		GROUPS_bucket_idx.append([ index for index, weight in bin])
	return GROUPS_weight, GROUPS_bucket_idx

# def split_cora(weights, values, capacity, fanout, K):
    
#     # weights.sort(reverse=True)
#     sorted_indices, sorted_values, _, sorted_dict = sort_with_original_index(weights)
#     # print('split cora: sorted_dict ', sorted_dict)
#     # print('weights after sort', sorted_values)
#     # print()
#     weights = sorted_values
#     values = sorted_values
#     GROUPS_weight =[]
#     GROUPS_bucket_idx =[]
#     while len(weights)>=1:
#         if sum(weights)< capacity: # sum of weights are smaller than the capacity, we can group them to one group
#             original_index = get_index_by_value(sorted_dict, weights)
#             # print('after original_index = get_index_by_value(sorted_dict, weights) ', original_index)
#             GROUPS_weight.append(weights)
#             GROUPS_bucket_idx.append(original_index)
#             break
#         else:    # sum(weights) >= capacity
#             # if 1.25 * np.mean(res_tmp) < max(res_tmp) and np.mean(res_tmp) < 1.25 * min(res_tmp):
            
#             _, packs = backpack_split(weights, values, capacity)
#             # print('max_values ', max_values)
#             # print('packs ', packs)
#             res_tmp = np.array(weights)[packs[0]]
#             # print('res_tmp ', res_tmp)
            
#             # if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 2.5) :
#             # if len(packs[0]) > 3 or  (len(packs[0]) > 2 and max(res_tmp) > min(res_tmp)* 1.5) :
#             if len(packs[0]) > int(fanout/K)+1 and  max(res_tmp) > min(res_tmp)*1.5 :
#                 aa = min(packs[0]) # remove this aa from current group
#                 packs[0].remove(aa)
#                 res_tmp = np.array(weights)[packs[0]]
            
#             GROUPS_weight.append(list(res_tmp))
#             # print('GROUPS_weight ', GROUPS_weight)
#             # print('sorted_dict before get_index_by_value(sorted_dict, res_tmp)', sorted_dict)
#             original_index = get_index_by_value(sorted_dict, res_tmp)
#             # print('sorted_dict after get_index_by_value(sorted_dict, res_tmp)', sorted_dict)
#             GROUPS_bucket_idx.append(original_index)
#             # print()
#             # print("remove bucket_id in sorted dict ids: ",packs[0])
#             # print('original bucket_ids :, ', original_index)
#             # print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
#             # print()
#             # print('before remove, the weights list ',weights)
            
#             weights = remove_items_by_indices(weights, packs[0])
            
#             sorted_dict = remove_bucket_ids(sorted_dict, original_index)
            
            
#             # print('after remove one pack weights, ', weights)
#             values = weights
#             # print('sum of left weights ', sum(weights))
                
#     if len(weights)==1 :
#         if sum(weights)< capacity:
#             print('the last batch value is ', weights[0])
#             GROUPS_weight.append([weights[0]])
#         else:
#             print('error, OOM!')
            
#     return GROUPS_weight, GROUPS_bucket_idx


def split_cora(weights, values, capacity, fanout, K):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, _, sorted_dict = sort_with_original_index(weights)
    # print('split cora: sorted_dict ', sorted_dict)
    # print('weights after sort', sorted_values)
    # print()
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>=1:
        if sum(weights)< capacity: # sum of weights are smaller than the capacity, we can group them to one group
            original_index = get_index_by_value(sorted_dict, weights)
            # print('after original_index = get_index_by_value(sorted_dict, weights) ', original_index)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            break
        else:    # sum(weights) >= capacity
            # if 1.25 * np.mean(res_tmp) < max(res_tmp) and np.mean(res_tmp) < 1.25 * min(res_tmp):
            
            _, packs = backpack_split(weights, values, capacity)
            # print('max_values ', max_values)
            # print('packs ', packs)
            res_tmp = np.array(weights)[packs[0]]
            # print('res_tmp ', res_tmp)
            
            # if len(packs[0]) > 4 or  (len(packs[0]) > 3 and max(res_tmp) > min(res_tmp)* 2.5) :
            # if len(packs[0]) > 3 or  (len(packs[0]) > 2 and max(res_tmp) > min(res_tmp)* 1.5) :
            if len(packs[0]) > int(fanout/K)+1 and  max(res_tmp) > min(res_tmp)*1.5 :
                aa = min(packs[0]) # remove this aa from current group
                packs[0].remove(aa)
                res_tmp = np.array(weights)[packs[0]]
            
            GROUPS_weight.append(list(res_tmp))
            # print('GROUPS_weight ', GROUPS_weight)
            # print('sorted_dict before get_index_by_value(sorted_dict, res_tmp)', sorted_dict)
            original_index = get_index_by_value(sorted_dict, res_tmp)
            # print('sorted_dict after get_index_by_value(sorted_dict, res_tmp)', sorted_dict)
            GROUPS_bucket_idx.append(original_index)
            # print()
            # print("remove bucket_id in sorted dict ids: ",packs[0])
            # print('original bucket_ids :, ', original_index)
            # print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
            # print()
            # print('before remove, the weights list ',weights)
            
            weights = remove_items_by_indices(weights, packs[0])
            
            sorted_dict = remove_bucket_ids(sorted_dict, original_index)
            
            
            # print('after remove one pack weights, ', weights)
            values = weights
            # print('sum of left weights ', sum(weights))
                
    if len(weights)==1 :
        if sum(weights)< capacity:
            print('the last batch value is ', weights[0])
            GROUPS_weight.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS_weight, GROUPS_bucket_idx

def grouping_pre(adjust, weights, capacity, fanout, K):
	print('the grouping_fanout_cora called successfully')
    # weights is a list
	
	weights = [int(item * adjust) for item in weights]
	values = weights
	capacity = int(capacity * adjust)
	print('capacity ', capacity)

	print(' ')
	GROUPS_weight, GROUPS_bucket_idx = split_cora(weights, values, capacity, fanout, K) #####
	
	return GROUPS_weight, GROUPS_bucket_idx

def grouping_cora(adjust, weights, capacity, fanout, K):
	print('grouping float:  the grouping_fanout_cora called successfully')
    # weights is a dict
	degrees = weights.keys()
	weights = weights.values()
	weights = [int(item * adjust) for item in weights]
	values = weights
	capacity = int(capacity * adjust)
	# print('grouping cora: capacity ', capacity)

    
	print(' enter split_cora function')
	GROUPS_weight, GROUPS_bucket_idx = split_cora(weights, values, capacity, fanout, K) #####
	
	return GROUPS_weight, GROUPS_bucket_idx


# def grouping_cora(adjust, weights, capacity, fanout, K):
# 	print('grouping float:  the grouping_fanout_cora called successfully')
#     # weights is a dict
# 	degrees = weights.keys()
# 	weights = weights.values()
# 	weights = [int(item * adjust) for item in weights]
# 	values = weights
# 	capacity = int(capacity * adjust)
# 	# print('capacity ', capacity)
    
# 	print(' enter split_cora function')
# 	GROUPS_weight, GROUPS_bucket_idx = split_cora(weights, values, capacity, fanout, K) #####
	
# 	return GROUPS_weight, GROUPS_bucket_idx

def grouping_fanout_1(adjust, weights, capacity):
    
	if len(weights) == 49:
		weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
	if len(weights) == 24:
		weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	weights = [int(item * adjust) for item in weights]

	values =  weights 
	capacity = int(capacity * adjust)
	print('capacity ', capacity)


	GROUPS_weight, GROUPS_bucket_idx = split_all(weights, values, capacity)

	# for itm in GROUPS_weight:
	# 	tmp = [it/adjust for it in itm]
	# print()
	# print('bucket ids in each group ')
	# for itm in GROUPS_bucket_idx:
	# 	print(itm)
	# print()
	# total = [ len(GROUPS_bucket_idx[i]) for i in range(len(GROUPS_bucket_idx)) ]
	# print('each group length ')
	# print(total)
		
	# print(sum(total))
	return GROUPS_weight, GROUPS_bucket_idx

# def main():

#     # weights = [2, 3, 4, 5, 2, 1]
#     # values = [2, 3, 4, 5, 2, 1]
#     adjust = 1000
#     weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	
#     # weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
#     weights = [int(item * adjust) for item in weights]

#     print()
#     values =  weights 
#     capacity = 7 * adjust
#     # capacity = 9 * adjust
    
#     GROUPS_weight, GROUPS_bucket_idx = split_all(weights, values, capacity)
    
#     for itm in GROUPS_weight:
#         tmp = [it/adjust for it in itm]
#     print()
#     print('bucket ids in each group ')
#     for itm in GROUPS_bucket_idx:
#         print(itm)
#     print()
#     total = [ len(GROUPS_bucket_idx[i]) for i in range(len(GROUPS_bucket_idx)) ]
#     print('each group length ')
#     print(total)
    
#     print(sum(total))
    
# if __name__=='__main__':
	
# 	adjust = 1000
	# weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
	# weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881]
	# print(len(weights))
	# print(sum(weights))
	# capacity = 0.596
	# G_WEIGHTS, G_BUCKET_ID = grouping_fanout_1(adjust, weights, capacity)
	# print(G_BUCKET_ID)
	# if G_WEIGHTS:
	# 	for itm in G_WEIGHTS:
	# 			tmp = [it/adjust for it in itm]
	# 	print()
	# 	print('bucket ids in each group ')
	# 	for itm in G_BUCKET_ID:
	# 		print(itm)
	# 	print()
	# 	total = [ len(G_BUCKET_ID[i]) for i in range(len(G_BUCKET_ID)) ]
	# 	print('each group length ')
	# 	print(total)
            
	# 	print(sum(total))
	# 	print(len(total))
if __name__=='__main__':

	adjust = 1000
	estimated_mem =[0.4535364771051893, 0.36959890964869774, 0.3526514615214476, 0.3520529200944431, 0.3502430736172785, 0.36746639428808736, 0.37070838534849293, 0.3745401657375872, 0.3829395052158705, 0.3897006004864236, 0.39194492014904353, 0.4089296170047096, 0.41463856040323055, 0.4250790632909854, 0.4276191680715249, 0.433518633755968, 0.4386219996684242, 0.4403945473679008, 0.4539703137199875, 0.43765308297576344, 0.4485483741064351, 0.46742121625797345, 0.48099263700595957, 0.46918894709559855, 0.45248479346855597, 0.47846493770442255, 0.47077801986316126, 0.47288452210262105, 0.47387407820263094]
		
	# estimated_mem = [11.09133707869788, 7.949351660231331, 4.247930594170108, 3.767815723282392, 3.521911913357682, 3.3171698192934964, 3.1136675746243436, 2.9493490757618086, 2.783152518733855, 2.690315310282632, 2.4780320925231085, 2.4736405822131586, 2.424331795810457, 2.3423791134065306, 2.2508307132173453, 2.0888736283425056, 2.1393341709313427, 2.0144231711761864, 2.028492122175591, 1.8952586057017728, 1.9272310948984677, 1.7648288977543771, 1.8025470147872276, 1.7435488087790354, 1.624197941655698, 1.6781451757272114, 1.585553364875989, 1.5579015641599088, 1.508019266507371]
	capacity_imp = 5.9
	fanout = 30
	K = 2
	G_WEIGHTS, G_BUCKET_ID = grouping_fanout_arxiv(adjust, estimated_mem, capacity_imp, fanout, K)
	print('length ' ,len(G_BUCKET_ID))
	print(G_WEIGHTS)

	# estimated_mem = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881]
				
	# # capacity_imp = 0.753  # self.K = 11
	# # capacity_imp = 0.754  # self.K = 11
	# # capacity_imp = 0.758  # self.K = 11
	# # capacity_imp = 0.760  # self.K = 11
	# # capacity_imp = 0.761  # self.K = 11, 10
	# # capacity_imp = 0.764  # self.K = 11
	# # capacity_imp = 0.767  # self.K = 11
	# # capacity_imp = 0.777  # self.K = 13
	# # capacity_imp = 0.72  # self.K = 14
	# # capacity_imp = 0.757  # self.K = 12
	# # capacity_imp = 0.595  # self.K = 14
    # capacity_imp = 0.595  # self.K = 14
	# Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_products(adjust, estimated_mem, capacity = capacity_imp)
	# print()
	# print(len(G_BUCKET_ID_list))
	# # print(G_BUCKET_ID_list)
	# # print(Groups_mem_list)
	