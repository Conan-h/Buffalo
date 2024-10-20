import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        print('The number of model layers: ', self.info.num_layers)
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            print(f' ')

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
def asnumpy(input):
	return input.cpu().detach().numpy()

def equal (x,y):
	return x == y

def nonzero_1d(input):
	x = torch.nonzero(input, as_tuple=False).squeeze()
	return x if x.dim() == 1 else x.view(-1)

def gather_row(data, row_index):
	return torch.index_select(data, 0, row_index.long())

def zerocopy_from_numpy(np_array):
	return torch.as_tensor(np_array)

def my_sort_1d(val):  # add new function here, to replace torch.sort()
		idx_dict = dict(zip(range(len(val)),val.tolist())) #####
		sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1])) ######
		sorted_val = torch.tensor(list(sorted_res.values())).to(val.device)  ######
		idx = torch.tensor(list(sorted_res.keys())).to(val.device) ######
		return sorted_val, idx

def split_list(input_list, k):
    avg = len(input_list) // k
    remainder = len(input_list) % k
    return [input_list[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(k)]





def _bucketing( val):
		# val : local index degrees 
		# sorted_val, idx = torch.sort(val)
		# print('degrees val')
		# print(val)
		sorted_val, idx = val.sort(stable=True)
		# print('sorted_val ', sorted_val)
		# print('idx ', idx)
		# sorted_val, idx = my_sort_1d(val) # keep the nodes in global order

		unique_val = asnumpy(torch.unique(sorted_val))
		bkt_idx = []
		for v in unique_val:
			bool_idx = (sorted_val == v)
			eqidx = torch.nonzero(bool_idx, as_tuple=False).squeeze().view(-1)
			# eqidx = nonzero_1d(equal(sorted_val, v))
			# bkt_idx.append(gather_row(idx, eqidx))
			local_nids = torch.index_select(idx, 0, eqidx.long())
			bkt_idx.append(local_nids)
			
		def bucketor(data):
			bkts = [gather_row(data, idx) for idx in bkt_idx]
			return bkts
		return unique_val, bucketor 

def get_in_degree_bucketing(layer_block):

		degs = layer_block.in_degrees()
		org_src=layer_block.srcdata['_ID']
		# print('get_in_degree_bucketing src global nid ', layer_block.srcdata['_ID'])
		# print('get_in_degree_bucketing dst global nid ', layer_block.dstdata['_ID'])
		# print('get_in_degree_bucketing corresponding in degs', degs)
		nodes = layer_block.dstnodes() # local dst nid (e.g. in full batch layer block)
		total_output_nids = 0
		# degree bucketing
		unique_degs, bucketor = _bucketing(degs)
		bkt_nodes = []
		for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
			if deg == 0:
				# skip reduce function for zero-degree nodes
				continue
			global_node_bkt = org_src[node_bkt] # local nid idx
			bkt_nodes.append(global_node_bkt) # global nid idx
			# print('len(bkt) ', len(node_bkt))
			# print('global bkt nids ', node_bkt)
			# total_output_nids += len(node_bkt)
		# print('total indegree bucketing result , ', total_output_nids)
		return bkt_nodes 

