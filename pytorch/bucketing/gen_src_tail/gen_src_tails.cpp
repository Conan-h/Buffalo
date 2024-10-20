#include <vector>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <omp.h>
#include <torch/torch.h>

std::vector<int> remove_duplicates(const std::vector<int>& data) {
    std::unordered_set<int> unique_items;
    std::vector<int> result;

    for (const auto& item : data) {
        if (unique_items.find(item) == unique_items.end()) {
            result.push_back(item);
            unique_items.insert(item);
        }
    }

    return result;
}

std::vector<int> remove_values(const std::vector<int>& data, const std::vector<int>& values_to_remove) {
    std::unordered_set<int> to_remove(values_to_remove.begin(), values_to_remove.end());

    std::vector<std::vector<int>> private_results(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        long value = data[i];
        if (!to_remove.count(value)) {
            private_results[omp_get_thread_num()].push_back(value);
        }
    }

    std::vector<int> result;
    for (auto& private_result : private_results) {
        result.insert(result.end(), private_result.begin(), private_result.end());
    }

    return result;
}

std::pair<std::unordered_map<int, std::vector<int>>, std::unordered_map<int, std::vector<int>>>
gen_src_tails(const std::vector<std::vector<std::vector<int>>>& local_in_edges_tensor_list,
              const std::vector<std::vector<int>>& global_batched_nids_list,
              const std::vector<int>& induced_src_vec,
              const std::vector<int>& eids_global_vec) {

    // Generate induced_src and eids_global mapping using PyTorch tensors
    torch::Tensor induced_src_tensor = torch::from_blob(const_cast<int*>(induced_src_vec.data()), {induced_src_vec.size()}, torch::kInt);
    std::map<int, int> induced_src;
    for (int i = 0; i < induced_src_tensor.size(0); i++) {
        induced_src[i] = induced_src_tensor[i].item<int>();
    }

    torch::Tensor eids_global_tensor = torch::from_blob(const_cast<int*>(eids_global_vec.data()), {eids_global_vec.size()}, torch::kInt);
    std::map<int, int> eids_global;
    for (int i = 0; i < eids_global_tensor.size(0); i++) {
        eids_global[i] = eids_global_tensor[i].item<int>();
    }

    std::unordered_map<int, std::vector<int>> eids_list;
    std::unordered_map<int, std::vector<int>> src_long_list;

    for (size_t i = 0; i < local_in_edges_tensor_list.size(); ++i) {
        const auto& local_in_edges_tensor = local_in_edges_tensor_list[i];

        // Access elements directly from std::vector
        const std::vector<int>& mini_batch_src_local_vec = local_in_edges_tensor[0];

        std::vector<int> mini_batch_src_global;
        for (const auto& local : mini_batch_src_local_vec) {
            auto it = induced_src.find(local);
            if (it == induced_src.end()) {
                throw std::runtime_error("Key not found in induced_src: " + std::to_string(local));
            }

            mini_batch_src_global.push_back(it->second);
        }

        std::vector<int> r_ = remove_values(mini_batch_src_global, global_batched_nids_list[i]);
        const std::vector<int>& eid_local_list = local_in_edges_tensor[2];

        std::vector<int> global_eid_tensor;
        for (const auto& local : eid_local_list) {
            auto it = eids_global.find(local);
            if (it == eids_global.end()) {
                throw std::runtime_error("Key not found in eids_global: " + std::to_string(local));
            }
            global_eid_tensor.push_back(it->second);
        }

        eids_list[i] = global_eid_tensor;
        src_long_list[i] = r_;
    }

    return std::make_pair(eids_list, src_long_list);
}
