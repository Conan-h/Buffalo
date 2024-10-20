#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <unordered_set>


namespace py = pybind11;

// // // std::vector<long> remove_values(const std::vector<long>& mini_batch_src_global, const std::vector<long>& global_output_nid){
// // std::vector<long> remove_values(std::vector<long> from, const std::vector<long>& values_to_remove) {
// //     // We're now operating on a copy of `from` instead of the original
// //     auto new_end = std::remove_if(from.begin(), from.end(), [&](long value) {
// //         return std::find(values_to_remove.begin(), values_to_remove.end(), value) != values_to_remove.end();
// //     });
// //     from.erase(new_end, from.end());
// //     return from;
// // }
// std::vector<int> remove_values(std::vector<int> data, std::vector<int> values_to_remove) {
//     std::unordered_set<int> to_remove(values_to_remove.begin(), values_to_remove.end());

//     std::vector<std::vector<int>> private_results(omp_get_max_threads());

//     #pragma omp parallel for
//     for (size_t i = 0; i < data.size(); ++i) {
//         int value = data[i];
//         if (!to_remove.count(value)) {
//             private_results[omp_get_thread_num()].push_back(value);
//         }
//     }

//     std::vector<int> result;
//     for (auto& private_result : private_results) {
//         result.insert(result.end(), private_result.begin(), private_result.end());
//     }

//     return result;
// }

std::vector<long> remove_values(std::vector<long> data, std::vector<long> values_to_remove) {
    std::unordered_set<long> to_remove(values_to_remove.begin(), values_to_remove.end());

    std::vector<std::vector<long>> private_results(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        long value = data[i];
        if (!to_remove.count(value)) {
            private_results[omp_get_thread_num()].push_back(value);
        }
    }

    std::vector<long> result;
    for (auto& private_result : private_results) {
        result.insert(result.end(), private_result.begin(), private_result.end());
    }

    return result;
}
std::vector<std::vector<long>> gen_tails(const std::vector<std::vector<long>>& src_long_list,
                                        const std::vector<std::vector<long>>& global_batched_nids_list) {
    std::vector<std::vector<long>> tails_list;
    tails_list.reserve(src_long_list.size());

    #pragma omp parallel for
    for (int i = 0; i < src_long_list.size(); ++i) {
        const auto& mini_batch_src_global = src_long_list[i];
        const auto& global_output_nid = global_batched_nids_list[i];

        auto r_ = remove_values(mini_batch_src_global, global_output_nid);

        #pragma omp critical
        tails_list.push_back(r_);
    }

    return tails_list;
}

// std::vector<std::vector<long>> gen_tails(const std::vector<std::vector<long>>& src_long_list, const std::vector<std::vector<long>>& global_batched_nids_list) {
//     std::vector<std::vector<long>> tails_list(src_long_list.size());

//     #pragma omp parallel for
//     for (size_t i = 0; i < src_long_list.size(); ++i) {
//         auto mini_batch_src_global = src_long_list[i];  // Make a copy
//         auto global_output_nid = global_batched_nids_list[i];

//         auto r_ = remove_values(mini_batch_src_global, global_output_nid);
//         tails_list[i] = r_;
//     }

//     return tails_list;
// }

PYBIND11_MODULE(gen_tails, m) {
    m.def("gen_tails", &gen_tails, "A function that generates tails");
}
