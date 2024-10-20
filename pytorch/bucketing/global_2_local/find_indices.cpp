#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>


namespace py = pybind11;

std::vector<std::vector<int>> find_indices(const std::vector<int>& src_nid, const std::vector<std::vector<int>>& nested_list) {
    std::unordered_map<int, int> src_nid_index;
    for (int i = 0; i < src_nid.size(); ++i) {
        src_nid_index[src_nid[i]] = i;
    }

    std::vector<std::vector<int>> indices(nested_list.size());

    #pragma omp parallel for
    for (int i = 0; i < nested_list.size(); ++i) {
        // std::cout << "Thread ID: " << omp_get_thread_num() << "\n";  
        const auto& sublist = nested_list[i];
        std::vector<int> sublist_indices(sublist.size());

        for (int j = 0; j < sublist.size(); ++j) {
            if(src_nid_index.find(sublist[j]) != src_nid_index.end()){
                sublist_indices[j] = src_nid_index[sublist[j]];
            }
            else{
                std::cout << "Key not found in src_nid_index: " << sublist[j] << "\n";
                sublist_indices[j] = -1;  // Or some other default value
            }
        }

        indices[i] = sublist_indices;
        
    }
    return indices;
}

PYBIND11_MODULE(find_indices, m) {
    m.def("find_indices", &find_indices, "A function which finds the indices of a list of items in a src_nid");
}
