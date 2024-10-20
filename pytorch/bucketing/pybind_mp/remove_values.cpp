// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <vector>
// #include <unordered_map>
// #include <omp.h>

// namespace py = pybind11;

// std::vector<int> remove_values(std::vector<int> data, std::vector<int> values_to_remove) {
//     // Build an unordered_map from values_to_remove for O(1) lookup.
//     std::unordered_map<int, bool> to_remove;
//     for (const auto& val : values_to_remove) {
//         to_remove[val] = true;
//     }

//     // Create a vector to hold the result
//     std::vector<int> result(data.size());
    
//     #pragma omp parallel for
//     for (size_t i = 0; i < data.size(); ++i) {
//         if (!to_remove[data[i]]) {
//             result[i] = data[i];
//         }
//     }

//     // Remove elements that were not assigned in parallel loop
//     result.erase(std::remove(result.begin(), result.end(), 0), result.end());

//     return result;
// }

// PYBIND11_MODULE(remove_values, m) {
//     m.def("remove_values", &remove_values, py::arg("data"), py::arg("values_to_remove"));
// }
// the above code can't make sure the original order



#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

std::vector<int> remove_values(std::vector<int> data, std::vector<int> values_to_remove) {
    std::unordered_set<int> to_remove(values_to_remove.begin(), values_to_remove.end());

    std::vector<std::vector<int>> private_results(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        int value = data[i];
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

PYBIND11_MODULE(remove_values, m) {
    m.def("remove_values", &remove_values, py::arg("data"), py::arg("values_to_remove"));
}
