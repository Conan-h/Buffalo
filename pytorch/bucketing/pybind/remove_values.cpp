// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <vector>

// namespace py = pybind11;

// std::vector<int> remove_values(std::vector<int> input_vector, int value_to_remove) {
//     input_vector.erase(std::remove(input_vector.begin(), input_vector.end(), value_to_remove), input_vector.end());
//     return input_vector;
// }

// PYBIND11_MODULE(remove_values, m) {
//     m.def("remove_values", &remove_values, "Remove values from a vector");
// }

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <vector>
// #include <algorithm>

// namespace py = pybind11;

// std::vector<int> remove_values(std::vector<int> data, std::vector<int> values_to_remove) {
//     std::vector<int> result;

//     for (auto& elem : data) {
//         if (std::find(values_to_remove.begin(), values_to_remove.end(), elem) == values_to_remove.end()) {
//             result.push_back(elem);
//         }
//     }
//     return result;
// }

// PYBIND11_MODULE(remove_values, m) {
//     m.def("remove_values", &remove_values, py::arg("data"), py::arg("values_to_remove"));
// }


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

std::vector<int> remove_values(std::vector<int> data, std::vector<int> values_to_remove) {
    std::vector<int> result;
    
    // Build an unordered_map from values_to_remove for O(1) lookup.
    std::unordered_map<int, bool> to_remove;
    for (const auto& val : values_to_remove) {
        to_remove[val] = true;
    }

    for (const auto& elem : data) {
        if (!to_remove[elem]) {
            result.push_back(elem);
        }
    }
    return result;
}

PYBIND11_MODULE(remove_values, m) {
    m.def("remove_values", &remove_values, py::arg("data"), py::arg("values_to_remove"));
}
