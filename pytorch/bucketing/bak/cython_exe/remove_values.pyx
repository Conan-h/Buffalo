# distutils: language=c++
# distutils: sources=remove_values.cpp

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set

cdef extern from "remove_values.cpp":
    vector[string] remove_values(vector[string] long_list, unordered_set[string] values_to_remove)

def remove_values_python(list long_list, set values_to_remove):
    cdef vector[string] long_list_cpp = long_list
    cdef unordered_set[string] values_to_remove_cpp = values_to_remove
    cdef vector[string] new_list_cpp

    new_list_cpp = remove_values(long_list_cpp, values_to_remove_cpp)

    return [item for item in new_list_cpp]
