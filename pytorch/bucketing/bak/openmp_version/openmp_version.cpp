#include <iostream>
#ifdef _OPENMP
    #include <omp.h>
#endif

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP version (YYYYMM): " << _OPENMP << "\n";
    std::cout << "OpenMP library version: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not supported.\n";
#endif
    return 0;
}
