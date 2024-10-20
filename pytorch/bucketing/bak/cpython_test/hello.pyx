# distutils: language=c++

cdef extern from "<iostream>" namespace "std":
    void cout(string)

def say_hello():
    cout("Hello from C++!")
