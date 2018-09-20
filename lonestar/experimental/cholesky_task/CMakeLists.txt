# Hard code the paths since cmake is apparently unable to find our repackaged cuda 9 without additional help.

link_directories(/org/centers/cdgc/cuda/cuda-9.0/lib64)

app(cholesky-task cholesky.cpp EXP_OPT)

target_include_directories(cholesky-task PRIVATE /org/centers/cdgc/cuda/cuda-9.0/include)

target_link_libraries(cholesky-task cudart cublas)
