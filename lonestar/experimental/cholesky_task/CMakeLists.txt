FIND_PACKAGE(CUDA REQUIRED)

app(cholesky-task cholesky.cpp EXP_OPT)

target_include_directories(cholesky-task PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(cholesky-task ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
