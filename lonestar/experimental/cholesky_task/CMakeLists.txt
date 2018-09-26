FIND_PACKAGE(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

app(cholesky-task cholesky.cpp EXP_OPT)

target_include_directories(cholesky-task PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cholesky-task PRIVATE ${LIBPARLA_INCLUDE_DIR})

target_link_libraries(cholesky-task ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
