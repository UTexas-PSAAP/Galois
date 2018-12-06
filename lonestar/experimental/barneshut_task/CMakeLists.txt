FIND_PACKAGE(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

app(barneshut-task barneshut.cpp EXP_OPT)

target_include_directories(barneshut-task PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(barneshut-task PRIVATE ${LIBPARLA_INCLUDE_DIR})

target_link_libraries(barneshut-task ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
