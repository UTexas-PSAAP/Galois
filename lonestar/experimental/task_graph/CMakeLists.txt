#FIND_PACKAGE(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

app(test-task-graph test_task_graph.cpp EXP_OPT)

#target_include_directories(task-graph-test PRIVATE ${CUDA_INCLUDE_DIRS})
#target_include_directories(task-graph-test PRIVATE ${LIBPARLA_INCLUDE_DIR})

#target_link_libraries(task-graph-test ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
