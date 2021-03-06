add_library(distbench STATIC src/DistBenchStart.cpp)
target_include_directories(distbench PUBLIC 
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# turn on cuda for distbench as well
if(ENABLE_HETERO_GALOIS)
  target_compile_definitions(distbench PRIVATE __GALOIS_HET_CUDA__=1)
endif()

target_link_libraries(distbench galois_shmem gllvm)

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/libdist/include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

if(ENABLE_HETERO_GALOIS)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

function(distApp name)
  app(${name} ${name}/${name}.cpp DISTSAFE)
  target_link_libraries(${name} distbench galois_dist)
  target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()

  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name} PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name} ${name}_cuda)

    cuda_add_library(${name}_cuda ${name}/${name}_cuda.cu)
    target_link_libraries(${name}_cuda galois_gpu)
  endif()

  app(${name}_async ${name}/${name}.cpp DISTSAFE)
  target_link_libraries(${name}_async distbench galois_dist_async)
  target_include_directories(${name}_async PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  target_compile_definitions(${name}_async PRIVATE __GALOIS_HET_ASYNC__=1)
  if(USE_BARE_MPI)
    target_compile_definitions(${name}_async PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()

  if(ENABLE_HETERO_GALOIS)
    target_compile_definitions(${name}_async PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name}_async ${name}_cuda)

    cuda_add_library(${name}_async_cuda ${name}/${name}_cuda.cu)
    target_link_libraries(${name}_async_cuda galois_gpu)
  endif()
endfunction()

function(distAppNoGPU name)
  app(${name} ${name}/${name}.cpp DISTSAFE)
  target_link_libraries(${name} distbench galois_dist)
  target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()
endfunction()

distAppNoGPU(bc_level)
distAppNoGPU(weighted_bc)

distApp(bfs_pull)
distApp(bfs_push)

distApp(cc_pull)
distApp(cc_push)

distApp(kcore_push)
distApp(kcore_pull)

distApp(pagerank_pull)
distApp(pagerank_push)

distAppNoGPU(sgd)

distApp(sssp_pull)
distApp(sssp_push)

distAppNoGPU(partition)

distAppNoGPU(bc_mr)
