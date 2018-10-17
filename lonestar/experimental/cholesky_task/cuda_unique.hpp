#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace cu {

template <typename T>
inline void simple_cuda_free(T *p) {
  /* TODO: handle error checking as well. */
  cudaFree(p);
}

template <typename T>
decltype(auto) make_unique(size_t size) {
  T *p;
  cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T) * size);
  return std::unique_ptr<T[], decltype(&simple_cuda_free<T>)>(
    p, &simple_cuda_free<T>);
}

} // namespace cu
