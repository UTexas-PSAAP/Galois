#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <random>
#include <tuple>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/MorphGraph.h"
#include "galois/graphs/FileGraph.h"
#include "galois/runtime/Context.h"
#include "galois/substrate/PerThreadStorage.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

static const char* name = "Task Based Cholesky Factorization";
static const char* desc = "";
static const char* url = "cholesky_task";

static llvm::cl::opt<int> dim_size("dim_size", llvm::cl::desc("Number of rows/columns in main matrix."), llvm::cl::init(100));
static llvm::cl::opt<int> block_size("block_size", llvm::cl::desc("Number of rows/columns in each block."), llvm::cl::init(10));
static llvm::cl::opt<std::size_t> seed("seed", llvm::cl::desc("Seed used to generate symmetric positive definite matrix."), llvm::cl::init(0));

using task_data = std::tuple<std::atomic<std::size_t>, int, int, int, char>;
using task_label = std::tuple<int, int, int, char>;

// Use Morph_Graph to get this up and running quickly.
using Graph = galois::graphs::MorphGraph<task_data, void, true>;
using GNode = Graph::GraphNode;
using LMap = std::map<task_label, GNode>;

void generate_tasks(int nblocks, Graph &g, LMap &label_map) {
  for (std::size_t j = 0; j < nblocks; j++) {
    for (std::size_t k = 0; k < j; k++) {
      auto n = g.createNode(1, 0, j, k, 1);
      g.addNode(n);
      label_map[task_label(0, j, k, 1)] = n;
      g.addEdge(label_map[task_label(0, j, k, 4)], n);
    }
    auto n = g.createNode(j, 0, 0, j, 2);
    g.addNode(n);
    label_map[task_label(0, 0, j, 2)] = n;
    for (std::size_t k = 0; k < j; k++) {
      g.addEdge(label_map[task_label(0, j, k, 1)], n);
    }
    for (std::size_t i = j+1; i < nblocks; i++) {
      for (std::size_t k = 0; k < j; k++) {
        auto n = g.createNode(2, i, j, k, 3);
        g.addNode(n);
        label_map[task_label(i, j, k, 3)] = n;
        g.addEdge(label_map[task_label(0, j, k, 4)], n);
        g.addEdge(label_map[task_label(0, i, k, 4)], n);
      }
      auto n = g.createNode(j+1, 0, i, j, 4);
      g.addNode(n);
      label_map[task_label(0, i, j, 4)] = n;
      g.addEdge(label_map[task_label(0, 0, j, 2)], n);
      for (std::size_t k = 0; k < j; k++) {
        g.addEdge(label_map[task_label(i, j, k, 3)], n);
      }
    }
  }
}

void print_deps(Graph &g) {
  for (auto n : g) {
    auto &d = g.getData(n);
    std::cout << int(std::get<4>(d)) << "(" << std::get<1>(d) << ", " << std::get<2>(d) << ", " << std::get<3>(d) << "):" << std::endl;
    for (auto e : g.edges(n)) {
      auto &a = g.getData(g.getEdgeDst(e));
      std::cout << "    " << int(std::get<4>(a)) << "(" << std::get<1>(a) << ", " << std::get<2>(a) << ", " << std::get<3>(a) << ")" << std::endl;
    }
  }
}

decltype(auto) generate_symmetric_positive_definite(std::size_t size, std::size_t seed) {
  auto tmp = std::make_unique<double[]>(size * size);
  auto out = std::make_unique<double[]>(size * size);
  std::mt19937 gen{seed};
  std::uniform_real_distribution<> dis(-1., 1.);
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j < size; j++) {
      tmp.get()[j + size * i] = dis(gen);
      //std::cout << tmp.get()[j + size * i] << "  ";
    }
    //std::cout << std::endl;
  }
  //std::cout << std::endl;
  // This really just computes tmp.T @ tmp.
  // There's no guarantee this whole thing will fit nicely into the gpu memory
  // though, and we aren't linking against any BLAS other than CUBLAS, so
  // we'll just do slow three nested loops here.
  for (std::size_t i = 0; i < size * size; i++) {
    out.get()[i] = 0.;
  }
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j < size; j++) {
      for (std::size_t k = 0; k < size; k++) {
        out.get()[j + size * i] += tmp.get()[k + size * j] * tmp.get()[k + size * i];
      }
      //std::cout << out.get()[j + size * i] << "  ";
    }
    //std::cout << std::endl;
  }
  //std::cout << std::endl;
  return out;
}

void print_cublas_status(cublasStatus_t stat) {
  if (stat == CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS_STATUS_SUCCESS" << std::endl;
  } else if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
    std::cout << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
  } else if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
    std::cout << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl;
  } else if (stat == CUBLAS_STATUS_INVALID_VALUE) {
    std::cout << "CUBLAS_STATUS_INVALID_VALUE" << std::endl;
  } else if (stat == CUBLAS_STATUS_ARCH_MISMATCH) {
    std::cout << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
  } else if (stat == CUBLAS_STATUS_MAPPING_ERROR) {
    std::cout << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl;
  } else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
    std::cout << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
  } else if (stat == CUBLAS_STATUS_INTERNAL_ERROR) {
    std::cout << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;
  } else if (stat == CUBLAS_STATUS_NOT_SUPPORTED) {
    std::cout << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl;
  } else if (stat == CUBLAS_STATUS_LICENSE_ERROR) {
    std::cout << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl;
  }
}

using PSChunk = galois::worklists::PerSocketChunkFIFO<16>;

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  int nblocks = dim_size / block_size;
  if (dim_size % block_size != 0) {
    GALOIS_DIE("Blocks that do not evenly divide the array dimensions are not yet supported");
  }

  auto spd_manager = generate_symmetric_positive_definite(dim_size, seed);
  auto spd = spd_manager.get();

  Graph g;
  LMap label_map;
  std::mutex map_lock;
  generate_tasks(nblocks, g, label_map);
  //print_deps(g);

  // Now execute the tasks.
  //cublasHandle_t handle;
  galois::substrate::PerThreadStorage<cublasHandle_t> handles;
  galois::substrate::PerThreadStorage<cusolverDnHandle_t> cusolver_handles;
  galois::substrate::PerThreadStorage<int> lwork_sizes;
  galois::substrate::PerThreadStorage<double*> b0s;
  galois::substrate::PerThreadStorage<double*> b1s;
  galois::substrate::PerThreadStorage<double*> b2s;
  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    auto stat = cublasCreate(handles.getLocal());
    if (stat != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cublas.");
    }
    auto stat2 = cusolverDnCreate(cusolver_handles.getLocal());
    if (stat2 != CUSOLVER_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cusolver.");
    }
    auto stat3 = cudaMalloc(b0s.getLocal(), static_cast<std::size_t>(block_size * block_size));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    // Maybe make this one slightly larger so it can be used as an "lwork" buffer for dpotrf.
    cusolverDnDpotrf_bufferSize(*cusolver_handles.getLocal(), CUBLAS_FILL_MODE_LOWER, block_size, *b0s.getLocal(), block_size, lwork_sizes.getLocal());
    stat3 = cudaMalloc(b1s.getLocal(), static_cast<std::size_t>(std::max(*lwork_sizes.getLocal(), block_size * block_size)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    stat3 = cudaMalloc(b2s.getLocal(), static_cast<std::size_t>(block_size * block_size));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
  });
  //auto stat = cublasCreate(&handle);
  //if (stat != CUBLAS_STATUS_SUCCESS) {
  //  GALOIS_DIE("Failed to initialize cublas.");
  //}
  auto locks = std::make_unique<galois::runtime::Lockable[]>(nblocks * nblocks);

  galois::for_each(
    galois::iterate({label_map[task_label(0, 0, 0, 2)]}),
    [&](GNode n, auto& ctx) {
      auto &d = g.getData(n);
      auto task_type = std::get<4>(d);
      if (task_type == 1) {
        auto j = std::get<2>(d);
        auto k = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::WRITE);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * k]), galois::MethodFlag::READ);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        auto b2 = *b2s.getLocal();
        auto handle = *handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j + j * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (1)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j + k * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (2)");
        double alpha = -1., beta = 1.;
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, block_size, block_size, block_size, &alpha, b1, block_size, b1, block_size, &beta, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dgemm operation failed. (3)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b1, block_size, spd + j + k * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (4)");
      } else if (task_type == 2) {
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::WRITE);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        auto cusolver_handle = *cusolver_handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j + j * dim_size, dim_size, b0, block_size);
        print_cublas_status(stat);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (5)");
        int info = 0;
        auto stat2 = cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, block_size, b0, block_size, b1, *lwork_sizes.getLocal(), &info);
        if (stat2 != CUSOLVER_STATUS_SUCCESS) GALOIS_DIE("Cholesky block solve failed. (6)");
        if (info != 0) {
          if (info < 0) GALOIS_DIE("Incorrect parameter passed to dpotrf. (7)");
          if (info > 0) GALOIS_DIE("Not positive definite. (8)");
        }
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + j + j * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (9)");
      } else if (task_type == 3) {
        auto i = std::get<1>(d);
        auto j = std::get<2>(d);
        auto k = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * j]), galois::MethodFlag::WRITE);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * k]), galois::MethodFlag::READ);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * k]), galois::MethodFlag::READ);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        auto b2 = *b2s.getLocal();
        auto handle = *handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i + j * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (10)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i + k * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (11)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j + k * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (12)");
        double alpha = -1, beta = 1;
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, block_size, block_size, block_size, &alpha, b1, block_size, b2, block_size, &beta, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dgemm operation failed. (13)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + i + j * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (14)");
      } else if (task_type == 4) {
        auto i = std::get<2>(d);
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::READ);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * j]), galois::MethodFlag::WRITE);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        auto handle = *handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j + j * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (15)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i + j * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (16)");
        double alpha = 1.;
        stat = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, block_size, block_size, &alpha, b0, block_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dtrsm operation failed. (17)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b1, block_size, spd + i + j * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (18)");
      } else {
        GALOIS_DIE("Unrecognized task type.");
      }

      for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        auto dst = g.getEdgeDst(e);
        if (0 == --std::get<0>(g.getData(dst, galois::MethodFlag::UNPROTECTED))) {
          ctx.push(dst);
        }
      }
    },
    galois::loopname("cholesky_tasks"), galois::wl<PSChunk>()
  );

  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    auto stat = cudaFree(reinterpret_cast<void*>(*b0s.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for blocks.");
    }
    stat = cudaFree(reinterpret_cast<void*>(*b1s.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for blocks.");
    }
    stat = cudaFree(reinterpret_cast<void*>(*b2s.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for blocks.");
    }
    auto stat2 = cusolverDnDestroy(*cusolver_handles.getLocal());
    if (stat2 != CUSOLVER_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to free cusolver resources.");
    }
    auto stat3 = cublasDestroy(*handles.getLocal());
    if (stat3 != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to free cublas resources.");
    }
  });
  //cublasDestroy(handle);
  
  return 0;
}
