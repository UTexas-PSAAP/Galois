#include <algorithm>
#include <atomic>
#include <cmath>
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

static llvm::cl::opt<int> dim_size("dim_size", llvm::cl::desc("Number of rows/columns in main matrix."), llvm::cl::init(9));
static llvm::cl::opt<int> block_size("block_size", llvm::cl::desc("Number of rows/columns in each block."), llvm::cl::init(3));
static llvm::cl::opt<int> seed("seed", llvm::cl::desc("Seed used to generate symmetric positive definite matrix."), llvm::cl::init(0));
static llvm::cl::opt<double> tolerance("tolerance", llvm::cl::desc("Tolerance used to check that the results are correct."), llvm::cl::init(.001));

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
    }
  }
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
    }
  }
  return out;
}

// Check against a naive cholesky factorization.
void check_correctness(double *result, std::size_t size, std::size_t seed, double tol) {
  // Just re-generate the input. In most cases the dominating cost will
  // be the naive cholesky decomposition below anyway.
  auto orig_manager = generate_symmetric_positive_definite(size, seed);
  auto orig = orig_manager.get();

  // Now check that the result satisfies L @ L^T = A where A is the input.
  // Just use the infinity norm for simplicity here. The main purpose of
  // the verification is to check that the right calls were made in the tasks.
  // NOTE: we're only checking the lower triangular portion of the array.
  // The blocked version actually overwrites some of the superdiagonal
  // elements with new values. That will cause some differences between
  // the results of a naive version and the blocke version, but the
  // relevant output is in the lower-triangular portion of the array,
  // so the differences in the rest of it don't matter.

  // First get the infinity norm of the array so we can compute the relative error.
  double nrm = 0.;
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j < i; j++) {
      nrm = std::max(nrm, std::abs(orig[i + j * size]));
    }
  }
  // Now compute the relative error at each element and check that
  // it's within the tolerance.
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j <= i; j++) {
      double reconstructed = 0.;
      for (std::size_t k = 0; k <= j; k++) {
        reconstructed += result[i + k * size] * result[j + k * size];
      }
      auto rel_err = std::abs(orig[i + j * size] - reconstructed) / nrm;
      if (rel_err >= tol) {
        std::stringstream ss;
        ss << "Verification failed at element (" << i << "," << j << ") with difference " << rel_err << ".";
        GALOIS_DIE(ss.str());
      }
    }
  }
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
  } else {
    std::cout << "Unknown cublas status." << std::endl;
  }
}

void print_cusolver_status(cusolverStatus_t stat) {
  if (stat == CUSOLVER_STATUS_SUCCESS) {
    std::cout << "CUSOLVER_STATUS_SUCCESS" << std::endl;
  } else if (stat == CUSOLVER_STATUS_NOT_INITIALIZED) {
    std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED" << std::endl;
  } else if (stat == CUSOLVER_STATUS_ALLOC_FAILED) {
    std::cout << "CUSOLVER_STATUS_ALLOC_FAILED" << std::endl;
  } else if (stat == CUSOLVER_STATUS_INVALID_VALUE) {
    std::cout << "CUSOLVER_STATUS_INVALID_VALUE" << std::endl;
  } else if (stat == CUSOLVER_STATUS_ARCH_MISMATCH) {
    std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH" << std::endl;
  } else if (stat == CUSOLVER_STATUS_EXECUTION_FAILED) {
    std::cout << "CUSOLVER_STATUS_EXECUTION_FAILED" << std::endl;
  } else if (stat == CUSOLVER_STATUS_INTERNAL_ERROR) {
    std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR" << std::endl;
  } else if (stat == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    std::cout << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << std::endl;
  } else {
    std::cout << "Unknown cusolver status." << std::endl;
  }
}

void print_mat(double *m, std::size_t rows, std::size_t cols, std::ptrdiff_t row_stride) {
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      std::cout << m[i + j * row_stride] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
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
  //std::cout.precision(19);
  //std::cout << "generated input:" << std::endl;
  //print_mat(spd, dim_size, dim_size, dim_size);

  Graph g;
  LMap label_map;
  std::mutex map_lock;
  generate_tasks(nblocks, g, label_map);
  //print_deps(g);

  // Now set up the needed resources for each thread.
  galois::substrate::PerThreadStorage<cublasHandle_t> handles(nullptr);
  galois::substrate::PerThreadStorage<cusolverDnHandle_t> cusolver_handles(nullptr);
  //galois::substrate::PerThreadStorage<cudaStream_t> cusolver_streams(nullptr);
  galois::substrate::PerThreadStorage<int> lwork_sizes;
  galois::substrate::PerThreadStorage<double*> b0s;
  galois::substrate::PerThreadStorage<double*> b1s;
  galois::substrate::PerThreadStorage<double*> b2s;
  galois::substrate::PerThreadStorage<double*> lworks;
  galois::substrate::PerThreadStorage<int*> dev_infos;
  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    int devices = 0;
    auto stat3 = cudaGetDeviceCount(&devices);
    if (devices < nthreads) {
      GALOIS_DIE("The number of threads desired is greater than the number of cuda devices available.");
    }
    stat3 = cudaSetDevice(tid);
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate one device per thread.");
    }
    auto stat = cublasCreate(handles.getLocal());
    if (stat != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cublas.");
    }
    auto stat2 = cusolverDnCreate(cusolver_handles.getLocal());
    if (stat2 != CUSOLVER_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cusolver.");
    }
    /*stat3 = cudaStreamCreateWithFlags(cusolver_streams.getLocal(), cudaStreamNonBlocking);
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to acquire cuda stream.");
    }
    stat2 = cusolverDnSetStream(*cusolver_handles.getLocal(), *cusolver_streams.getLocal());
    if (stat2 != CUSOLVER_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to set stream for cusolver.");
    }*/
    stat2 = cusolverDnDpotrf_bufferSize(*cusolver_handles.getLocal(), CUBLAS_FILL_MODE_LOWER, block_size, *b0s.getLocal(), block_size, lwork_sizes.getLocal());
    if (stat2 != cudaSuccess) {
      GALOIS_DIE("Failed to determine lwork size for cusolver dpotrf.");
    }
    stat3 = cudaMalloc(lworks.getLocal(), static_cast<std::size_t>(((*lwork_sizes.getLocal()) * sizeof(double))));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU workspace for per-block Cholesky factorization.");
    }
    stat3 = cudaMalloc(dev_infos.getLocal(), sizeof(int));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU int for Cholesky call status.");
    }
    stat3 = cudaMalloc(b0s.getLocal(), static_cast<std::size_t>(block_size * block_size * sizeof(double)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    stat3 = cudaMalloc(b1s.getLocal(), static_cast<std::size_t>(block_size * block_size * sizeof(double)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    stat3 = cudaMalloc(b2s.getLocal(), static_cast<std::size_t>(block_size * block_size * sizeof(double)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
  });

  auto locks = std::make_unique<galois::runtime::Lockable[]>(nblocks * nblocks);

  // Now execute the tasks.
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
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (1)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + k * block_size * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (2)");
        double alpha = -1., beta = 1.;
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, block_size, block_size, block_size, &alpha, b1, block_size, b1, block_size, &beta, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dgemm operation failed. (3)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + j * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (4)");
      } else if (task_type == 2) {
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::WRITE);
        auto b0 = *b0s.getLocal();
        auto lwork = *lworks.getLocal();
        auto lwork_size = *lwork_sizes.getLocal();
        auto cusolver_handle = *cusolver_handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (5)");
        int info = 0;
        auto stat2 = cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, block_size, b0, block_size, lwork, lwork_size, *dev_infos.getLocal());
        if (stat2 != CUSOLVER_STATUS_SUCCESS) GALOIS_DIE("Cholesky block solve failed. (6)");
        auto stat3 = cudaMemcpy(&info, *dev_infos.getLocal(), sizeof(int), cudaMemcpyDeviceToHost);
        if (stat3 != cudaSuccess) GALOIS_DIE("Recieve status after Cholesky on block failed. (7)");
        if (info != 0) {
          std::cout << info << std::endl;
          std::stringstream ss;
          if (info < 0) {
            ss << "Parameter " << -info << " incorrect when passed to dpotrf. (8)";
            GALOIS_DIE(ss.str());
          }
          if (info > 0) {
            ss << "Diagonal block " << j << " not positive definite at minor " << info << " during per-block Cholesky computation. (9)";
            GALOIS_DIE(ss.str());
          }
        }
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + j * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (10)");
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
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (11)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i * block_size + k * block_size * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (12)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + k * block_size * dim_size, dim_size, b2, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (13)");
        double alpha = -1, beta = 1;
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, block_size, block_size, block_size, &alpha, b1, block_size, b2, block_size, &beta, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dgemm operation failed. (14)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + i * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (15)");
      } else if (task_type == 4) {
        auto i = std::get<2>(d);
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::READ);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * j]), galois::MethodFlag::WRITE);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        auto handle = *handles.getLocal();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (16)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i * block_size + j * block_size * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (17)");
        double alpha = 1.;
        stat = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, block_size, block_size, &alpha, b0, block_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dtrsm operation failed. (18)");
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b1, block_size, spd + i * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (19)");
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

  //std::cout << "result:" << std::endl;
  //print_mat(spd, dim_size, dim_size, dim_size);

  // Now free the per-thread resources.
  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    auto stat = cudaFree(reinterpret_cast<void*>(*b0s.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for blocks.");
    }
    stat = cudaFree(reinterpret_cast<void*>(*lworks.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for cholesky workspace.");
    }
    stat = cudaFree(reinterpret_cast<void*>(*dev_infos.getLocal()));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free gpu buffer for cholesky return status.");
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
    /*stat = cudaStreamDestroy(*cusolver_streams.getLocal());
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to free cuda stream for cusolver.");
    }*/
    auto stat3 = cublasDestroy(*handles.getLocal());
    if (stat3 != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to free cublas resources.");
    }
    stat = cudaDeviceReset();
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to reset cuda device after use");
    }
  });

  check_correctness(spd, dim_size, seed, tolerance);
  
  return 0;
}
