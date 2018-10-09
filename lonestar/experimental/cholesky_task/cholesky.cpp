#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <fstream>
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

#include "strided_array.hpp"
#include "cuda_unique.hpp"
#include "cublas_wrappers.hpp"

static const char* name = "Task Based Cholesky Factorization";
static const char* desc = "";
static const char* url = "cholesky_task";

static llvm::cl::opt<int> dim_size("dim_size", llvm::cl::desc("Number of rows/columns in main matrix."), llvm::cl::init(9));
static llvm::cl::opt<int> block_size("block_size", llvm::cl::desc("Number of rows/columns in each block."), llvm::cl::init(3));
static llvm::cl::opt<unsigned int> max_queue_size("max_queue_size", llvm::cl::desc("Maximum number of tasks in task queue"), llvm::cl::init(100));
static llvm::cl::opt<unsigned int> min_queue_size("min_queue_size", llvm::cl::desc("Threshold for generating more tasks in the task queue"), llvm::cl::init(30));
static llvm::cl::opt<int> seed("seed", llvm::cl::desc("Seed used to generate symmetric positive definite matrix."), llvm::cl::init(0));
static llvm::cl::opt<double> tolerance("tolerance", llvm::cl::desc("Tolerance used to check that the results are correct."), llvm::cl::init(.001));
static llvm::cl::opt<std::string> dependence_outfile("dependence_outfile", llvm::cl::desc("Write the dependence graph to the given file (in .dot format)."), llvm::cl::init(""));

using task_data = std::tuple<std::atomic<std::size_t>, int, int, int, char>;
using task_label = std::tuple<int, int, int, char>;

task_label data_to_label(task_data &d) noexcept {
  return std::make_tuple(std::get<1>(d), std::get<2>(d), std::get<3>(d), std::get<4>(d));
}

// Use Morph_Graph to get this up and running quickly.
using Graph = galois::graphs::MorphGraph<task_data, void, true>;
using GNode = Graph::GraphNode;
using LMap = std::map<task_label, GNode>;

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

std::string task_to_str(task_data &t) {
  std::stringstream ss;
  int type = std::get<4>(t);
  ss << "T" << type << "(";
  if (type == 1 || type == 4) {
    ss << std::get<2>(t) << ", " << std::get<3>(t) << ")";
  } else if (type == 2) {
    ss << std::get<3>(t) << ")";
  } else if (type == 3) {
    ss << std::get<1>(t) << ", " << std::get<2>(t) << ", " << std::get<3>(t) << ")";
  } else {
    // Invalid task type
    assert(false);
    }
    return ss.str();
    }

void write_dependences(Graph &g, std::string fname) {
  // TODO: RAII for this would be better!
  std::ofstream f;
  f.open(fname);
  f << "digraph cholesky_deps {\n";
  for (auto n : g) {
    auto &d = g.getData(n);
    for (auto e : g.edges(n)) {
      auto &a = g.getData(g.getEdgeDst(e));
      f << "    \"" << task_to_str(d) << "\" -> \"" << task_to_str(a) << "\";\n";
    }
  }
  f << "}\n";
  f.close();
}

void generate_tasks(int nblocks, Graph &g, LMap &label_map) {
  auto register_task = [&](int tp, int i0, int i1, int i2, int count) {
    auto n = g.createNode(count, i0, i1, i2, tp);
    g.addNode(n);
    task_label dep{i0, i1, i2, tp};
    label_map[task_label(i0, i1, i2, tp)] = n;
    return n;
  };
  auto add_dep = [&](GNode n, int tp, int i0, int i1, int i2) {
    g.addEdge(label_map[task_label(i0, i1, i2, tp)], n);
  };
  for (std::size_t j = 0; j < nblocks; j++) {
    for (std::size_t k = 0; k < j; k++) {
      if (k == 0) {
        auto n = register_task(1, 0, j, k, 1);
        add_dep(n, 4, 0, j, k);
      } else {
        auto n = register_task(1, 0, j, k, 2);
        add_dep(n, 4, 0, j, k);
        add_dep(n, 1, 0, j, k-1);
      }
    }
    if (j == 0) {
      auto n = register_task(2, 0, 0, j, 0);
    } else {
      auto n = register_task(2, 0, 0, j, 1);
      add_dep(n, 1, 0, j, j-1);
    }
    for (std::size_t i = j + 1; i < nblocks; i++) {
      for (std::size_t k = 0; k < j; k++) {
        if (k == 0) {
          auto n = register_task(3, i, j, k, 2);
          add_dep(n, 4, 0, j, k);
          add_dep(n, 4, 0, i, k);
        } else {
          auto n = register_task(3, i, j, k, 3);
          add_dep(n, 4, 0, j, k);
          add_dep(n, 4, 0, i, k);
          add_dep(n, 3, i, j, k-1);
        }
      }
      if (j == 0) {
        auto n = register_task(4, 0, i, j, 1);
        add_dep(n, 2, 0, 0, j);
      } else {
        auto n = register_task(4, 0, i, j, 2);
        add_dep(n, 2, 0, 0, j);
        add_dep(n, 3, i, j, j-1);
      }
    }
  }
} 

struct task_generator {
  Graph &graph;
  LMap &label_map;
  int num_blocks;
  int j = 0;
  int i = 1;
  int k = 0;
  int task_type = 0;
  std::atomic<bool> finished = false;

  task_generator(Graph &g, LMap &l, int nb): graph(g), label_map(l), num_blocks(nb) {}

  task_generator() = delete;
  task_generator(task_generator&) = delete;
  task_generator(task_generator&&) = delete;
  task_generator &operator=(task_generator&) = delete;
  task_generator &operator=(task_generator&&) = delete;

private:
  int num_deps() noexcept {
    if (task_type == 1) return k == 0 ? 1 : 2;
    if (task_type == 2) return j == 0 ? 0 : 1;
    if (task_type == 3) return k == 0 ? 2 : 3;
    if (task_type == 4) return j == 0 ? 1 : 2;
    std::abort();
  }

  auto create_current() noexcept {
    auto create = [&](int tp, int i0, int i1, int i2, int count) {
      auto n = graph.createNode(count, i0, i1, i2, tp);
      graph.addNode(n);
      label_map[task_label(i0, i1, i2, tp)] = n;
      return n;
    };
    // TODO Fix the "leading by 0" convention so branching like
    // this isn't necessary.
    if (task_type == 1) return create(1, 0, j, k, num_deps());
    if (task_type == 2) return create(2, 0, 0, j, num_deps());
    if (task_type == 3) return create(3, i, j, k, num_deps());
    if (task_type == 4) return create(4, 0, i, j, num_deps());
    std::abort();
  }

  void add_deps(auto node, auto &context) noexcept {
    auto add_dep = [&](GNode n, int tp, int i0, int i1, int i2) {
      auto dependency_label = task_label(i0, i1, i2, tp);
      if (label_map.count(dependency_label)) {
        graph.addEdge(label_map[dependency_label], n);
      } else {
        // If the node isn't there, then it's completed already.
        std::get<0>(graph.getData(n))--;
      }
    };
    if (task_type == 1) {
      add_dep(node, 4, 0, j, k);
      if (k > 0) add_dep(node, 1, 0, j, k-1);
    } else if (task_type == 2) {
      if (j > 0) add_dep(node, 1, 0, j, j-1);
    } else if (task_type == 3) {
      add_dep(node, 4, 0, j, k);
      add_dep(node, 4, 0, i, k);
      if (k > 0) add_dep(node, 3, i, j, k-1);
    } else if (task_type == 4) {
      add_dep(node, 2, 0, 0, j);
      if (j > 0) add_dep(node, 3, i, j, j-1);
    } else {
      std::abort();
    }
    // Push the work to the list of ready tasks if all
    // its dependencies have finished.
    if (std::get<0>(graph.getData(node)) == 0) context.push(node);
  }

  auto register_current(auto &context) noexcept {
    auto n = create_current();
    add_deps(n, context);
  }

  void advance() noexcept {
    if (task_type == 0 && !finished) {
      task_type = 2;
    }else if (task_type == 1) {
      if (k + 1 < j) {
        k++;
      } else {
        k = 0;
        task_type = 2;
      }
    } else if (task_type == 2) {
      if (j > 0) {
        if (j + 1 < num_blocks) {
          task_type = 3;
        } else {
          // Advance after finished iteration.
          std::abort();
        }
      } else {
        task_type = 4;
      }
    } else if (task_type == 3) {
      if (k + 1 < j) {
        k++;
      } else {
        k = 0;
        task_type = 4;
      }
    } else if (task_type == 4) {
      if (i + 1 < num_blocks) {
        i++;
        if (j > 0) {
          task_type = 3;
        } else {
          task_type = 4;
        }
      } else {
        j++;
        i = j + 1;
        task_type = 1;
      }
    }
  }

  void register_next(auto &context) noexcept {
    advance();
    register_current(context);
    if (task_type == 2 && j + 1 == num_blocks) finished = true;
  }

public:

  void print_state() noexcept {
    std::cout << task_type << "(" << i << ", " << j << ", " << k << ")" << std::endl;
  }

  void expand_new_tasks(std::size_t min_size, std::size_t max_size, auto &context, std::unique_lock<std::mutex> && graph_lock) noexcept {
    if (graph.size() >= min_size) return;
    while (graph.size() < max_size && !finished) {
      register_next(context);
    }
  }
};

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
  sa::strided_array<double, 2> spd_ar{spd, sa::array_axes<2>({{{dim_size, sizeof(double)}, {dim_size, dim_size * sizeof(double)}}})};
  auto spd_bk = spd_ar.block((int)block_size, (int)block_size);
  //std::cout.precision(19);
  //std::cout << "generated input:" << std::endl;
  //print_mat(spd, dim_size, dim_size, dim_size);

  //galois::StatTimer construction_timer{"Task graph construction"};

  //construction_timer.start();
  Graph g;
  LMap label_map;
  // Coarse-grained lock to limit access to the graph and label map.
  // This is needed because we don't want the Galois loop to abort
  // when acquiring locks to delete nodes after a task completes.
  // Note! If there were a thread-safe way to check the number of nodes in
  // a graph, this would probably not be necessary.
  std::mutex graph_lock;
  auto generator = task_generator(g, label_map, nblocks);
  // Use an all zeros node to denote the "start generating" task.
  auto init_node = g.createNode(0, 0, 0, 0, 0);
  g.addNode(init_node);
  //generate_tasks(nblocks, g, label_map);
  //construction_timer.stop();
  //print_deps(g);
  if (dependence_outfile != "") {
    // Use the non-lazy generation code to write the dependency
    // information to the desired file.
    Graph g2;
    LMap label_map_2;
    generate_tasks(nblocks, g2, label_map_2);
    write_dependences(g, dependence_outfile);
  }

  // Timers to measure the portion of time spent on different portions of the given operator.
  galois::PerThreadTimer<true> updating_tasks_timer{"cholesky_tasks", "task updates"};
  galois::PerThreadTimer<true> computation_calls_timer{"cholesky_tasks", "computation calls"};
  galois::PerThreadTimer<true> data_movement_timer{"cholesky_tasks", "data movement"};
  galois::StatTimer verification_timer{"Verification of Cholesky decomposition."};

  // Now set up the needed resources for each thread.
  galois::substrate::PerThreadStorage<sa::cublas::context> cublas_contexts(nullptr);
  galois::substrate::PerThreadStorage<cusolverDnHandle_t> cusolver_handles(nullptr);
  galois::substrate::PerThreadStorage<int> lwork_sizes;
  sa::array_axes<2> block_axes{{{{block_size, sizeof(double)}, {block_size, block_size * sizeof(double)}}}};
  galois::substrate::PerThreadStorage<double*> b0s;
  galois::substrate::PerThreadStorage<sa::strided_array<double, 2>> b0as{nullptr, block_axes};
  galois::substrate::PerThreadStorage<double*> b1s;
  galois::substrate::PerThreadStorage<sa::strided_array<double, 2>> b1as{nullptr, block_axes};
  galois::substrate::PerThreadStorage<double*> b2s;
  galois::substrate::PerThreadStorage<sa::strided_array<double, 2>> b2as{nullptr, block_axes};
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
    auto stat = cublasCreate(&(cublas_contexts.getLocal()->handle));
    if (stat != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cublas.");
    }
    auto stat2 = cusolverDnCreate(cusolver_handles.getLocal());
    if (stat2 != CUSOLVER_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cusolver.");
    }
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
    b0as.getLocal()->data = *b0s.getLocal();
    stat3 = cudaMalloc(b1s.getLocal(), static_cast<std::size_t>(block_size * block_size * sizeof(double)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    b1as.getLocal()->data = *b1s.getLocal();
    stat3 = cudaMalloc(b2s.getLocal(), static_cast<std::size_t>(block_size * block_size * sizeof(double)));
    if (stat3 != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for blocked operations.");
    }
    b2as.getLocal()->data = *b2s.getLocal();
  });

  std::atomic<int> counter{0};

  // Now execute the tasks.
  galois::for_each(
    galois::iterate({init_node}),
    [&](GNode n, auto& ctx) {
      auto &d = g.getData(n);
      auto task_type = std::get<4>(d);
      if (task_type == 1) {
        auto j = std::get<2>(d);
        auto k = std::get<3>(d);
        auto &b0a = *b0as.getLocal();
        auto &b1a = *b1as.getLocal();
        auto &ctx = *cublas_contexts.getLocal();
        data_movement_timer.start();
        auto stat = sa::cublas::SetMatrix(spd_bk(j, j, sa::slice(), sa::slice()), b0a);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU Failed");
        stat = sa::cublas::SetMatrix(spd_bk(j,k, sa::slice(), sa::slice()), b1a);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU Failed");
        double alpha = -1, beta = 1;
        data_movement_timer.stop();
        computation_calls_timer.start();
        stat = sa::cublas::gemm(ctx, alpha, b1a, b1a.transpose(), beta, b0a);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          print_cublas_status(stat);
          GALOIS_DIE("gemm failed.");
        }
        cudaDeviceSynchronize();
        computation_calls_timer.stop();
        data_movement_timer.start();
        stat = sa::cublas::GetMatrix(b0a, spd_bk(j, j, sa::slice(), sa::slice()));
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("recieve failed.");
        data_movement_timer.stop();
      } else if (task_type == 2) {
        auto j = std::get<3>(d);
        auto b0 = *b0s.getLocal();
        auto lwork = *lworks.getLocal();
        auto lwork_size = *lwork_sizes.getLocal();
        auto cusolver_handle = *cusolver_handles.getLocal();
        data_movement_timer.start();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (5)");
        int info = 0;
        data_movement_timer.stop();
        computation_calls_timer.start();
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
        cudaDeviceSynchronize();
        computation_calls_timer.stop();
        data_movement_timer.start();
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b0, block_size, spd + j * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (10)");
        data_movement_timer.stop();
      } else if (task_type == 3) {
        auto i = std::get<1>(d);
        auto j = std::get<2>(d);
        auto k = std::get<3>(d);
        auto &b0a = *b0as.getLocal();
        auto &b1a = *b1as.getLocal();
        auto &b2a = *b2as.getLocal();
        auto &ctx = *cublas_contexts.getLocal();
        data_movement_timer.start();
        auto stat = sa::cublas::SetMatrix(spd_bk(i,j,sa::slice(), sa::slice()), b0a);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed");
        stat = sa::cublas::SetMatrix(spd_bk(i,k,sa::slice(), sa::slice()), b1a);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed.");
        stat = sa::cublas::SetMatrix(spd_bk(j,k,sa::slice(), sa::slice()), b2a);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed.");
        data_movement_timer.stop();
        computation_calls_timer.start();
        double alpha = -1, beta = 1;
        stat = sa::cublas::gemm(ctx, alpha, b1a, b2a.transpose(), beta, b0a);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          print_cublas_status(stat);
          GALOIS_DIE("gemm on GPU failed.");
        }
        cudaDeviceSynchronize();
        computation_calls_timer.stop();
        data_movement_timer.start();
        stat = sa::cublas::GetMatrix(b0a, spd_bk(i,j,sa::slice(),sa::slice()));
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed.");
        data_movement_timer.stop();
      } else if (task_type == 4) {
        auto i = std::get<2>(d);
        auto j = std::get<3>(d);
        auto b0 = *b0s.getLocal();
        auto b1 = *b1s.getLocal();
        //auto handle = *handles.getLocal();
        auto handle = cublas_contexts.getLocal()->handle;
        data_movement_timer.start();
        auto stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + j * block_size + j * block_size * dim_size, dim_size, b0, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (16)");
        stat = cublasSetMatrix(block_size, block_size, sizeof(double), spd + i * block_size + j * block_size * dim_size, dim_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed. (17)");
        data_movement_timer.stop();
        computation_calls_timer.start();
        double alpha = 1.;
        stat = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, block_size, block_size, &alpha, b0, block_size, b1, block_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("dtrsm operation failed. (18)");
        cudaDeviceSynchronize();
        computation_calls_timer.stop();
        data_movement_timer.start();
        stat = cublasGetMatrix(block_size, block_size, sizeof(double), b1, block_size, spd + i * block_size + j * block_size * dim_size, dim_size);
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve from GPU failed. (19)");
        data_movement_timer.stop();
      } else if (task_type != 0) {
        // Task type 0 signals to jump ahead to generation.
        // Otherwise, something has gone terribly wrong.
        GALOIS_DIE("Unrecognized task type.");
      }

      updating_tasks_timer.start();
      for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        auto dst = g.getEdgeDst(e);
        if (0 == --std::get<0>(g.getData(dst, galois::MethodFlag::UNPROTECTED))) {
          ctx.push(dst);
        }
      }

      // Now remove this node entirely.
      g.removeNode(n);
      std::unique_lock graph_lock_handle{graph_lock};
      // Lock covers the map from labels to nodes too, so remove it after qcquiring the lock.
      label_map.erase(data_to_label(d));
      generator.expand_new_tasks(min_queue_size, max_queue_size, ctx, std::move(graph_lock_handle));
      updating_tasks_timer.stop();
    },
    galois::loopname("cholesky_tasks"), galois::wl<PSChunk>()
  );

  {
    auto spd2_manager = generate_symmetric_positive_definite(dim_size, seed);
    auto spd2 = spd2_manager.get();
    galois::StatTimer cusolver_dpotrf_with_movement{"cusolver dpotrf with movement time"};
    galois::StatTimer cusolver_dpotrf_timer{"cusolver dpotrf time"};
    int info;
    double *dev_spd2;
    auto stat = cudaMalloc(&dev_spd2, dim_size * dim_size * sizeof(double));
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to allocate GPU buffers for fully on-device cholesky.");
    }
    cusolver_dpotrf_with_movement.start();
    auto stat2 = cublasSetMatrix(dim_size, dim_size, sizeof(double), spd2, dim_size, dev_spd2, dim_size);
    if (stat2 != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Send to GPU failed.");
    int work_size;
    auto stat3 = cusolverDnDpotrf_bufferSize(*cusolver_handles.getLocal(), CUBLAS_FILL_MODE_LOWER, dim_size, dev_spd2, dim_size, &work_size);
    if (stat3 != CUSOLVER_STATUS_SUCCESS) GALOIS_DIE("Could not get work size for cusolver dpotrf.");
    double *work;
    stat = cudaMalloc(&work, work_size * sizeof(double));
    if (stat != cudaSuccess) GALOIS_DIE("Failed to allocate gpu work buffer");
    cusolver_dpotrf_timer.start();
    stat3 = cusolverDnDpotrf(*cusolver_handles.getLocal(), CUBLAS_FILL_MODE_LOWER, dim_size, dev_spd2, dim_size, work, work_size, *dev_infos.getLocal());
    if (stat3 != CUSOLVER_STATUS_SUCCESS) GALOIS_DIE("Cholesky block solve failed.");
    stat = cudaMemcpy(&info, *dev_infos.getLocal(), sizeof(int), cudaMemcpyDeviceToHost);
    if (stat != cudaSuccess) GALOIS_DIE("Recieve status after Cholesky on block failed. (7)");
    if (info != 0) {
      std::cout << info << std::endl;
      std::stringstream ss;
      if (info < 0) {
        ss << "Parameter " << -info << " incorrect when passed to dpotrf.";
        GALOIS_DIE(ss.str());
      }
      if (info > 0) {
        ss << "Not positive definite at minor " << info << " during per-block Cholesky computation.";
        GALOIS_DIE(ss.str());
      }
    }
    cudaDeviceSynchronize();
    cusolver_dpotrf_timer.stop();
    stat2 = cublasGetMatrix(dim_size, dim_size, sizeof(double), dev_spd2, dim_size, spd2, dim_size);
    if (stat2 != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Recieve failed.");
    cusolver_dpotrf_with_movement.stop();
  }
  
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
    if (stat != cudaSuccess) {
      GALOIS_DIE("Failed to reset cuda device after use");
    }
  });

  verification_timer.start();
  check_correctness(spd, dim_size, seed, tolerance);
  verification_timer.stop();

  return 0;
}
