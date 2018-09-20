#include <atomic>
#include <cstddef>
#include <mutex>
#include <random>
#include <tuple>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    auto stat = cublasCreate(handles.getLocal());
    if (stat != CUBLAS_STATUS_SUCCESS) {
      GALOIS_DIE("Failed to initialize cublas");
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
      } else if (task_type == 2) {
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::WRITE);
      } else if (task_type == 3) {
        auto i = std::get<1>(d);
        auto j = std::get<2>(d);
        auto k = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * j]), galois::MethodFlag::WRITE);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * k]), galois::MethodFlag::READ);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * k]), galois::MethodFlag::READ);
      } else if (task_type == 4) {
        auto i = std::get<2>(d);
        auto j = std::get<3>(d);
        galois::runtime::doAcquire(&(locks.get()[j + nblocks * j]), galois::MethodFlag::READ);
        galois::runtime::doAcquire(&(locks.get()[i + nblocks * j]), galois::MethodFlag::WRITE);
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
    cublasDestroy(*handles.getLocal());
  });
  //cublasDestroy(handle);
  
  return 0;
}
