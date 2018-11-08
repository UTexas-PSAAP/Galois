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
#include "galois/gstl.h"
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
static const char* url = "barneshut_task";

static llvm::cl::opt<unsigned long long> dims("dims", llvm::cl::desc("Number of dimensions"), llvm::cl::init(2u));
static llvm::cl::opt<unsigned long long> num_pts("num_pts", llvm::cl::desc("Number of generated points."), llvm::cl::init(10u));
static llvm::cl::opt<unsigned long long> max_group_size("max_group_size", llvm::cl::desc("Max number of points per leaf in tree."), llvm::cl::init(1u));
static llvm::cl::opt<unsigned long long> seed("seed", llvm::cl::desc("Seed for random number generation."), llvm::cl::init(0u));
static llvm::cl::opt<double> error_threshold("error_threshold", llvm::cl::desc("Maximum allowable error at a given point during verification."), llvm::cl::init(1E-15));
static llvm::cl::opt<bool> verify("verify", llvm::cl::desc("Whether or not to run the verification."), llvm::cl::init(true));
static llvm::cl::opt<unsigned long long> max_task_num("max_task_num", llvm::cl::desc("Maximum number of tasks."), llvm::cl::init(100u));
static llvm::cl::opt<unsigned long long> min_task_num("min_task_num", llvm::cl::desc("Minimum number of tasks."), llvm::cl::init(50u));

using PSChunk = galois::worklists::PerSocketChunkFIFO<1>;

struct tree_node_data;

using tree_t = galois::graphs::MorphGraph<tree_node_data, void, true>::with_no_lockable<true>::type;
using tree_node_t = tree_t::GraphNode;

struct tree_node_data {
  std::size_t start_idx;
  std::size_t num_pts;
  double *bounds = nullptr;
  double *summary = nullptr;
  tree_node_t parent = nullptr;

  tree_node_data(std::size_t start, std::size_t num, double *bds, double *sum=nullptr, tree_node_t pt=nullptr) : start_idx(start), num_pts(num), bounds(bds), summary(sum), parent(pt) {
    assert(bds != nullptr);
    for (std::size_t i = 0; i < 4; i++) {
      assert(-1. <= bds[i] && bds[i] <= 1.);
    }
  }
};

enum task_type {
  recurse_summary,
  compute_summary,
  compute_forces
};

struct task_label {
  tree_node_t tree_node;
  task_type task;
};

struct task_data {
  std::atomic<std::size_t> remaining_dependencies;
  task_label label;
};

// Use Morph_Graph to get this up and running quickly.
using task_graph_base = galois::graphs::MorphGraph<task_data, void, true>::with_no_lockable<true>::type;
using task_graph_node = task_graph_base::GraphNode;
using label_map_t = std::map<task_label, task_graph_node>;

struct task_graph {
  task_graph_base g;
  label_map_t label_map;
  // This lock manages creation and deletion of tasks in the graph and in the label map.
  std::mutex lock;
  std::condition_variable condition;

  task_graph() = default;
  task_graph(task_graph&) = delete;
  // Might make sense to make this moveable, but it seems ulikely to work without
  // at least some additional fiddling, so disable it explicitly to avoid inscrutable errors.
  // Same goes for assignment operators.
  task_graph(task_graph&&) = delete;
  task_graph& operator=(task_graph&) = delete;
  task_graph& operator=(task_graph&&) = delete;

  void sleep_till_size(std::size_t size) {
    std::unique_lock<std::mutex> lk{lock};
    condition.wait(lk, [&](){return g.size() < size;});
  }

  void register_task(auto &ctx, task_label label, std::size_t num_deps, task_label* dependencies) {
    std::unique_lock<std::mutex> lk{lock};
    auto n = g.createNode(num_deps, label);
    g.addNode(n);
    label_map[label] = n;
    for (std::size_t i = 0; i < num_deps; i++) {
      auto dep = dependencies[i];
      if (label_map.count(dep)) {
        g.addEdge(label_map[dep], n);
      } else {
        if (!(--g.getData(n).remaining_dependencies)) {
          ctx.push(n);
        }
      }
    }
    if (g.size() < max_task_num) {
      return;
    }
    condition.wait(lk, [&](){return g.size() < min_task_num;});
  }

  void remove_task_node(auto &ctx, std::unique_lock<std::mutex> &&lk, task_graph_node node) {
    for (auto e : g.edges(node, galois::MethodFlag::UNPROTECTED)) {
      auto dst = g.getEdgeDst(e);
      if (!(--g.getData(dst, galois::MethodFlag::UNPROTECTED).remaining_dependencies)) {
        ctx.push(dst);
      }
    }
    g.removeNode(node);
    if (g.size() == min_task_num) {
      lk.release();
      condition.notify_one();
    }
  }

  void remove_task(auto &ctx, task_graph_node node) {
    std::unique_lock<std::mutex> lk{lock};
    task_label label{g.getData(node, galois::MethodFlag::UNPROTECTED).label};
    label_map.erase(label);
    remove_task_node(ctx, std::move(lk), node);
  }

  void remove_task(auto &ctx, task_label label) {
    std::unique_lock<std::mutex> lk{lock};
    task_graph_node node{label_map[label]};
    label_map.erase(label);
    remove_task_node(ctx, std::move(lk), node);
  }

};

void generate_tasks_lazy(task_graph &tsk, auto &ctx) {
  ;
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

void print_2d_array(double *m, std::size_t rows, std::size_t cols, std::ptrdiff_t row_stride) {
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      std::cout << m[i + j * row_stride] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_2d_array(sa::strided_array<double, 2> ar) {
  std::cout << "[" << std::endl;
  for (std::size_t i = 0; i < ar.axes[0].shape; i++) {
    std::cout << "  [";
    for (std::size_t j = 0; j < ar.axes[1].shape; j++) {
      std::cout << ar(i, j) << ",";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;
}

decltype(auto) generate_random(std::size_t size, std::size_t seed) {
  auto out = std::make_unique<double[]>(size);
  std::mt19937 gen{seed};
  std::uniform_real_distribution<> dis(-1., 1.);
  for (std::size_t i = 0; i < size; i++) {
    out.get()[i] = dis(gen);
  }
  return out;
}

std::size_t split_cloud_impl(std::size_t dims, double *pts, std::size_t num_pts, double split, std::size_t dim_idx) {
  assert(num_pts > 0);
  std::size_t lower = 0, upper = num_pts - 1;
  std::size_t i = 0;
  while (true) {
    while (pts[dims * lower + dim_idx] <= split) {
      lower++;
      if (lower > upper) return lower;
    }
    while (pts[dims * upper + dim_idx] > split) {
      if (lower >= upper) return lower;
      upper--;
    }
    std::swap_ranges(pts + dims * lower, pts + dims * (lower + 1), pts + dims * upper);
    lower++;
    if (lower >= upper) {
      return lower;
    }
    upper--;
  }
}

struct cloud_splitter {
  sa::strided_array<double, 2> &points;
  std::size_t dims;
  galois::gstl::Vector<double> &split_point;
  galois::gstl::Vector<std::tuple<std::size_t, std::size_t, double*>> &children;
  galois::Pow_2_VarSizeAlloc<double> &allocator;
  cloud_splitter(sa::strided_array<double, 2> &p, std::size_t d, galois::gstl::Vector<double> &s, galois::gstl::Vector<std::tuple<std::size_t, std::size_t, double*>>& c, galois::Pow_2_VarSizeAlloc<double> &a) noexcept : points(p), dims(d), split_point(s), children(c), allocator(a) {}

  void split(std::size_t start_idx, std::size_t num_pts, double *bounds, std::size_t dim_idx = 0) {
    auto start_ptr = &points(start_idx, 0);
    std::size_t lower = split_cloud_impl(std::size_t(dims), start_ptr, num_pts, split_point[dim_idx], dim_idx);
    for (std::size_t i = 0; i < lower; i++) {
      if (points(start_idx + i, dim_idx) > split_point[dim_idx]) {
        GALOIS_DIE("split_cloud_impl failed");
      }
    }
    for (std::size_t i = lower; i < num_pts; i++) {
      if (points(start_idx + i, dim_idx) <= split_point[dim_idx]) {
        GALOIS_DIE("split_cloud_impl failed");
      }
    }
    assert(lower <= num_pts);
    if (!lower) {
      bounds[dims * dim_idx] = split_point[dim_idx];
      if (dim_idx + 1 == dims) {
        children.emplace_back(start_idx, num_pts, bounds);
      } else {
        split(start_idx, num_pts, bounds, dim_idx + 1);
      }
    } else if (lower == num_pts) {
      bounds[dims * dim_idx + 1] = split_point[dim_idx];
      if (dim_idx + 1 == dims) {
        children.emplace_back(start_idx, num_pts, bounds);
      } else {
        split(start_idx, num_pts, bounds, dim_idx + 1);
      }
    } else {
      auto lower_bounds = bounds;
      //auto lower_bounds = allocator.allocate(2 * dims);
      //auto upper_bounds = reinterpret_cast<double*>(malloc(2 * dims * sizeof(double)));
      auto upper_bounds = allocator.allocate(2 * dims);
      //std::cout << "malloc: " << std::uintptr_t(upper_bounds) << std::endl;
      for (std::size_t i = 0; i < 2 * dims; i++) {
        upper_bounds[i] = bounds[i];
        //lower_bounds[i] = bounds[i];
      }
      lower_bounds[dims * dim_idx + 1] = split_point[dim_idx];
      upper_bounds[dims * dim_idx] = split_point[dim_idx];
      if (dim_idx + 1 == dims) {
        children.emplace_back(start_idx, lower, lower_bounds);
        children.emplace_back(start_idx + lower, num_pts - lower, upper_bounds);
      } else {
        split(start_idx, lower, lower_bounds, dim_idx + 1);
        split(start_idx + lower, num_pts - lower, upper_bounds, dim_idx + 1);
      }
    }
  }
};

void print_tree(auto &tree, auto node, std::size_t prefix = 0) {
  for (std::size_t i = 0; i < prefix; i++) {
    std::cout << "  ";
  }
  auto bounds = tree.getData(node).bounds;
  for (std::size_t i = 0; i < dims; i++) {
    std::cout << "(" << bounds[2 * i] << ", " << bounds[2 * i + 1] << ")";
  }
  std::cout << std::endl;
  for (auto e : tree.edges(node)) {
    print_tree(tree, tree.getEdgeDst(e), prefix + 1);
  }
}

void print_tree_ptrs(auto &tree, auto node, std::size_t prefix = 0) {
  for (std::size_t i = 0; i < prefix; i++) {
    std::cout << "  ";
  }
  auto bounds = tree.getData(node).bounds;
  std::cout << std::uintptr_t(bounds) << std::endl;
  for (auto e : tree.edges(node)) {
    print_tree_ptrs(tree, tree.getEdgeDst(e), prefix + 1);
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  galois::Pow_2_VarSizeAlloc<double> allocator;

  auto base_buffer = generate_random(3 * dims * num_pts, seed);
  sa::strided_array<double, 3> base_arr(base_buffer.get(), sa::array_axes<3>({{{3, dims * num_pts * sizeof(double)}, {num_pts, dims * sizeof(double)}, {dims, sizeof(double)}}}));
  auto points = base_arr(0, sa::slice(), sa::slice());
  auto velocities = base_arr(1, sa::slice(), sa::slice());
  auto forces = base_arr(2, sa::slice(), sa::slice());

  task_graph tasks;
  tree_t tree;
  double *root_bounds = allocator.allocate(2 * dims);
  //auto root_bounds = reinterpret_cast<double*>(malloc(2 * dims * sizeof(double)));
  for (std::size_t i = 0; i < dims; i++) {
    root_bounds[2 * i] = -1.;
    root_bounds[2 * i + 1] = 1.;
  }
  auto root_node = tree.createNode(std::size_t(0), std::size_t(num_pts), root_bounds, nullptr, nullptr);
  tree.addNode(root_node);
  tree.getData(root_node, galois::MethodFlag::UNPROTECTED).bounds = root_bounds;
  if (num_pts > max_group_size) {
    galois::for_each(
      galois::iterate({root_node}),
      [&](tree_node_t node, auto& ctx) {
        // Start index, number of points, bounds pointer.
        galois::gstl::Vector<std::tuple<std::size_t, std::size_t, double*>> children;
        galois::gstl::Vector<double> split_point;
        split_point.reserve(dims);
        auto &data = tree.getData(node, galois::MethodFlag::UNPROTECTED);
        // TODO: Use a unique_ptr for this.
        double *bounds = allocator.allocate(2 * dims);
        //std::cout << "malloc: " << std::uintptr_t(bounds) << std::endl;
        //auto bounds = reinterpret_cast<double*>(malloc(2 * dims * sizeof(double)));
        for (std::size_t i = 0; i < dims; i++) {
          bounds[2 * i] = data.bounds[2 * i];
          bounds[2 * i + 1] = data.bounds[2 * i + 1];
          split_point.push_back(.5 * (bounds[2 * i] + bounds[2 * i + 1]));
        }
        /*std::cout << "parent bounds: ";
        for (std::size_t i = 0; i < dims; i++) {
          std::cout << "(" << bounds[2 * i] << ", " << bounds[2 * i + 1] << ")";
        }
        std::cout << std::endl;
        */
        // Transfer ownership of bounds to this function.
        cloud_splitter(points, dims, split_point, children, allocator).split(data.start_idx, data.num_pts, bounds);
        /*for (auto &child : children) {
          std::cout << "  child: ";
          auto child_bounds = std::get<2>(child);
          for (std::size_t i = 0; i < dims; i++) {
            std::cout << "(" << child_bounds[2 * i] << ", " << child_bounds[2 * i + 1] << ")";
          }
          std::cout << std::endl;
        }*/
        if (children.size() == 1) {
          //std::cout << "free: " << std::uintptr_t(data.bounds) << std::endl << std::endl;
          allocator.deallocate(data.bounds, 2 * dims);
          //free(data.bounds);
          data.bounds = std::get<2>(children[0]);
          ctx.push(node);
          return;
        }
        //std::cout << std::endl;
        for (auto &child : children) {
          auto child_node = tree.createNode(std::get<0>(child), std::get<1>(child), std::get<2>(child), nullptr, node);
          tree.addNode(child_node);
          tree.addEdge(node, child_node);
          if (std::get<1>(child) > max_group_size) {
            ctx.push(child_node);
          }
        }
      },
      galois::loopname("build_tree"),
      galois::wl<galois::worklists::PerSocketChunkFIFO<16>>(),
      galois::no_conflicts()
    );
  }

  //print_2d_array(points);

  //print_tree(tree, root_node);
  //print_tree_ptrs(tree, root_node);

  if (verify) {
    for (auto node : tree) {
      auto data = tree.getData(node, galois::MethodFlag::UNPROTECTED);
      if (data.bounds == nullptr) {
        GALOIS_DIE("Null bounds pointer.");
      }
      /*for (auto other : tree) {
        if (node == other) continue;
        auto other_data = tree.getData(other, galois::MethodFlag::UNPROTECTED);
        if (other_data.bounds == data.bounds) {
          for (std::size_t i = 0; i < dims; i++) {
            std::cout << "(" << data.bounds[2 * i] << ", " << data.bounds[2 * i + 1] << ")";
          }
          std::cout << std::endl;
          GALOIS_DIE("Aliased bounds pointers.");
        }
      }*/
      for (std::size_t i = data.start_idx; i < data.start_idx + data.num_pts; i++) {
        for (std::size_t j = 0; j < dims; j++) {
          if (data.bounds == nullptr) {
            GALOIS_DIE("Uninitialized bounds.");
          }
          if (points(i, j) < data.bounds[2 * j] or data.bounds[2 * j + 1] < points(i, j)) {
            std::cout << points(i,j) << std::endl;
            std::cout << data.bounds[2 * j] << " " << data.bounds[2 * j + 1] << std::endl;
            GALOIS_DIE("Verification of tree construction failed");
          }
        }
      }
    }
  }

  

  /*;;;;

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
  // Use an atomic to track (approximately) the graph size separately.
  // Don't include the "generate tasks" node in this count.
  std::atomic<std::size_t> graph_size = 0;
  LMap label_map;
  // Coarse-grained lock to limit access to the label map and graph.
  std::mutex map_lock;
  std::condition_variable resume_generation_condition;
  std::mutex condition_lock;
  //auto generator = task_generator(g, label_map, nblocks);
  // Use an all zeros node to denote the "start generating" task.
  auto init_node = g.createNode(0, task_label(0, 0, 0, 0));
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
      auto task_type = std::get<3>(d.label);
      if (task_type == 1) {
        auto j = std::get<1>(d.label);
        auto k = std::get<2>(d.label);
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
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("receive failed.");
        data_movement_timer.stop();
      } else if (task_type == 2) {
        auto j = std::get<2>(d.label);
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
        if (stat3 != cudaSuccess) GALOIS_DIE("Receive status after Cholesky on block failed. (7)");
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
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Receive from GPU failed. (10)");
        data_movement_timer.stop();
      } else if (task_type == 3) {
        auto i = std::get<0>(d.label);
        auto j = std::get<1>(d.label);
        auto k = std::get<2>(d.label);
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
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Receive from GPU failed.");
        data_movement_timer.stop();
      } else if (task_type == 4) {
        auto i = std::get<1>(d.label);
        auto j = std::get<2>(d.label);
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
        if (stat != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Receive from GPU failed. (19)");
        data_movement_timer.stop();
      } else if (task_type == 0) {
        generate_tasks_lazy(nblocks, min_queue_size, max_queue_size, g, label_map, map_lock, ctx, graph_size, resume_generation_condition, condition_lock);
      } else {
        GALOIS_DIE("Unrecognized task type.");
      }

      updating_tasks_timer.start();
      {
        std::lock_guard<std::mutex> lock{map_lock};
        for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (!(--g.getData(dst, galois::MethodFlag::UNPROTECTED).waiting_on)) {
            ctx.push(dst);
          }
        }
        // Now remove this node entirely.
        g.removeNode(n);
        label_map.erase(d.label);
      
        if ((graph_size--) == min_queue_size) {
          resume_generation_condition.notify_one();
        }
      }
      updating_tasks_timer.stop();
    },
    galois::loopname("cholesky_tasks"), galois::wl<PSChunk>(),
    galois::no_conflicts()
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
    if (stat != cudaSuccess) GALOIS_DIE("Receive status after Cholesky on block failed. (7)");
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
    if (stat2 != CUBLAS_STATUS_SUCCESS) GALOIS_DIE("Receive failed.");
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
  */
}
