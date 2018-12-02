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

#include <boost/coroutine2/coroutine.hpp>

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

#define GRAVITATIONAL_CONSTANT 6.6740831E-11

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
  generate_all,
  recurse_summary,
  compute_summary,
  signal_summaries_completed,
  compute_near_forces,
  compute_far_forces
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
  // One of it's primary purposes is to guarantee that updates that need to happen in both
  // places appear as atomic to the observer.
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

  // Helper function. DOESN'T LOCK.
  // Create a new node with the given dependencies,
  // pushing it as ready if it's ready,
  // though not flushing the worklist.
  task_graph_node new_task_node_from_label(auto &ctx, task_label label, std::size_t num_deps, task_label *dependencies, std::size_t prealloc_size = 0) {
    auto n = g.createNode(std::max(prealloc_size, num_deps), label);
    g.addNode(n);
    label_map[label] = n;
    for (std::size_t i = 0; i < num_deps; i++) {
      auto &dep = dependencies[i];
      if (label_map.count(dep)) {
        g.addEdge(label_map[dep], n);
      } else {
        if (!(--g.getData(n).remaining_dependencies)) {
          ctx.push(n);
        }
      }
    }
    return n;
  }

  // User-facing task registration interface.
  // Create a task that depends on the given things.
  void register_task(auto &ctx, task_label label, std::size_t num_deps, task_label* dependencies) {
    std::unique_lock<std::mutex> lk{lock};
    new_task_node_from_label(ctx, label, num_deps, dependencies);
    ctx.send_work();
    condition.wait(lk, [&](){return g.size() < min_task_num;});
  }

  // Hard to describe. This one needs some more thought/description.
  // It implements recursion by allowing a task to perform a "blocking call"
  // to a series of other ready tasks. It changes its own label to the given
  // new label, and adds the created tasks as things it depends on.
  // The things that depend on it remain the same.
  // Conceptually, the new label is the label for the new task
  // representing the work that happens after the "blocking calls" finish.
  // Note that this does not actually block the current execution thread.
  void blocking_calls(auto &ctx, task_graph_node node, task_label new_label, std::size_t num_called, task_label *called) {
    std::unique_lock<std::mutex> lk{lock};
    // Switch the label on the node from the current (now completed) task
    // instead of creating a new node and then forwarding its dependencies.
    // Conceptually that's what this does though.
    auto &data = g.getData(node, galois::MethodFlag::UNPROTECTED);
    auto &label = data.label;
    label_map.erase(old_label);
    label = new_label;
    label_map[label] = node;
    data.remaining_dependencies = num_called;
    // Create the requested new tasks and make the newly labeled task depend on them.
    for (std::size_t i = 0; i < num_called; i++) {
      // Make the new replacement for this node depend on the created tasks.
      auto called_node = new_task_node_from_label(ctx, called[i], 0, nullptr, 1);
      g.addEdge(called_node, node);
    }
    // If nothing was actually called, the newly created task is ready
    // and must be marked as such since no other task will mark it ready later.
    if (num_called == 0) {
      ctx.push(node);
    }
    // Now that all that is done, flush the work list.
    // TODO: Confirm that this can be done at a finer granularity
    // and move it in to the new task node creation routine.
    ctx.send_work();
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

  // Remove a task, forwarding the things that depend on it to a
  // different task instead of marking them as ready.
  void remove_and_forward(task_graph_node node, task_label new_label) {
    std::unique_lock<std::mutex> lk{lock};
    auto &node_data = g.getData(node, galois::MethodFlag::UNPROTECTED);
    label_map.erase(node_data.label);
    label_map[new_label] = node;
    node_data.label = new_label;
  }
};

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

// TODO: This should be a bit more tolerant to floating point rounding errors.
bool separated_from(tree_t &tree, tree_node_t source, tree_node_t other) {
  double *bounds = tree.getData(source, galois::MethodFlag::UNPROTECTED).bounds;
  double *other_bounds = tree.getData(other, galois::MethodFlag::UNPROTECTED).bounds;
  for (std::size_t i = 0; i < dims; i++) {
    double mn = 2 * bounds[2 * i] - bounds[2 * i + 1];
    double mx = 2 * bounds[2 * i + 1] - bounds[2 * i];
    double omn = other_bounds[2 * i];
    double omx = other_bounds[2 * i + 1];
    if (!((mn < omn && omn < mx) || (mn < omx && omx < mx))) {
      return true;
    }
  }
  return false;
}

bool is_final(tree_t &tree, tree_node_t node) {
  return (tree.getData(node, galois::MethodFlag::UNPROTECTED).num_pts <= max_group_size);
}

template <typename T>
void apply_to_neighbors_impl(tree_t &tree, tree_node_t source, tree_node_t other, const T &operation) {
  if (is_final(tree, other) || separated_from(tree, source, other)) {
    operation(tree, other);
  } else {
    for (auto edge : tree.edges(other, galois::MethodFlag::UNPROTECTED)) {
      auto child = tree.getEdgeDst(edge);
      apply_to_neighbors_impl(tree, source, child, operation);
    }
  }
}

template <typename T>
void apply_to_neighbors(tree_t &tree, tree_node_t source, const T &operation) {
  auto current = source;
  auto current_data = &tree.getData(current, galois::MethodFlag::UNPROTECTED);
  while (current_data->parent != nullptr) {
    for (auto edge : tree.edges(current_data->parent, galois::MethodFlag::UNPROTECTED)) {
      auto sibling = tree.getEdgeDst(edge);
      if (sibling == current) continue;
      apply_to_neighbors_impl(tree, source, sibling, operation);
    }
    current = current_data->parent;
    current_data = &tree.getData(current, galois::MethodFlag::UNPROTECTED);
  }
}

template <typename T>
void apply_to_leaves(tree_t &tree, tree_node_t source, const T &operation) {
  if (is_final(tree, source)) {
    operation(tree, source);
  } else {
    for (auto edge : tree.edges(source, galois::MethodFlag::UNPROTECTED)) {
      auto child = tree.getEdgeDst(edge);
      apply_to_leaves(tree, child, operation);
    }
  }
}

void print_node(tree_t &tree, tree_node_t leaf) {
  auto &data = tree.getData(leaf, galois::MethodFlag::UNPROTECTED);
  double *bounds = data.bounds;
  for (std::size_t i = 0; i < dims; i++) {
    std::cout << "(" << bounds[2 * i] << ", " << bounds[2 * i + 1] << ")";
  }
  std::cout << std::endl;
}

void print_leaves(tree_t &tree, tree_node_t root_node) {
  apply_to_leaves(tree, root_node, print_node);
  std::cout << std::endl;
}

tree_node_t first_leaf(tree_t &tree, tree_node_t node) {
  while(!is_final(tree, node)) {
    auto &data = tree.getData(node, galois::MethodFlag::UNPROTECTED);
    node = tree.getEdgeDst(*tree.edges(node, galois::MethodFlag::UNPROTECTED).begin());
  }
  return node;
}

void print_neighbors(tree_t &tree, tree_node_t node) {
  apply_to_neighbors(tree, node, print_node);
  std::cout << std::endl;
}

void print_neighbors_all(tree_t &tree, tree_node_t root) {
  apply_to_leaves(tree, root, print_neighbors);
}

void generate_tasks_lazy(task_graph &tsk, tree_t &tree, tree_node_t root, auto &ctx) {
  task_label root_recursion{recurse_summary, root};
  tsk.register_task(ctx, root_recursion, 0, nullptr);
  task_label signal_recursion_complete{signal_summaries_completed, root};
  tsk.register_task(ctx, signal_recursion_complete, 1, &root_recursion);
  // TODO: After switching so execution waits only on ready tasks, just generate these tasks in a single pass
  // on the leaf nodes. It'll deadlock if that's done before the switch in waiting semantics.
  apply_to_leaves(tree, root, [&](auto &tree, auto leaf) {
    tsk.register_task(ctx, task_label(compute_near_forces, leaf), 0, nullptr);
  });
  apply_to_leaves(tree, root, [&](auto &tree, auto leaf) {
    tsk.register_task(ctx, task_label(compute_far_forces, leaf), 1, &signal_recursion_complete);
  });
}

template <typename T, typename S, std::size_t dims>
void set_array(sa::strided_array<T, dims> a, S value) {
  static_assert(dims <= 2, "TODO: generalize array setting routine to work in higher dimensional arrays. Note: this is best done via an indexing refactor...");
  for (std::size_t i = 0; i < a.axes[0].shape; i++) {
    if constexpr (dims == 1) {
      a(i) = value;
    } else {
      set_array(a(i, sa::slice()), value);
    }
  }
}

double set_differences_and_coef(sa::strided_array<double, 1> p1, sa::strided_array<double, 1> p2, sa::strided_array<double, 1> tmp) {
  assert(p1.axes[0].shape > 0);
  assert(p1.axes[0].shape == p2.axes[0].shape);
  double dist2 = 0;
  for (std::size_t j = 0; j < p1.axes[0].shape; j++) {
    temp(i,j) = p1(j) - p2(j);
    dist2 += temp(i,j) * temp(i,j);
  }
  return GRAVITATIONAL_CONSTANT / (dist2 * std::sqrt(dist2));
}

enum force_add_mode {bidirectional, unidirectional};

template <force_add_mode mode>
struct force_between_impl {
  // This is an attempt to avoid repeating code between the unidirectional and bidirectional case.
  // This would be nicer with non-scoped constexpr-if, but whatever.
  if constexpr (mode == bidirectional) {
    static void points(sa::strided_array<double, 1> p1, sa::strided_array<double, 1> p2, sa::strided_array<double, 1> tmp, sa::strided_array<double, 1> f1, sa::strided_array<double, 1> f2) {
      auto coef = set_differences_and_distance2(p1, p2, tmp);
      for (std::size_t j = 0; j < p1.axes[0].shape; j++) {
        auto force_in_direction = temp(j) * coef;
        f1(j) -= force_in_direction;
        f2(j) += force_in_direction;
      }
    }
  } else {
    static_assert(mode == unidirectional);
    static void points(sa::strided_array<double, 1> p1, sa::strided_array<double, 1> p2, sa::strided_array<double, 1> tmp, sa::strided_array<double, 1> f2) {
      auto coef = set_differences_and_distance(p1, p2, tmp);
      for (std::size_t j = 0; j < p1.axes[0].shape; j++) {
        auto force_in_direction = temp(j) * coef;
        f2(j) += force_in_direction;
      }
    }
  }

  if constexpr (mode == bidirectional) {
    static void point_and_cloud(sa::strided_array<double, 1> p1, sa::strided_array<double, 2> ps2, sa::strided_array<double, 1> tmp, sa::strided_array<double, 1> f1, sa::strided_array<double, 2> fs2) {
      for (std::size_t i = 0; i < p2.axes[0]; i++) {
        points(p1, ps2(i), tmp, f1, fs2(i));
      }
    }
  } else {
    static_assert(mode == unidirectional);
    static void point_and_cloud(sa::strided_array<double, 1> p1, sa::strided_array<double 2> ps2, sa::strided_array<double 1> tmp, sa::strided_array<double, 2> fs2) {
      for (std::size_t i = 0; i < p2.axes[0]; i++) {
        points(p1, ps2(i), tmp, fs2(i));
      }
    }
  }

  if constexpr (mode == bidirectional) {
    static void clouds(sa::strided_array<double, 2> ps1, sa::strided_array<double, 2> ps2, sa::strided_array<double, 1> tmp, sa::strided_array<double, 2> fs1, sa::strided_array<double, 2> fs2) {
      for (std::size_t i = 0; i < p1.axes[0].shape; i++) {
        point_and_cloud(ps1(i), ps2, tmp, fs1(i), fs2);
      }
    }
  } else {
    static_assert(mode == unidirectional);
    static void clouds(sa::strided_array<double, 2> ps1, sa::strided_array<double, 2> ps2, sa::strided_array<double, 1> tmp, sa::strided_array<double, 2> fs2) {
      for (std::size_t i = 0; i < p1.axes[0].shape; i++) {
        points(ps1(i), ps2, tmp, fs2);
      }
    }
  }

  if constexpr (mode == bidirectional) {
    template <std::size_t d1, std::size_t d2>
    static void force_between(sa::strided_array<double, d1> p1, sa::strided_array<double, d2> p2, sa::strided_array<double, 1> tmp, sa::strided_array<double, d1> f1, sa::strided_array<double, d2> f2) {
      static_assert(d1 <= d2, "Force of cloud on point not implemented.");
      if constexpr (d1 == 1 && d2 == 1) {
        points(p1, p2, tmp, f1, f2);
      } else if (d1 == 1 && d2 == 2) {
        point_and_cloud(p1, p2, tmp, f1, f2);
      } else if (d1 == 2 && d2 == 2) {
        clouds(p1, p2, tmp, f1, f2);
      } else {
        static_assert(false, "Unexpected dimensions of input arrays.");
      }
    }
  } else {
    static_assert(mode == unidirectional)
    static void force_between(sa::strided_array<double, d1> p1, sa::strided_array<double, d2> p2, sa::strided_array<double, 1> tmp, sa::strided_array<double, d2> f2) {
      static_assert(d1 <= d2, "Force of cloud on point not implemented.");
      if constexpr (d1 == 1 && d2 == 1) {
        points(p1, p2, tmp, f2);
      } else if (d1 == 1 && d2 == 2) {
        point_and_cloud(p1, p2, tmp, f2);
      } else if (d1 == 2 && d2 == 2) {
        clouds(p1, p2, tmp, f2);
      } else {
        static_assert(false, "Unexpected dimensions of input arrays.");
      }
    }
  }
};

template <force_add_mode mode, std::size_t d1, std::size_t d2>
using force_between = force_between_impl<mode>::template force_between<d1, d2>;

void intra_cloud_force(sa::strided_array<double, 2> cloud, sa::strided_array<double, 1> temp, sa::strided_array<double, 2> forces) {
  assert(cloud.axes[0].shape == forces.axes[0].shape);
  assert(forces.axes[0].shape > 0);
  for (std::size_t i = 1; i < cloud.axes[0].shape; i++) {
    force_between<bidirectional>(cloud(i, sa::slice()), cloud(sa::slice(i), sa::slice()), temp, forces(i, sa::slice()), forces(sa::slice(i), sa::slice));
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
  //std::cout << std::endl;
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

  //print_leaves(tree, root_node);
  //print_neighbors(tree, first_leaf(tree, root_node));
  //print_neighbors_all(tree, root_node);

  // Insert the generation task manually.
  // TODO: There should be some sort of machinery in the main task graph that hides this.
  auto generation_task = tasks.g.createNode(0, task_label(root_node, generate_all));
  tasks.g.addNode(init_node);

  // Set up temporary buffers used by the threads.
  galois::substrate::PerThreadStorage<std::unique_ptr<double>> temp_buffers{};
  sa::array_axes<1> tmp_axes{{{{std::size_t(dims), sizeof(double)}}}};
  galois::substrate::PerThreadStorage<sa::strided_array<double, 1>> temps{nullptr, tmp_axes};
  galois::on_each([&](unsigned int tid, unsigned int nthreads) {
    *temp_buffers.getLocal() = std::make_unique<double*>(dims);
    temps.getLocal()->data = (*temp_buffers.getLocal()).get();
  });

  galois::for_each(
    galois::iterate({generation_task}),
    [&](task_graph_node n, auto &ctx) {
      auto &current_task_data = tree.getData(n);
      auto task = current_task_data.task;
      auto tree_node = current_task_data.tree_node;
      auto &node_data = tree.getData(tree_node, galois::MethodFlag::UNPROTECTED);
      if (task == generate_all) {
        generate_tasks_lazy(tasks, tree, root_node, ctx);
      } else if (task == recurse_summary) {
        // TODO: Add an option for controlling where the recursion stops better.
        if (is_final(tree, tree_node)) {
          // Compute the summary, don't return, since this task will actually complete
          // instead of just forward its dependencies.
          node_data.summary = allocator.allocate(dims);
          for (std::size_t j = 0; j < dims; j++) {
            node_data.summary[j] = 0.;
          }
          auto local_points = points(sa::slice(start_idx, start_idx + node_data.num_points), sa::slice());
          for (std::size_t i = 0; i < node_data.num_points; i++) {
            for (std::size_t j = 0; j < dims; j++) {
              node_data_summary[j] += points(i, j);
            }
          }
          for (std::size_t j = 0; j < dims; j++) {
            node_data.summary[j] / node_data.num_points;
          }
        } else {
          // TODO: This seems a bit counterintuitive. Maybe more can be done to make it work cleanly.
          tasks.remove_and_forward(n, task_label(compute_summary, tree_node));
          for (auto e : tree.edges(tree_node)) {
            ;
          }
          // Dependencies have been forwarded and the recursion task
          // has been destroyed, so there's no need to remove the node,
          // so just return now.
          return;
        }
      } else if (task == compute_summary) {
        ;
      } else if (task == signal_recursion_complete) {
        ;
      } else if (task == compute_near_forces) {
        ;
      } else if (task == compute_far_forces) {
        ;
      } else {
        // Unrecognized task type.
        GALOIS_DIE("Unrecognized task type.");
        assert(false);
      }
      tasks.remove_task(n);
    },
    galois::loopname("barneshut_tasks"),
    galois::wl<PSChunk>(),
    galois::no_conflicts()
  )

;;;;;;;;;;;;;;;

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
}
