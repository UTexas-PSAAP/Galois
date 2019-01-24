#pragma once

#include <atomic>
#include <cmath>
//#include <compare>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/MorphGraph.h"
#include "galois/runtime/Context.h"
#include "galois/substrate/PerThreadStorage.h"

using PSChunk = galois::worklists::PerSocketChunkFIFO<1>;

struct task_data {
public:
  // Refcount that determines when the actual task object will be deleted.
  // The task graph itself also owns a reference to each task and releases
  // that reference once the task is completed.
  // Note: The creator of the task must keep a reference to
  // it if they want to create any additional tasks that depend on it.
  std::atomic<std::size_t> refcount;
  // This counter tracks how many tasks the current one is waiting on.
  // The task must be queued as "ready" as soon as this counter
  // reaches 0. When the counter reaches 0, no further modifications
  // should be made to the current task since the runtime is allowed
  // to start executing it at any time after the counter reaches 0.
  // To maintain that consistency, if modifications of any kind are
  // needed for a given task, they must be made by ancestor tasks
  // of task being modified.
  // The value std::numeric_limits<std::size_t>::max() is reserved
  // and is used to mark the given task as completed.
  std::atomic<std::size_t> remaining_dependencies;
  // Bleargh. No obvious way to avoid this at the moment.
  // This lock covers flagging this task as done 
  // for this node and adding edges to this node.
  // Hopefully the explicit lock can be removed
  // when these data structures are implemented as
  // standalone stuff instead of using morph_graph and for_each.
  std::mutex finished_and_edges_lock;
  // The task's actual work is done by calling
  // operation(closure);
  void (*operation)(void*, void*);
  void *closure;
  // Non-copyable, non-movable, and non-default-constructible.
  // Non-default-constructible because it's not clear what that
  // would even mean.
  // Non-movable because user-held references shouldn't be invalidated.
  // Non-copyable is primarily to prevent unintentionally
  // passing by value. An explicit method for copying could
  // maybe be useful at some future point, but that's still something
  // that will need to be done with an explicit method call.
  task_data() = delete;
  task_data(task_data&) = delete;
  task_data(task_data&&) = delete;
  // THe only allowed constructor...
  task_data(std::size_t initial_refcount,
            std::size_t number_of_dependencies,
            void (*op)(void*, void*),
            void* clos) noexcept:
    refcount(initial_refcount),
    remaining_dependencies(number_of_dependencies),
    finished_and_edges_lock(),
    operation(op),
    closure(clos) {}
  // Actually run the given task.
  // This isn't the full interface for running a given task.
  // All this does is encapsulate the task's ABI into something we
  // can call inside the code that handles things like decrementing
  // the waiting counters on dependent tasks.
  // TODO: How do we handle exceptions?
  // Too much about C++ exceptions is compiler specific,
  // so something custom is probably necessary here.
  // Once we're not calling user-provided things, C++
  // exceptions are fine for propagating the fact that
  // something went wrong, but at the boundary between
  // user and runtime code something else is needed.
  // These are only expected to work with T = user_context_t,
  // but this saves some complexity in forward declarations for now.
  template <typename T>
  void operator()(T &ctx) {
    // The dependencies should have completed and this task
    // should not have already executed.
    assert(!remaining_dependencies);
    operation(reinterpret_cast<void*>(&ctx), closure);
  }
  template <typename T>
  void execute(T &ctx) { (*this)(ctx); }
  // Easter egg...
  template <typename T>
  void exeggcute(T &ctx) { (*this)(ctx); }
};

using task_graph_base_t = galois::graphs::MorphGraph<task_data, void, true>::with_no_lockable<true>::type;
using task_graph_base_node_t = task_graph_base_t::GraphNode;

// TODO: set up a shared object for this instead of
// dumping it into everything that includes the header.
// Currently it's only included in the Python extension though,
// so dumping a global into an anonymous namespace is no big deal.
// That or... maybe roll these together into a tasking runtime class?
// It may not necessarily be relocatable though.
namespace {
  static galois::SharedMemSys Galois_runtime;
  static task_graph_base_t task_graph_base;
}

void task_incref(task_graph_base_node_t node) noexcept {
  auto &data = ::task_graph_base.getData(node, galois::MethodFlag::UNPROTECTED);
  data.refcount.fetch_add(1, std::memory_order_relaxed);
}

void task_decref(task_graph_base_node_t node) noexcept {
  auto &data = ::task_graph_base.getData(node, galois::MethodFlag::UNPROTECTED);
  // A refcount that's already 0 indicates either memory
  // corruption or a lost reference.
  if (!data.refcount.fetch_sub(1, std::memory_order_relaxed)) {
    ::task_graph_base.removeNode(node);
  }
}

// This type is a RAII wrapper around an owned reference to the underlying
// node in the task graph.
// NOTE: It'd be nice to make the task graph non-static, but then the destructors
// for tasks can't just reference the static task graph, which messes up RAII here
// unless we also carry a pointer to the graph in which a given task resides.
// Maybe make the task type carry a pointer but not the task_ref type...
struct task {
  // The node type of a given graph may be just an index or pointer.
  // It's conceptually some sort of reference though.
  task_graph_base_node_t node_ref;
  // For the time being, disallow nulls and disable the default constructor.
  // How exactly that can work will ultimately depend on the underlying
  // structure of task_graph_base_node_t, and whether or not that even
  // should be allowed needs further consideration.
  task() = delete;
  explicit task(task_graph_base_node_t const node) noexcept : node_ref(node){
    task_incref(node_ref);
  }
  task(task const &other) noexcept : node_ref(other.node_ref) {
    auto &data = ::task_graph_base.getData(node_ref, galois::MethodFlag::UNPROTECTED);
    data.refcount++;
  }
  // For the time being we're assuming very little about the
  // type of the underlying node reference. If that supports
  // null values, then a moved from state can be added and used.
  task(task &&other) = delete;
  ~task() noexcept {
    task_decref(node_ref);
  }
  task &operator=(task &other) = delete;
  task &operator=(task&&) = delete;
  // Do this the old way for now.
  friend bool operator==(task const &first, task const &second) noexcept {
    return first.node_ref == second.node_ref;
  }
  friend bool operator!=(task const &first, task const &second) noexcept {
    return first.node_ref != second.node_ref;
  }
};

// Borrowed reference to a node in the task graph instead of an owned reference.
// Hides the node_ref type from the user sumewhat. Can also be used to add
// extra assertions about the validity of the reference at some point.
struct task_ref {
  task_graph_base_node_t node_ref;
  explicit task_ref(task_graph_base_node_t const node) noexcept : node_ref(node) {}
  task_ref(task_ref const &other) noexcept : node_ref(other.node_ref) {}
  task_ref(task const &other) noexcept : node_ref(other.node_ref) {}
  task_ref &operator=(task_graph_base_node_t node) noexcept {
    node_ref = node;
    return *this;
  }
  task_ref &operator=(task_ref const &other) noexcept {
    node_ref = other.node_ref;
    return *this;
  }
  task_ref &operator=(task const &other) noexcept {
    node_ref = other.node_ref;
    return *this;
  }
  friend bool operator==(task_ref const &first, task_ref const &second) noexcept {
    return first.node_ref == second.node_ref;
  }
  friend bool operator!=(task_ref const &first, task_ref const &second) noexcept {
    return first.node_ref != second.node_ref;
  }
};

// Type of use context passed to operator inside Galois loop.
// This is needed to push new tasks/work items.
using user_context_t = galois::UserContext<task_ref>;

// Not currently user-facing.
// This creates a node in the task graph but does not take care of any
// of the logic necessary to register it as ready if it has no dependencies.
// For that reason, it's not currently intended to be user-facing.
// Note! initial_refcount will default to 1 since the task graph
// itself owns a reference. An additional incref for the reference
// returned happens when the "task" is constructed.
task create_task_node(void (*operation)(void*, void*), void *closure, std::size_t num_deps, task_ref *deps, std::size_t initial_refcount = 1) {
  task_graph_base_node_t node_ref = ::task_graph_base.createNode(initial_refcount, num_deps, operation, closure);
  ::task_graph_base.addNode(node_ref);
  task_data &data = ::task_graph_base.getData(node_ref, galois::MethodFlag::UNPROTECTED);
  // Passing nullptr as "dependencies" is allowed if there are no dependencies to add.
  // If there are dependencies though, that pointer must not be null.
  assert(!num_deps || deps != nullptr);
  for (std::size_t i = 0; i < num_deps; i++) {
    task_graph_base_node_t dep = deps[i].node_ref;
    // As is, this needs some kind of locking, but using
    // std::mutex here seems like major overkill.
    // Maybe there's a way to do this without locks once
    // it no longer has a morph-graph underneath.
    task_data &other_data = ::task_graph_base.getData(dep, galois::MethodFlag::UNPROTECTED);
    std::unique_lock<std::mutex> lk{other_data.finished_and_edges_lock};
    if (!(~other_data.remaining_dependencies)) {
      data.remaining_dependencies--;
    } else {
      ::task_graph_base.addEdge(dep, node_ref);
    }
  }
  // NOTE!
  // If the remaining dependency count hits 0 after the above loop,
  // the task has to be queued as ready, but the caller
  // is responsible for doing that since this routine may be
  // used to create ready tasks outside of galois parallel contexts.
  return node_ref;
}

// Not user-facing.
void mark_complete(task_ref tsk) {
  auto &data = ::task_graph_base.getData(tsk.node_ref, galois::MethodFlag::UNPROTECTED);
  std::unique_lock<std::mutex> lk{data.finished_and_edges_lock};
  data.remaining_dependencies = std::numeric_limits<std::size_t>::max();
}

// Not user-facing.
// Current limitation: Must be called from inside a Galois parallel block.
void enqueue_ready_task(void *ctx, task_ref ready) {
  auto &user_context = *reinterpret_cast<user_context_t*>(ctx);
  user_context.push(ready);
}

// Not user-facing.
// Current limitation: Must be called from inside a Galois parallel block.
void notify_dependers(void *ctx, task_ref tsk) {
  for (auto e : ::task_graph_base.edges(tsk.node_ref, galois::MethodFlag::UNPROTECTED)) {
    auto dst = ::task_graph_base.getEdgeDst(e);
    auto &dst_data = ::task_graph_base.getData(dst, galois::MethodFlag::UNPROTECTED);
    if (!(dst_data.remaining_dependencies.fetch_sub(1, std::memory_order_relaxed))) {
      enqueue_ready_task(ctx, task_ref(dst));
    }
  }
}

// Not user-facing.
// Actually run a task, including the notifying logic, etc.
// Current limitation: Must be called from inside a Galois parallel block.
void run_task(user_context_t &ctx, task_ref tsk) {
  auto &data = ::task_graph_base.getData(tsk.node_ref, galois::MethodFlag::UNPROTECTED);
  // Run it.
  data(ctx);
  mark_complete(tsk);
  notify_dependers(reinterpret_cast<void*>(&ctx), tsk);
  // Release the reference owned by the task graph.
  // This reference represents the fact that the
  // execution engine is waiting to run the task
  // and requires that it not be deleted until it
  // has actually run.
  task_decref(tsk.node_ref);
}

// User facing!
// TODO: This interface should be revised. It shouldn't be necessary to
// have a separate "create task" function depending on whether or not the
// execution engine is already running. That's an artifact of the fact that
// this is a thin wrapper around a modified Galois for_each loop. Inside
// the for_each loop we expect a ready task to be queued immediately
// and queueing it requires the Galois user context object.
// This routine is to create and run a "generation task" that
// generates all the other tasks using the other API.
void run_generation_task(void (*operation)(void*, void*), void *closure) {
  auto base_task = create_task_node(operation, closure, 0, nullptr);
  task_ref base_ref = base_task;
  galois::for_each(galois::iterate({base_ref}),
    [](task_ref t, user_context_t& ctx) {
      run_task(ctx, t);
    },
//    galois::loopname("run_tasks"),
    galois::wl<galois::worklists::PerSocketChunkFIFO<16>>(),
    galois::no_conflicts()
  );
}

// User facing way to create tasks from inside another task.
// Current limitation, must be called from inside a Galois parallel context,
// it is only good for creating tasks from a currently running task.
task create_task(void *ctx, void (*operation)(void*, void*), void *closure, std::size_t num_deps, task_ref *dependencies) {
  auto new_task = create_task_node(operation, closure, num_deps, dependencies);
  auto &remaining_deps = ::task_graph_base.getData(new_task.node_ref, galois::MethodFlag::UNPROTECTED).remaining_dependencies;
  if (!remaining_deps) {
    enqueue_ready_task(ctx, new_task);
  }
  return new_task;
}

