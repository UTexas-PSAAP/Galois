/** Residual based Page Rank -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Compute pageRank using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/gstl.h"

#include "Galois/Runtime/CompilerHelperFunctions.h"
#include "Galois/Runtime/Tracer.h"

#ifdef __GALOIS_VERTEX_CUT_GRAPH__
#include "Galois/Dist/vGraph.h"
#else
#include "Galois/Dist/hGraph.h"
#endif

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Dist/DistBag.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
struct CUDA_Worklist cuda_wl;

enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};
std::string personality_str(Personality p) {
   switch (p) {
   case CPU:
      return "CPU";
   case GPU_CUDA:
      return "GPU_CUDA";
   case GPU_OPENCL:
      return "GPU_OPENCL";
   }
   assert(false&& "Invalid personality");
   return "";
}
#endif

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));
#endif
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.000001));
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(1000));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
static cll::opt<unsigned> scalegpu("scalegpu", cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
#endif


static const float alpha = (1.0 - 0.85);
//static const float TOLERANCE = 0.01;
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;

};

#ifdef __GALOIS_VERTEX_CUT_GRAPH__
typedef vGraph<PR_NodeData, void> Graph;
#else
typedef hGraph<PR_NodeData, void> Graph;
#endif
typedef typename Graph::GraphNode GNode;

typedef GNode WorkItem;

struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		reset_CUDA_context(cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), ResetGraph{ &_graph }, Galois::loopname("reset"));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = 0;
    sdata.nout = 0;
    sdata.residual = 0;
  }
};

struct InitializeGraph {
  const float &local_alpha;
  Graph* graph;

  InitializeGraph(const float &_alpha, Graph* _graph) : local_alpha(_alpha), graph(_graph){}
  void static go(Graph& _graph) {
      	struct Syncer_0 {
      		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.residual;
      		}
      		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				{ Galois::atomicAdd(node.residual, y);}
      		}
      		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
      			else if (personality == CPU)
      		#endif
      				{node.residual = 0 ; }
      		}
      		typedef float ValTy;
      	};
      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
      		InitializeGraph_cuda(alpha, cuda_ctx);
      	} else if (personality == CPU)
      #endif
      Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ alpha, &_graph }, Galois::loopname("Init"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"));
      _graph.sync_push<Syncer_0>();
      
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = local_alpha;
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

    if(sdata.nout > 0 ){
      float delta = sdata.value*(1-local_alpha)/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};

template <typename GraphTy>
struct Get_info_functor : public Galois::op_tag {
	GraphTy &graph;
	struct Syncer_0 {
		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.residual;
		}
		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				{ Galois::atomicAdd(node.residual, y);}
		}
		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
			else if (personality == CPU)
		#endif
				{node.residual = 0 ; }
		}
		typedef float ValTy;
	};
	Get_info_functor(GraphTy& _g): graph(_g){}
	unsigned operator()(GNode n) const {
		return graph.getHostID(n);
	}
	GNode getGNode(uint32_t local_id) const {
		return GNode(graph.getGID(local_id));
	}
	uint32_t getLocalID(GNode n) const {
		return graph.getLID(n);
	}
	void sync_graph(){
		 sync_graph_static(graph);
	}
	void static sync_graph_static(Graph& _graph) {

		_graph.sync_push<Syncer_0>();
	}
};

struct PageRank {
  const float &local_alpha;
  cll::opt<float> &local_tolerance;
  Graph* graph;

  PageRank(cll::opt<float> &_tolerance, const float &_alpha, Graph* _g): local_tolerance(_tolerance), local_alpha(_alpha), graph(_g){}
  void static go(Graph& _graph) {
     #ifdef __GALOIS_HET_CUDA__
     	if (personality == GPU_CUDA) {
     		Galois::Timer T_compute, T_comm_syncGraph, T_comm_bag;
     		unsigned num_iter = 0;
     		auto __sync_functor = Get_info_functor<Graph>(_graph);
     		typedef Galois::DGBag<GNode, Get_info_functor<Graph> > DBag;
     		DBag dbag(__sync_functor);
     		auto &local_wl = DBag::get();
     		T_compute.start();
     		cuda_wl.num_in_items = _graph.getNumOwned();
     		for (int __i = 0; __i < cuda_wl.num_in_items; ++__i) cuda_wl.in_items[__i] = __i;
     		if (cuda_wl.num_in_items > 0)
     			PageRank_cuda(alpha, tolerance, cuda_ctx);
     		T_compute.stop();
     		T_comm_syncGraph.start();
     		__sync_functor.sync_graph();
     		T_comm_syncGraph.stop();
     		T_comm_bag.start();
     		dbag.set_local(cuda_wl.out_items, cuda_wl.num_out_items);
     		dbag.sync();
     		cuda_wl.num_out_items = 0;
     		T_comm_bag.stop();
     		//std::cout << "[" << Galois::Runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " T_compute : " << T_compute.get() << "(msec) T_comm_syncGraph : " << T_comm_syncGraph.get() << "(msec) T_comm_bag : " << T_comm_bag.get() << "(msec) \n";
     		while (!dbag.canTerminate()) {
     		++num_iter;
     		cuda_wl.num_in_items = local_wl.size();
     		//std::cout << "[" << Galois::Runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " Total items to work on : " << cuda_wl.num_in_items << "\n";
     		T_compute.start();
     		std::copy(local_wl.begin(), local_wl.end(), cuda_wl.in_items);
     		if (cuda_wl.num_in_items > 0)
     			PageRank_cuda(alpha, tolerance, cuda_ctx);
     		T_compute.stop();
     		T_comm_syncGraph.start();
     		__sync_functor.sync_graph();
     		T_comm_syncGraph.stop();
     		T_comm_bag.start();
     		dbag.set_local(cuda_wl.out_items, cuda_wl.num_out_items);
     		dbag.sync();
     		cuda_wl.num_out_items = 0;
     		T_comm_bag.stop();
     		//std::cout << "[" << Galois::Runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " T_compute : " << T_compute.get() << "(msec) T_comm_syncGraph : " << T_comm_syncGraph.get() << "(msec) T_comm_bag : " << T_comm_bag.get() << "(msec) \n";
     		}
     	} else if (personality == CPU)
     #endif
     Galois::for_each(_graph.begin(), _graph.end(), PageRank(tolerance, alpha, &_graph), Galois::workList_version(), Galois::loopname("PageRank"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"), Get_info_functor<Graph>(_graph));
  }

  void operator()(WorkItem& src, Galois::UserContext<WorkItem>& ctx) const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    //sdata.residual = residual_old;
    if (sdata.nout > 0){
      float delta = residual_old*(1-local_alpha)/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        auto dst_residual_old = Galois::atomicAdd(ddata.residual, delta);

        //Schedule TOLERANCE threshold crossed.
        if((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance)) {
          ctx.push(WorkItem(graph->getGID(dst)));
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::Timer T_total, T_graph_load, T_init;
    std::vector<Galois::Timer> T_compute;
    T_compute.resize(numRuns);

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;
    //Parse arg string when running on multiple hosts and update/override personality
    //with corresponding value.
    if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
      case 'g':
        personality = GPU_CUDA;
        break;
      case 'o':
        assert(0);
        personality = GPU_OPENCL;
        break;
      case 'c':
      default:
        personality = CPU;
        break;
      }
#ifdef __GALOIS_SINGLE_HOST_MULTIPLE_GPUS__
      if (gpu_device == -1) {
        gpu_device = 0;
        for (unsigned i = 0; i < my_host_id; ++i) {
          if (personality_set.c_str()[i] != 'c') ++gpu_device;
        }
      }
#endif
      for (unsigned i=0; i<personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c') 
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif

    T_total.start();

    T_graph_load.start();
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
    Graph hg(inputFile, partFolder, net.ID, net.Num, scalefactor);
#else
    Graph hg(inputFile, net.ID, net.Num, scalefactor);
#endif
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = hg.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, &cuda_wl, m);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    T_graph_load.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    std::cout << "[" << net.ID << "] PageRank::go run1 called\n";
    T_compute[0].start();
    PageRank::go(hg);
    T_compute[0].stop();

    for (unsigned i = 1; i < numRuns; ++i) {
      Galois::Runtime::getHostBarrier().wait();
      ResetGraph::go(hg);
      InitializeGraph::go(hg);

      std::cout << "[" << net.ID << "] PageRank::go run" << i+1 << " called\n";
      T_compute[i].start();
      PageRank::go(hg);
      T_compute[i].stop();
    }

    T_total.stop();

    double mean_time = 0;
    for (unsigned i = 0; i < numRuns; ++i) {
      mean_time += T_compute[i].get();
    }
    mean_time /= numRuns;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " Graph : " << T_graph_load.get() << " Init : " << T_init.get();
    for (unsigned i = 0; i < numRuns; ++i) {
      std::cout << " Pagerank " <<  i << " : " << T_compute[i].get();
    }
    std::cout << " Pagerank mean of " << numRuns << " runs : " << mean_time << " (msec)\n\n";

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), get_node_value_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
