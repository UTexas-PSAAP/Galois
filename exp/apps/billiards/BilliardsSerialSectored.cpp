#include "Billiards.h"

class BilliardsSerialSectored: public Billiards {
  typedef std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator> PriorityQueue;
public:

  virtual const std::string version () const { return "Serial Ordered with Priority Queue"; }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void processEvent (Event& e, TableSectored& table, std::vector<Event>& addList, const double endtime) {
      addList.clear ();
      e.simulate ();
      table.addNextEvents (e, addList, endtime);
  }

  virtual size_t runSim (TableSectored& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;

    while (!pq.empty ()) {

      Event e = pq.top ();
      pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      processEvent (e, table, addList, endtime);

      for (std::vector<Event>::iterator i = addList.begin (), ei = addList.end ();
          i != ei; ++i) {

        pq.push (*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str () << std::endl;
        }
      }

      if (enablePrints) {
        table.printState (std::cout);
      }

      ++iter;
    }

    return iter;

  }
};

int main (int argc, char* argv[]) {
  BilliardsSerialSectored s;
  s.run (argc, argv);
  return 0;
}
