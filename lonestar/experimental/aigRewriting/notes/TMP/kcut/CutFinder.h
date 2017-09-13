#ifndef CUTFINDER_CUTFINDER_H_
#define CUTFINDER_CUTFINDER_H_

#include "Aig.h"
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"

namespace algorithm {

class CutFinder {

private:

	aig::Aig & aig;

public:

	CutFinder( aig::Aig & aig );
	virtual ~CutFinder();

	void run( int k, int c );
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* CUTFINDER_H_ */