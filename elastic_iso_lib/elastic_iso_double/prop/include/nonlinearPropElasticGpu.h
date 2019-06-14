#ifndef NL_PROP_ELASTIC_GPU_H
#define NL_PROP_ELASTIC_GPU_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu.h"
#include "fdParamElastic.h"
#include "seismicElasticOperator2D.h"
#include "interpTimeLinTbb.h"
#include "nonlinearPropElasticGpuFunctions.h"

using namespace SEP;
//! Propogates one elastic wavefield for one shot on one gpu.
/*!
 A more elaborate description of the class.
*/
class nonlinearPropElasticGpu : public seismicElasticOperator2D<SEP::double3DReg, SEP::double3DReg> {

	protected:

		std::shared_ptr<double4DReg> _wavefield;

	public:
    //! Constructor.
		nonlinearPropElasticGpu(std::shared_ptr<fdParamElastic> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		//! Mutators.
		void setAllWavefields(int wavefieldFlag);

  	//! QC
		virtual bool checkParfileConsistency(std::shared_ptr<SEP::double3DReg> model, std::shared_ptr<SEP::double3DReg> data) const;

  	//! FWD
  	void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;

	  //! ADJ
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		//! Desctructor
		~nonlinearPropElasticGpu(){};

		//! Accesor
		std::shared_ptr<double4DReg> getWavefield() { return _wavefield; }

};

#endif
