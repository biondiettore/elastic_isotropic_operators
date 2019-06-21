#ifndef BORN_ELASTIC_GPU_H
#define BORN_ELASTIC_GPU_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu.h"
#include "fdParamElastic.h"
#include "seismicElasticOperator2D.h"
#include "interpTimeLinTbb.h"
#include "BornElasticGpuFunctions.h"

using namespace SEP;
//! Propogates one elastic wavefield for one shot on one gpu.
/*!
 A more elaborate description of the class.
*/
class BornElasticGpu : public seismicElasticOperator2D<SEP::float3DReg, SEP::float3DReg> {

	protected:

		std::shared_ptr<float4DReg> _srcWavefield, _secWavefield;

	public:
	  	//! Constructor.
		BornElasticGpu(std::shared_ptr<fdParamElastic> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		//! Mutators.
		void setAllWavefields(int wavefieldFlag);

    	//! QC
		virtual bool checkParfileConsistency(std::shared_ptr<SEP::float3DReg> model, std::shared_ptr<SEP::float3DReg> data) const;

    	//! FWD
    	void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

		//! ADJ
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
		~BornElasticGpu(){};

		//! Accesor
		std::shared_ptr<float4DReg> getSrcWavefield() { return _srcWavefield; }
    	std::shared_ptr<float4DReg> getSecWavefield() { return _secWavefield; }


};

#endif
