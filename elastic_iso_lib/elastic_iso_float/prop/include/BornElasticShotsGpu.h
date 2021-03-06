#ifndef BORN_SHOTS_GPU_H
#define BORN_SHOTS_GPU_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu.h"
#include "fdParamElastic.h"
#include "operator.h"

using namespace SEP;

class BornElasticShotsGpu : public Operator<SEP::float3DReg, SEP::float4DReg> {

	private:
		int _nExp, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber, _useStreams;
		std::shared_ptr<SEP::float3DReg> _elasticParam;
		std::shared_ptr<paramObj> _par;
	    std::vector<std::shared_ptr<SEP::float3DReg>> _sourcesSignalsVector;
		std::vector<std::shared_ptr<spaceInterpGpu>> _sourcesVectorCenterGrid, _sourcesVectorXGrid, _sourcesVectorZGrid, _sourcesVectorXZGrid;
		std::vector<std::shared_ptr<spaceInterpGpu>> _receiversVectorCenterGrid, _receiversVectorXGrid, _receiversVectorZGrid, _receiversVectorXZGrid;
		std::shared_ptr<fdParamElastic> _fdParamElastic;
		std::vector<int> _gpuList;

	protected:
		std::shared_ptr<SEP::float4DReg> _srcWavefield, _secWavefield;

	public:
	    /* Overloaded constructors */
		BornElasticShotsGpu(std::shared_ptr<SEP::float3DReg> elasticParam, std::shared_ptr<paramObj> par,
						   std::vector<std::shared_ptr<SEP::float3DReg>> sourcesSignalsVector,
			 			   std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorCenterGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorZGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXZGrid,
			               std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorCenterGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorZGrid,
						   std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXZGrid);

		/* Destructor */
		~BornElasticShotsGpu(){};

		/* Create Gpu list */
		void createGpuIdList();

	    /* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<SEP::float3DReg> model, std::shared_ptr<SEP::float4DReg> data) const;
		// void forwardWavefield(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float4DReg> data);
		void adjoint(const bool add, std::shared_ptr<SEP::float3DReg> model, const std::shared_ptr<SEP::float4DReg> data) const;
		// void adjointWavefield(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float4DReg> data);

		/* Accessor */
		std::shared_ptr<float4DReg> getSrcWavefield() { return _srcWavefield; }
		std::shared_ptr<float4DReg> getSecWavefield() { return _secWavefield; }

		/* Mutators */
		void setBackground(std::shared_ptr<float3DReg> elasticParam){ _fdParamElastic = std::make_shared<fdParamElastic>(elasticParam, _par); }

};

#endif
