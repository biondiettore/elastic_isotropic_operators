#ifndef NL_PROP_SHOTS_GPU_H
#define NL_PROP_SHOTS_GPU_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "double2DReg.h"
#include "double3DReg.h"
#include "double4DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu.h"
#include "fdParamElastic.h"
#include "operator.h"

using namespace SEP;

class nonlinearPropElasticShotsGpu : public Operator<SEP::double4DReg, SEP::double4DReg> {

	private:
		int _nExp, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber;
		std::shared_ptr<SEP::double3DReg> _elasticParam;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<spaceInterpGpu>> _sourcesVectorCenterGrid, _sourcesVectorXGrid, _sourcesVectorZGrid, _sourcesVectorXZGrid;
		std::vector<std::shared_ptr<spaceInterpGpu>> _receiversVectorCenterGrid, _receiversVectorXGrid, _receiversVectorZGrid, _receiversVectorXZGrid;
		std::shared_ptr<fdParamElastic> _fdParamElastic;
		std::vector<int> _gpuList;
	protected:
		std::shared_ptr<SEP::double4DReg> _wavefield;

	public:

		/* Overloaded constructors */
		nonlinearPropElasticShotsGpu(std::shared_ptr<SEP::double3DReg> elasticParam, std::shared_ptr<paramObj> par,
			 														std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorCenterGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorZGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorCenterGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorZGrid,
																	std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXZGrid);

		/* Destructor */
		~nonlinearPropElasticShotsGpu(){};

		/* Create Gpu list */
		void createGpuIdList();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<SEP::double4DReg> model, std::shared_ptr<SEP::double4DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double4DReg> data);
		void adjoint(const bool add, std::shared_ptr<SEP::double4DReg> model, const std::shared_ptr<SEP::double4DReg> data) const;
		void adjointWavefield(const bool add, std::shared_ptr<double4DReg> model, const std::shared_ptr<double4DReg> data);

		/* Accessor */
		std::shared_ptr<SEP::double4DReg> getWavefield(){ return _wavefield; }

};

#endif
