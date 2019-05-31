#ifndef SEISMIC_ELASTIC_OPERATOR_2D_H
#define SEISMIC_ELASTIC_OPERATOR_2D_H 1

#include "interpTimeLinTbb.h"
#include "operator.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "operator.h"
#include "fdParamElastic.h"
#include "spaceInterpGpu.h"
#include <omp.h>

using namespace SEP;

template <class V1, class V2>
class seismicElasticOperator2D : public Operator <V1, V2> {

	protected:

		std::shared_ptr<fdParamElastic> _fdParamElastic;
		std::shared_ptr<spaceInterpGpu> _sourcesCenterGrid, _sourcesXGrid, _sourcesZGrid, _sourcesXZGrid;
		std::shared_ptr<spaceInterpGpu> _receiversCenterGrid, _receiversXGrid, _receiversZGrid, _receiversXZGrid;
// 		std::shared_ptr<spaceInterpGpu> _sources_vx,_sources_vz,_sources_sigmaxx,_sources_sigmazz,_sources_sigmaxz;
//     std::shared_ptr<spaceInterpGpu> _receivers_vx, _receivers_vz, _receivers_sigmaxx, _receivers_sigmazz, _receivers_sigmaxz;
		int *_sourcesPositionRegCenterGrid, *_sourcesPositionRegXGrid, *_sourcesPositionRegZGrid, *_sourcesPositionRegXZGrid;
		int *_receiversPositionRegCenterGrid, *_receiversPositionRegXGrid, *_receiversPositionRegZGrid, *_receiversPositionRegXZGrid;
		int _nSourcesRegCenterGrid,_nSourcesRegXGrid,_nSourcesRegZGrid,_nSourcesRegXZGrid;
		int _nSourcesIrregCenterGrid,_nSourcesIrregXGrid,_nSourcesIrregZGrid,_nSourcesIrregXZGrid;
		int _nReceiversRegCenterGrid,_nReceiversRegXGrid,_nReceiversRegZGrid,_nReceiversRegXZGrid;
		int _nReceiversIrregCenterGrid,_nReceiversIrregXGrid,_nReceiversIrregZGrid,_nReceiversIrregXZGrid;
		int _nts;
		int _saveWavefield,_useStreams;
		int _iGpu, _nGpu, _iGpuId;
		std::shared_ptr<interpTimeLinTbb> _timeInterp;

    //these variables hold all five compnenets of elastic source signal. Should be a 3d reg
		std::shared_ptr<V2> _sourcesSignals, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2, _sourcesSignalsRegDtw;

	public:

		// QC
		virtual bool checkParfileConsistency(std::shared_ptr<V1> model, std::shared_ptr<V2> data) const = 0; // Pure virtual: needs to implemented in derived class

		// Sources
		void setSources(std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid); // This one is for the nonlinear modeling operator
		// void setSources(std::shared_ptr<spaceInterpGpu> sources, std::shared_ptr<V2> sourcesSignals); // For the other operators (Born + Tomo + Wemva)

		// Receivers
		void setReceivers(std::shared_ptr<spaceInterpGpu> receiversCenterGrid, std::shared_ptr<spaceInterpGpu> receiversXGrid, std::shared_ptr<spaceInterpGpu> receiversZGrid, std::shared_ptr<spaceInterpGpu> receiversXZGrid);

		// Acquisition
		void setAcquisition(
			std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid,
			std::shared_ptr<spaceInterpGpu> receiversCenterGrid, std::shared_ptr<spaceInterpGpu> receiversXGrid, std::shared_ptr<spaceInterpGpu> receiversZGrid, std::shared_ptr<spaceInterpGpu> receiversXZGrid,
			const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Nonlinear
		//void setAcquisition(std::shared_ptr<spaceInterpGpu> sources, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<spaceInterpGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Born + Tomo + Wemva

		// Scaling
		//void scaleSeismicSource(const std::shared_ptr<spaceInterpGpu> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParamElastic> parObj) const;

		// Other mutators
		void setGpuNumber(int iGpu, int iGpuId){_iGpu = iGpu; _iGpuId = iGpuId;}
		std::shared_ptr<float4DReg> setWavefield(int wavefieldFlag); // Allocates and returns a wavefield if flag = 1
		virtual void setAllWavefields(int wavefieldFlag) = 0; // Allocates all wavefields associated with a seismic operator --> this function has to be implemented by child classes

		// Accessors
		std::shared_ptr<fdParamElastic> getFdParam(){ return _fdParamElastic; }

};

#include "seismicElasticOperator2D.cpp"

#endif
