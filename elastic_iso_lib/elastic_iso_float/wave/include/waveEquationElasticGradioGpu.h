#ifndef WAVE_EQUATION_ELASTIC_H
#define WAVE_EQUATION_ELASTIC_H 1

#include <float4DReg.h>
#include <float3DReg.h>
//#include <staggerWfld.h>
#include "ioModes.h"
#include <operator.h>
#include "fdParamElasticWaveEquation.h"

using namespace SEP;
//! Apply the elastic wave equation to a wavefield
/*!

*/
class waveEquationElasticGradioGpu : public Operator<SEP::float4DReg, SEP::float4DReg> {

  	private:
  		std::shared_ptr<fdParamElasticWaveEquation> _fdParamElastic;
      int _info;
      int _nGpu,_iGpuAlloc;
      std::vector<int> _gpuList;
      std::vector<int> _firstTimeSamplePerGpu;
      std::vector<int> _lastTimeSamplePerGpu;
      std::shared_ptr<paramObj> _par;
      std::shared_ptr<SEP::float4DReg> _wfld;

      //std::shared_ptr<staggerWfld> _staggerWfldOp;

  	public:
      //! Constructor.
  		/*!
      * Overloaded constructors from operator
      */
  		waveEquationElasticGradioGpu(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float4DReg> data, std::shared_ptr<SEP::float4DReg> wfld, std::shared_ptr<SEP::float4DReg> forcingTerm);

      //! FWD
  		/*!
      *
      */
      void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float4DReg> data) const;

      //! ADJ
      /*!
      *
      */
  		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float4DReg> data) const;


      void updateWfld(std::shared_ptr<float4DReg> wfld)
  		//! Desctructor
      /*!
      * A more elaborate description of Desctructor
      */

      bool checkGpuMemLimits(float byteLimits=15);

      void createGpuIdList();

      void createGpuSamplesList();

  		~waveEquationElasticGradioGpu(){};
};

#endif
