#ifndef STAGGERWFLD_H
#define STAGGERWFLD_H 1

#include <double4DReg.h>
#include <double3DReg.h>
//#include <staggerWfld.h>
#include "ioModes.h"
#include "fdParamElasticWaveEquation.h"
#include <operator.h>
#include "waveEquationElasticGpuFunctions.h"

using namespace SEP;
//! Apply the elastic wave equation to a wavefield
/*!

*/
class waveEquationElasticGpu : public Operator<SEP::double4DReg, SEP::double4DReg> {

  	private:
  		std::shared_ptr<fdParamElasticWaveEquation> _fdParamElastic;
      int _info;
      int _nGpu,_iGpuAlloc;
      std::vector<int> _gpuList;
      std::vector<int> _firstTimeSamplePerGpu;
      std::vector<int> _lastTimeSamplePerGpu;
      std::shared_ptr<paramObj> _par;

      //std::shared_ptr<staggerWfld> _staggerWfldOp;

  	public:
      //! Constructor.
  		/*!
      * Overloaded constructors from operator
      */
  		waveEquationElasticGpu(const std::shared_ptr<double4DReg> model, const std::shared_ptr<double4DReg> data, std::shared_ptr<SEP::double3DReg> elasticParam, std::shared_ptr<paramObj> par);

      //! FWD
  		/*!
      *
      */
      void forward(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double4DReg> data) const;

      //! ADJ
      /*!
      *
      */
  		void adjoint(const bool add, std::shared_ptr<double4DReg> model, const std::shared_ptr<double4DReg> data) const;


  		//! Desctructor
      /*!
      * A more elaborate description of Desctructor
      */

      bool checkGpuMemLimits(float byteLimits=15);

      void createGpuIdList();

      void createGpuSamplesList();

  		~waveEquationElasticGpu(){};
};

#endif
