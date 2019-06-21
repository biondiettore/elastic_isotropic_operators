#ifndef FD_PARAM_ELASTIC_H
#define FD_PARAM_ELASTIC_H 1

#include <string>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include <iostream>

using namespace SEP;
//! This class is meant to analyze the finite difference parameters of a given elastic prop.
/*!
 Class allows us to test dipersion, stability, model size, and to retrieve finite difference parameters.
*/
class fdParamElastic{

 	public:

		// Constructor
		fdParamElastic(const std::shared_ptr<float3DReg> elasticParam, const std::shared_ptr<paramObj> par); /** given a parameter file and a velocity model, ensure the dimensions match the prop will be stable and will avoid dispersion.*/
		// Destructor
		~fdParamElastic();

		// QC stuff
		bool checkParfileConsistencyTime(const std::shared_ptr<float3DReg> seismicTraces, int timeAxisIndex) const; /** ensure time axis of traces matches nts from parfile */
		bool checkParfileConsistencySpace(const std::shared_ptr<float3DReg> model) const; /** ensure space axes of model matches those from parfile */
		//bool checkParfileConsistencySpace(const std::shared_ptr<float3DReg> modelExt) const;

		bool checkFdStability(float courantMax=0.45); /** checks stability */
		bool checkFdDispersion(float dispersionRatioMin=3.0); /** checks dispersion */
		bool checkModelSize(); /** Make sure the domain size (without the FAT) is a multiple of the dimblock size */
		void getInfo(); /** prints finite difference variables to stdout */

		// Variables
		std::shared_ptr<paramObj> _par;
		std::shared_ptr<float3DReg> _elasticParam, _smallElasticParam; //[rho; lambda; mu] [0; 1; 2]

		axis _timeAxisCoarse, _timeAxisFine, _zAxis, _xAxis, _extAxis, _wavefieldCompAxis;

        // Precomputed scaling dtw / rho_x , dtw / rho_z , (lambda + 2*mu) * dtw , lambda * dtw , mu_xz * dtw
        std::shared_ptr<float2DReg> _rhoxDtwReg, _rhozDtwReg, _lamb2MuDtwReg, _lambDtwReg, _muxzDtwReg;
        //pointers to float arrays holding values. These are later passed to the device.
        float *_rhoxDtw,*_rhozDtw,*_lamb2MuDtw,*_lambDtw,*_muxzDtw;

		//float *_reflectivityScale;
		float _errorTolerance;
		float _minVpVs, _maxVpVs, _minDzDx, _maxDzDx;
		int _nts, _sub, _ntw;
		float _ots, _dts, _otw, _dtw;
		float _Courant, _dispersionRatio;
		int _nz, _nx;
        const int _nwc=5;
		int _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _zPad, _xPad, _minPad;
		float _dz, _dx, _oz, _ox, _fMax;
		int _saveWavefield, _blockSize, _fat;
		float _alphaCos;
        int _surfaceCondition;

};

#endif
