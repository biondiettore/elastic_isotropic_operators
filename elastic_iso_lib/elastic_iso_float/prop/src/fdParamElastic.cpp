#include <string>
#include <float2DReg.h>
#include "fdParamElastic.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <stagger.h>
#include <cstring>
#include "varDeclare.h"
using namespace SEP;

fdParamElastic::fdParamElastic(const std::shared_ptr<float3DReg> elasticParam, const std::shared_ptr<paramObj> par) {

	_elasticParam = elasticParam; //[rho; lambda; mu] [0; 1; 2]
	_par = par;

	/***** Coarse time-sampling *****/
	_surfaceCondition = _par->getInt("surfaceCondition",0);

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_sub = _par->getInt("sub");
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Fine time-sampling *****/
	_sub = _sub*2;
	if(_sub > SUB_MAX) {
		std::cerr << "**** ERROR: 2*_sub is greater than the allowed SUB_MAX value. " << _sub << " > " << SUB_MAX << " ****" << std::endl;
		throw std::runtime_error("");
	}
	_ntw = (_nts - 1) * _sub + 1;
	_dtw = _dts / float(_sub);
	 //since the time stepping is a central difference first order derivative we need to take twice as many time steps as the given sub variable.
	_otw = _ots;
	_timeAxisFine = axis(_ntw, _otw, _dtw);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus");
	_zPadMinus = _par->getInt("zPadMinus");
	if(_surfaceCondition==0) _zPad = std::min(_zPadMinus, _zPadPlus);
	else if(_surfaceCondition==1) _zPad = _zPadPlus;
	_mod_par = _par->getInt("mod_par",1);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _elasticParam->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _elasticParam->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

	if (_mod_par == 2){
		// Scaling spatial samplings if km were provided
		_dz *= 1000.;
		_ox *= 1000.;
		_dx *= 1000.;
		_oz *= 1000.;
	}

	/***** Wavefield component axis *****/
	_wavefieldCompAxis = axis(_nwc, 0, 1);

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat",4);
	_minPad = std::min(_zPad, _xPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = _par->getFloat("alphaCos", 0.99);
	_errorTolerance = _par->getFloat("errorTolerance", 0.000001);

	/***** Other parameters *****/

	_minVpVs = 10000;
	_maxVpVs = -1;
	//#pragma omp for collapse(2)
	for (int ix = _fat; ix < _nx-2*_fat; ix++){
		for (int iz = _fat; iz < _nz-2*_fat; iz++){
			float rhoTemp = (*_elasticParam->_mat)[0][ix][iz];
			float lamdbTemp = (*_elasticParam->_mat)[1][ix][iz];
			float muTemp = (*_elasticParam->_mat)[2][ix][iz];

			float vpTemp = sqrt((lamdbTemp + 2*muTemp)/rhoTemp);
			float vsTemp = sqrt(muTemp/rhoTemp);

			if (vpTemp < _minVpVs) _minVpVs = vpTemp;
			if (vpTemp > _maxVpVs) _maxVpVs = vpTemp;
			if (vsTemp < _minVpVs && vsTemp!=0) _minVpVs = vsTemp;
			if (vsTemp > _maxVpVs) _maxVpVs = vsTemp;
		}
	}

	/***** QC *****/
	if( not checkParfileConsistencySpace(_elasticParam)){
		throw std::runtime_error("");
	}; // Parfile - velocity file consistency
	if( not checkFdStability()){
		throw std::runtime_error("");
	};
	if ( not checkFdDispersion()){
		throw std::runtime_error("");
	};
	if( not checkModelSize()){
		throw std::runtime_error("");
	};

	/***** Scaling for propagation *****/

	//initialize 2d slices
	_rhoxDtwReg = std::make_shared<float2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_rhozDtwReg = std::make_shared<float2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	std::shared_ptr<float2DReg> _temp(new float2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));
	_lambDtwReg = std::make_shared<float2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_lamb2MuDtwReg = std::make_shared<float2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_muxzDtwReg = std::make_shared<float2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));

	//slice _elasticParam into 2d
	std::memcpy( _temp->getVals(), _elasticParam->getVals(), _nx*_nz*sizeof(float) );
	std::memcpy( _muxzDtwReg->getVals(), _elasticParam->getVals()+2*_nx*_nz, _nx*_nz*sizeof(float) );
	//stagger 2d density, mu
	std::shared_ptr<staggerX> _staggerX(new staggerX(_temp,_rhoxDtwReg));
	std::shared_ptr<staggerZ> _staggerZ(new staggerZ(_temp,_rhozDtwReg));

	//_temp holds density. _rhoxDtwReg, _rhozDtwReg are empty.
	_staggerX->adjoint(0, _rhoxDtwReg, _temp);
	_staggerZ->adjoint(0, _rhozDtwReg, _temp);
	//_muxzDtwReg holds mu. _temp holds density still, but will be zeroed.
	_staggerX->adjoint(0, _temp, _muxzDtwReg); //temp now holds x staggered _mu
	_staggerZ->adjoint(0, _muxzDtwReg, _temp); //_muxzDtwReg now holds x and z staggered mu

	//scaling factor for shifted
	#pragma omp for collapse(2)
	for (int ix = 0; ix < _nx; ix++){
		for (int iz = 0; iz < _nz; iz++) {
			(*_rhoxDtwReg->_mat)[ix][iz] = 2.0*_dtw / (*_rhoxDtwReg->_mat)[ix][iz];
			(*_rhozDtwReg->_mat)[ix][iz] = 2.0*_dtw / (*_rhozDtwReg->_mat)[ix][iz];
			(*_lambDtwReg->_mat)[ix][iz] = 2.0*_dtw * (*_elasticParam->_mat)[1][ix][iz];
			(*_lamb2MuDtwReg->_mat)[ix][iz] = 2.0*_dtw * ((*_elasticParam->_mat)[1][ix][iz] + 2 * (*_elasticParam->_mat)[2][ix][iz]);
			(*_muxzDtwReg->_mat)[ix][iz] = 2.0*_dtw * (*_muxzDtwReg->_mat)[ix][iz];
		}
	}

	// //get pointer to float array holding values. This is later passed to the device.
	_rhoxDtw = _rhoxDtwReg->getVals();
	_rhozDtw = _rhozDtwReg->getVals();
	_lambDtw = _lambDtwReg->getVals();
	_lamb2MuDtw = _lamb2MuDtwReg->getVals();
	_muxzDtw = _muxzDtwReg->getVals();

}

void fdParamElastic::getInfo(){

		std::cerr << " " << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << "************************ FD PARAMETERS INFO ***********************" << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;

		// Coarse time sampling
		std::cerr << "------------------------ Coarse time sampling ---------------------" << std::endl;
		std::cerr << std::fixed;
		std::cerr << std::setprecision(3);
		std::cerr << "nts = " << _nts << " [samples], dts = " << _dts << " [s], ots = " << _ots << " [s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Nyquist frequency = " << 1.0/(2.0*_dts) << " [Hz]" << std::endl;
		std::cerr << "Maximum frequency from seismic source = " << _fMax << " [Hz]" << std::endl;
		std::cerr << std::setprecision(6);
		std::cerr << "Total recording time = " << (_nts-1) * _dts << " [s]" << std::endl;
		std::cerr << "Subsampling = " << _sub << std::endl;
		std::cerr << " " << std::endl;

		// Coarse time sampling
		std::cerr << "------------------------ Fine time sampling -----------------------" << std::endl;
		std::cerr << "ntw = " << _ntw << " [samples], dtw = " << _dtw << " [s], otw = " << _otw << " [s]" << std::endl;
		std::cerr << "derivative sampling (dtw*2) = " << _dtw*2 << std::endl;
		std::cerr << " " << std::endl;

		// Vertical spatial sampling
		std::cerr << "-------------------- Vertical spatial sampling --------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nz = " << _nz << " [samples], dz = " << _dz << "[m], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [m]" << std::endl;
		std::cerr << "Model depth = " << (_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [m]" << std::endl;
		std::cerr << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Horizontal spatial sampling
		std::cerr << "-------------------- Horizontal spatial sampling ------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nx = " << _nx << " [samples], dx = " << _dx << " [m], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [m]" << std::endl;
		std::cerr << "Model width = " << (_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [m]" << std::endl;
		std::cerr << "Left padding = " << _xPadMinus << " [samples], right padding = " << _xPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// // Extended axis
		// if ( (_nExt>1) && (_extension=="time")){
		// 	std::cerr << std::setprecision(3);
		// 	std::cerr << "-------------------- Extended axis: time-lags ---------------------" << std::endl;
		// 	std::cerr << "nTau = " << _hExt << " [samples], dTau= " << _dExt << " [s], oTau = " << _oExt << " [s]" << std::endl;
		// 	std::cerr << "Total extension length nTau = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [s]" << std::endl;
		// }
		//
		// if ( (_nExt>1) && (_extension=="offset") ){
		// 	std::cerr << std::setprecision(2);
		// 	std::cerr << "---------- Extended axis: horizontal subsurface offsets -----------" << std::endl;
		// 	std::cerr << "nOffset = " << _hExt << " [samples], dOffset= " << _dExt << " [m], oOffset = " << _oExt << " [m]" << std::endl;
		// 	std::cerr << "Total extension length nOffset = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [m]" << std::endl;
		// }

		// GPU FD parameters
		std::cerr << "---------------------- GPU kernels parameters ---------------------" << std::endl;
		std::cerr << "Block size in z-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Block size in x-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Halo size for staggered 8th-order derivative [FAT] = " << _fat << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Stability and dispersion
		std::cerr << "---------------------- Stability and dispersion -------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "Courant number = " << _Courant << " [-]" << std::endl;
		std::cerr << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cerr << "Minimum velocity value (of either vp or vs) = " << _minVpVs << " [m/s]" << std::endl;
		std::cerr << "Maximum velocity value (of either vp or vs) = " << _maxVpVs << " [m/s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Maximum frequency without dispersion = " << _minVpVs/(3.0*std::max(_dz, _dx)) << " [Hz]" << std::endl;
		std::cerr << " " << std::endl;


		// Free Surface Condition
		std::cerr << "----------------------- Surface Condition --------------------" << std::endl;
		std::cerr << "Chosen surface condition parameter: ";
		if(_surfaceCondition==0) std::cerr << "(0) no free surface condition" << '\n';
		else if(_surfaceCondition==1 ) std::cerr << "(1) free surface condition from Robertsson (1998) chosen." << '\n';
		else{
			std::cerr << "ERROR NO IMPROPER FREE SURFACE PARAM PROVIDED" << '\n';
			throw std::runtime_error("");;
		}

		std::cerr << "\n----------------------- Source Interp Info -----------------------" << std::endl;
		std::cerr << "Chosen source interpolation method: " << _par->getString("sourceInterpMethod","linear") << std::endl;
		std::cerr << "Chosen number of filter on one side of device: " << _par->getInt("sourceInterpNumFilters",1) << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << "\n*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << std::scientific; // Reset to scientific formatting notation
		std::cerr << std::setprecision(6); // Reset the default formatting precision
}

bool fdParamElastic::checkFdStability(float CourantMax){
	_minDzDx = std::min(_dz, _dx);
	_Courant = _maxVpVs * _dtw * 2.0 / _minDzDx;
	if (_Courant > CourantMax){
		std::cerr << "**** ERROR: Courant is too big: " << _Courant << " ****" << std::endl;
		std::cerr << "Max velocity value: " << _maxVpVs << " [m/s]" << std::endl;
		std::cerr << "Dtw: " << _dtw << " [s]" << std::endl;
		std::cerr << "Min (dz, dx): " << _minDzDx << " [m]" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElastic::checkFdDispersion(float dispersionRatioMin){
	_maxDzDx = std::max(_dz, _dx);
	_dispersionRatio = _minVpVs / (_fMax*_maxDzDx);

	if (_dispersionRatio < dispersionRatioMin){
		std::cerr << "**** ERROR: Dispersion is too small: " << _dispersionRatio <<  " > " << dispersionRatioMin << " ****" << std::endl;
		std::cerr << "Min velocity value = " << _minVpVs << " [m/s]" << std::endl;
		std::cerr << "Max (dz, dx) = " << _maxDzDx << " [m]" << std::endl;
		std::cerr << "Max frequency = " << _fMax << " [Hz]" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElastic::checkModelSize(){
	if ((_nz-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nz-2*_fat not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nx-2*_fat not a multiple of block size ****" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElastic::checkParfileConsistencyTime(const std::shared_ptr<float3DReg> seismicTraces, int timeAxisIndex) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cerr << "**** ERROR: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cerr << "**** ERROR: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cerr << "**** ERROR: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

bool fdParamElastic::checkParfileConsistencySpace(const std::shared_ptr<float3DReg> model) const {

	// Scaling factor in case km were provided within the parameter file
	float scale = 1.0;
	if (_mod_par == 2){
		scale = 1000.0;
	}

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cerr << "**** ERROR: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d*scale) > _errorTolerance ) {std::cerr << "**** ERROR: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o*scale) > _errorTolerance ) {std::cerr << "**** ERROR: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cerr << "**** ERROR nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d*scale) > _errorTolerance ) {std::cerr << "**** ERROR: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o*scale) > _errorTolerance ) {std::cerr << "**** ERROR: ox not consistent with parfile ****" << std::endl; return false;}

	// Elastic Parameter axis
	// if (3 != model->getHyper()->getAxis(3).n) {std::cerr << "**** ERROR number of elastic parameters != 3 ****" << std::endl; return false;}

	return true;
}

fdParamElastic::~fdParamElastic(){
  _rhoxDtw = NULL;
  _rhozDtw = NULL;
  _lamb2MuDtw = NULL;
  _lambDtw = NULL;
  _muxzDtw = NULL;
}
