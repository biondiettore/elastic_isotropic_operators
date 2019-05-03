#include <string>
#include <double2DReg.h>
#include "fdParamElasticWaveEquation.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <stagger.h>
#include <cstring>
using namespace SEP;

fdParamElasticWaveEquation::fdParamElasticWaveEquation(const std::shared_ptr<double3DReg> elasticParam, const std::shared_ptr<paramObj> par) {

	_elasticParam = elasticParam; //[rho; lambda; mu] [0; 1; 2]
	_par = par;

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus",0);
	_zPadMinus = _par->getInt("zPadMinus",0);
	_zPad = std::min(_zPadMinus, _zPadPlus);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _elasticParam->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus",0);
	_xPadMinus = _par->getInt("xPadMinus",0);
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _elasticParam->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

	/***** Wavefield component axis *****/
	_wavefieldCompAxis = axis(_nwc, 0, 1);

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat");
	_minPad = std::min(_zPad, _xPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = par->getFloat("alphaCos", 0.99);
	_errorTolerance = par->getFloat("errorTolerance", 0.000001);

	_minVpVs = 10000;
	_maxVpVs = -1;
	//#pragma omp for collapse(2)
	for (int ix = _fat; ix < _nx-2*_fat; ix++){
		for (int iz = _fat; iz < _nz-2*_fat; iz++){
			double rhoTemp = (*_elasticParam->_mat)[0][ix][iz];
			double lamdbTemp = (*_elasticParam->_mat)[1][ix][iz];
			double muTemp = (*_elasticParam->_mat)[2][ix][iz];

			double vpTemp = sqrt((lamdbTemp + 2*muTemp)/rhoTemp);
			double vsTemp = sqrt(muTemp/rhoTemp);

			// if(vpTemp<=0){
			// std::cerr << "**** ERROR: a vp value <=0 exists within the fat boundary ****" << std::endl;
			// assert(false);
			// }

			if (vpTemp < _minVpVs) _minVpVs = vpTemp;
			if (vpTemp > _maxVpVs) _maxVpVs = vpTemp;
			if (vsTemp < _minVpVs && vsTemp!=0) _minVpVs = vsTemp;
			if (vsTemp > _maxVpVs) _maxVpVs = vsTemp;
		}
	}

	/***** QC *****/
	assert(checkParfileConsistencySpace(_elasticParam)); // Parfile - velocity file consistency
	// assert(checkFdStability());
	// assert(checkFdDispersion());
	assert(checkModelSize());

	/***** Scaling for propagation *****/
	_rhox = new double[_nz * _nx * sizeof(double)]; // Precomputed scaling dtw / rho_x
	_rhoz = new double[_nz * _nx * sizeof(double)]; // Precomputed scaling dtw / rho_z
	_lamb2Mu = new double[_nz * _nx * sizeof(double)]; // Precomputed scaling (lambda + 2*mu) * dtw
	_lamb = new double[_nz * _nx * sizeof(double)]; // Precomputed scaling lambda * dtw
	_muxz = new double[_nz * _nx * sizeof(double)]; // Precomputed scaling mu_xz * dtw

	//initialize 2d slices
	_rhoxReg = std::make_shared<double2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_rhozReg = std::make_shared<double2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	std::shared_ptr<double2DReg> _temp(new double2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));
	_lambReg = std::make_shared<double2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_lamb2MuReg = std::make_shared<double2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));
	_muxzReg = std::make_shared<double2DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2));

	//slice _elasticParam into 2d
	std::memcpy( _temp->getVals(), _elasticParam->getVals(), _nx*_nz*sizeof(double) );
	std::memcpy( _muxzReg->getVals(), _elasticParam->getVals()+2*_nx*_nz, _nx*_nz*sizeof(double) );
	//stagger 2d density, mu
	std::shared_ptr<staggerX> _staggerX(new staggerX(_temp,_rhoxReg));
	std::shared_ptr<staggerZ> _staggerZ(new staggerZ(_temp,_rhozReg));

	//_temp holds density. _rhoxDtwReg, _rhozDtwReg are empty.
	_staggerX->adjoint(0, _rhoxReg, _temp);
	_staggerZ->adjoint(0, _rhozReg, _temp);
	//_muxzDtwReg holds mu. _temp holds density still, but will be zeroed.
	_staggerX->adjoint(0, _temp, _muxzReg); //temp now holds x staggered _mu
	_staggerZ->adjoint(0, _muxzReg, _temp); //_muxzDtwReg now holds x and z staggered mu

	//scaling factor for shifted
	#pragma omp for collapse(2)
	for (int ix = 0; ix < _nx; ix++){
		for (int iz = 0; iz < _nz; iz++) {
			(*_rhoxReg->_mat)[ix][iz] = (*_rhoxReg->_mat)[ix][iz]/(2*_dts);
			(*_rhozReg->_mat)[ix][iz] = (*_rhozReg->_mat)[ix][iz]/(2*_dts);
			// (*_lambReg->_mat)[ix][iz] = 2.0*_dtw * (*_elasticParam->_mat)[1][ix][iz];
			(*_lambReg->_mat)[ix][iz] = (*_elasticParam->_mat)[1][ix][iz];
			(*_lamb2MuReg->_mat)[ix][iz] = ((*_elasticParam->_mat)[1][ix][iz] + 2 * (*_elasticParam->_mat)[2][ix][iz]);
			// (*_muxzReg->_mat)[ix][iz] = 2.0*_dtw * (*_muxzReg->_mat)[ix][iz];
		}
	}

	// //get pointer to double array holding values. This is later passed to the device.
	_rhox = _rhoxReg->getVals();
	_rhoz = _rhozReg->getVals();
	_lamb = _lambReg->getVals();
	_lamb2Mu = _lamb2MuReg->getVals();
	_muxz = _muxzReg->getVals();

}

void fdParamElasticWaveEquation::getInfo(){

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
		std::cerr << " " << std::endl;


		// Vertical spatial sampling
		std::cerr << "-------------------- Vertical spatial sampling --------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nz = " << _nz-2*_fat-_zPadMinus-_zPadPlus << " [samples], dz = " << _dz << "[km], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [km]" << std::endl;
		std::cerr << "Model depth = " << _oz+(_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [km]" << std::endl;
		std::cerr << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Horizontal spatial sampling
		std::cerr << "-------------------- Horizontal spatial sampling ------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nx = " << _nx << " [samples], dx = " << _dx << " [km], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [km]" << std::endl;
		std::cerr << "Model width = " << _ox+(_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [km]" << std::endl;
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
		// 	std::cerr << "nOffset = " << _hExt << " [samples], dOffset= " << _dExt << " [km], oOffset = " << _oExt << " [km]" << std::endl;
		// 	std::cerr << "Total extension length nOffset = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [km]" << std::endl;
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
		// std::cerr << "Courant number = " << _Courant << " [-]" << std::endl;
		// std::cerr << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cerr << "Minimum velocity value (of either vp or vs) = " << _minVpVs << " [km/s]" << std::endl;
		std::cerr << "Maximum velocity value (of either vp or vs) = " << _maxVpVs << " [km/s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Maximum frequency without dispersion = " << _minVpVs/(3.0*std::max(_dz, _dx)) << " [Hz]" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << std::scientific; // Reset to scientific formatting notation
		std::cerr << std::setprecision(6); // Reset the default formatting precision
}

// bool fdParamElasticWaveEquation::checkFdStability(double CourantMax){
// 	_minDzDx = std::min(_dz, _dx);
// 	_Courant = 2.0 * _maxVpVs * _dtw / _minDzDx; //The Courant number is computed on the actual time-derivative sampling 2*dtw
// 	// if (_Courant > CourantMax){
// 	// 	std::cerr << "**** ERROR: Courant is too big: " << _Courant << " ****" << std::endl;
// 	// 	std::cerr << "Max velocity value: " << _maxVpVs << " [m/s]" << std::endl;
// 	// 	std::cerr << "Dtw: " << _dtw << " [s]" << std::endl;
// 	// 	std::cerr << "Min (dz, dx): " << _minDzDx << " [m]" << std::endl;
// 	// 	return false;
// 	// }
// 	return true;
// }
//
// bool fdParamElasticWaveEquation::checkFdDispersion(double dispersionRatioMin){
// 	_maxDzDx = std::max(_dz, _dx);
// 	_dispersionRatio = _minVpVs / (_fMax*_maxDzDx);
//
// 	// if (_dispersionRatio < dispersionRatioMin){
// 	// 	std::cerr << "**** ERROR: Dispersion is too small: " << _dispersionRatio <<  " > " << dispersionRatioMin << " ****" << std::endl;
// 	// 	std::cerr << "Min velocity value = " << _minVpVs << " [m/s]" << std::endl;
// 	// 	std::cerr << "Max (dz, dx) = " << _maxDzDx << " [m]" << std::endl;
// 	// 	std::cerr << "Max frequency = " << _fMax << " [Hz]" << std::endl;
// 	// 	return false;
// 	// }
// 	return true;
// }

bool fdParamElasticWaveEquation::checkModelSize(){
	if ( (_nz-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nz not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nx not a multiple of block size ****" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElasticWaveEquation::checkParfileConsistencyTime(const std::shared_ptr<double3DReg> seismicTraces, int timeAxisIndex,  std::string fileToCheck) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cerr << "**** ERROR: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cerr << "**** ERROR: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cerr << "**** ERROR: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

bool fdParamElasticWaveEquation::checkParfileConsistencySpace(const std::shared_ptr<double3DReg> model) const {

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cerr << "**** ERROR: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cerr << "**** ERROR: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cerr << "**** ERROR: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cerr << "**** ERROR nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cerr << "**** ERROR: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cerr << "**** ERROR: ox not consistent with parfile ****" << std::endl; return false;}

  // Elastic Parameter axis
	if (3 != model->getHyper()->getAxis(3).n) {std::cerr << "**** ERROR number of elastic parameters != 3 ****" << std::endl; return false;}

	return true;
}

// bool fdParamElastic::checkParfileConsistencySpace(const std::shared_ptr<double3DReg> modelExt) const {

// 	// Vertical axis
// 	if (_nz != modelExt->getHyper()->getAxis(1).n) {std::cerr << "**** ERROR: nz not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_dz - modelExt->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cerr << "**** ERROR: dz not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_oz - modelExt->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cerr << "**** ERROR: oz not consistent with parfile ****" << std::endl; return false;}

// 	// Vertical axis
// 	if (_nx != modelExt->getHyper()->getAxis(2).n) {std::cerr << "**** ERROR: nx not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_dx - modelExt->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cerr << "**** ERROR: dx not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_ox - modelExt->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cerr << "**** ERROR: ox not consistent with parfile ****" << std::endl; return false;}

// 	// Extended axis
// 	if (_nExt != modelExt->getHyper()->getAxis(3).n) {std::cerr << "**** ERROR: nExt not consistent with parfile ****" << std::endl; return false;}
// 	if (_nExt>1) {
// 		if ( std::abs(_dExt - modelExt->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cerr << "**** ERROR: dExt not consistent with parfile ****" << std::endl; return false;}
// 		if ( std::abs(_oExt - modelExt->getHyper()->getAxis(3).o) > _errorTolerance ) { std::cerr << "**** ERROR: oExt not consistent with parfile ****" << std::endl; return false;}
// 	}

// 	return true;
// }

fdParamElasticWaveEquation::~fdParamElasticWaveEquation(){
	_rhox = NULL;
  _rhoz = NULL;
  _lamb2Mu = NULL;
  _lamb = NULL;
  _muxz = NULL;
}
