#include <double1DReg.h>
#include <double2DReg.h>
#include <iostream>
#include "spaceInterpGpu.h"
#include <vector>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/distributions/normal.hpp>
#include <math.h>

// Constructor #1
spaceInterpGpu::spaceInterpGpu(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<SEP::hypercube> elasticParamHypercube, int &nt, std::string interpMethod, int nFilt, int dipole, double zDipoleShift, double xDipoleShift) {

	_interpMethod = interpMethod;
  	_elasticParamHypercube = elasticParamHypercube;
	_oz = _elasticParamHypercube->getAxis(1).o;
	_ox = _elasticParamHypercube->getAxis(2).o;
	_dz = _elasticParamHypercube->getAxis(1).d;
	_dx = _elasticParamHypercube->getAxis(2).d;
	_zCoord = zCoord;
	_xCoord = xCoord;

	if(!_interpMethod.compare("linear")){ // if linear force nfilt to 1
		_nFilt = 1;
	}
	else {
		_nFilt = nFilt;
	}
	_nFilt2D=2*_nFilt;
	_nFiltTotal=_nFilt2D*_nFilt2D;
	checkOutOfBounds(_zCoord, _xCoord); // Make sure no device is on the edge of the domain
	_nDeviceIrreg = _zCoord->getHyper()->getAxis(1).n; // Nb of devices on irregular grid
	_nt = nt;
	_nz = _elasticParamHypercube->getAxis(1).n;

	// Treat the specific case wher dipole=1 AND interpolation method is linear
	_dipole = dipole;
	_zDipoleShift = zDipoleShift * _dz;
	_xDipoleShift = xDipoleShift * _dx;

	// If dipole is required, set _nFiltTotal to 8
	// Otherwise, set it to 4
	// Dipole is only implemented for linear interpolation (so far)
	if (_dipole == 1){_nFiltTotal=8;}
	else {_nFiltTotal=4;}

	_gridPointIndex = new int[_nFiltTotal*_nDeviceIrreg]; // Index of all the neighboring points of each device (non-unique) on the regular "1D" grid
	_weight = new double[_nFiltTotal*_nDeviceIrreg]; // Weights for spatial interpolation

	// calcualte  weights
	// linear
	if(!_interpMethod.compare("linear")){
		calcLinearWeights();
	}
	// sinc
	else if(!_interpMethod.compare("sinc")){
		// If user asks for dipole + sinc interpolation, throw an error (not implemented ey)
		if (_dipole == 1) {
			std::cerr << "**** ERROR: Spatial sinc interpolation and dipole acquisition not implemented ****" << std::endl;
			throw std::runtime_error("");;
		}
		calcSincWeights();
	}
	// gaussian
	else if(!_interpMethod.compare("gauss")){
		if (_dipole == 1) {
			std::cerr << "**** ERROR: Gaussian spatial interpolation and dipole acquisition not implemented ****" << std::endl;
			throw std::runtime_error("");;
		}
		calcGaussWeights();
	}
	else{
		std::cerr << "**** ERROR: Space interp method not defined ****" << std::endl;
		throw std::runtime_error("");;
	}

	convertIrregToReg();
}
void spaceInterpGpu::calcLinearWeights(){
	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		// Find the 4 neighboring points for all devices and compute the weights for the spatial interpolation
		int i1 = iDevice * _nFiltTotal;
		double wz = ( (*_zCoord->_mat)[iDevice] - _elasticParamHypercube->getAxis(1).o ) / _elasticParamHypercube->getAxis(1).d;
		double wx = ( (*_xCoord->_mat)[iDevice] - _elasticParamHypercube->getAxis(2).o ) / _elasticParamHypercube->getAxis(2).d;
		int zReg = wz; // z-coordinate on regular grid
		wz = wz - zReg;
		wz = 1.0 - wz;
		int xReg = wx; // x-coordinate on regular grid
		wx = wx - xReg;
		wx = 1.0 - wx;

		// Top left
		_gridPointIndex[i1] = xReg * _nz + zReg; // Index of this point for a 1D array representation
		_weight[i1] = wz * wx;

		// Bottom left
		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
		_weight[i1+1] = (1.0 - wz) * wx;

		// Top right
		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
		_weight[i1+2] = wz * (1.0 - wx);

		// Bottom right
		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
		_weight[i1+3] = (1.0 - wz) * (1.0 - wx);

		if (_dipole == 1){

			// Find the 4 neighboring points for all devices dipole points and compute the weights for the spatial interpolation
			double wzDipole = ( (*_zCoord->_mat)[iDevice] + _zDipoleShift - _elasticParamHypercube->getAxis(1).o ) / _elasticParamHypercube->getAxis(1).d;
			double wxDipole = ( (*_xCoord->_mat)[iDevice] + _xDipoleShift - _elasticParamHypercube->getAxis(2).o ) / _elasticParamHypercube->getAxis(2).d;
			int zRegDipole = wzDipole; // z-coordinate on regular grid
			wzDipole = wzDipole - zRegDipole;
			wzDipole = 1.0 - wzDipole;
			int xRegDipole = wxDipole; // x-coordinate on regular grid
			wxDipole = wxDipole - xRegDipole;
			wxDipole = 1.0 - wxDipole;

			// Top left (dipole point)
			_gridPointIndex[i1+4] = xRegDipole * _nz + zRegDipole; // Index of this point for a 1D array representation
			_weight[i1+4] = (-1.0) * wzDipole * wxDipole;

			// Bottom left (dipole point)
			_gridPointIndex[i1+5] = _gridPointIndex[i1+4] + 1;
			_weight[i1+5] = (-1.0) * (1.0 - wzDipole) * wxDipole;

			// Top right (dipole point)
			_gridPointIndex[i1+6] = _gridPointIndex[i1+4] + _nz;
			_weight[i1+6] = (-1.0) * wzDipole * (1.0 - wxDipole);

			// Bottom right (dipole point)
			_gridPointIndex[i1+7] = _gridPointIndex[i1+4] + _nz + 1;
			_weight[i1+7] = (-1.0) * (1.0 - wzDipole) * (1.0 - wxDipole);

		}
	}
}
void spaceInterpGpu::calcSincWeights(){
	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {
		int gridPointIndexOffset = iDevice*_nFiltTotal;

		//find double index (x,y) value of current device
		double z_irreg_double = (*_zCoord->_mat)[iDevice];
		double x_irreg_double = (*_xCoord->_mat)[iDevice];
		double z_reg_double = ( z_irreg_double - _oz ) / _dz;
		double x_reg_double = ( x_irreg_double - _ox ) / _dx;
		//find int index (x,y) immediatly above and to the left of the irregular device (x,z) value
		int z_reg_int = z_reg_double;
		int x_reg_int = x_reg_double;

		boost::math::normal normal_dist_x(x_irreg_double,2*_dx);
		boost::math::normal normal_dist_z(z_irreg_double,2*_dz);

		double weight_sum=0;
		// loop over filter points surrounding device irregular (x,z) value
		for(int ix=0; ix<_nFilt2D; ix++){
			for(int iz=0; iz<_nFilt2D; iz++){
				double x_irreg_cur = (x_reg_int+ix-_nFilt+1)*_dx+_ox;
				double z_irreg_cur = (z_reg_int+iz-_nFilt+1)*_dz+_oz;

				double x_input = (x_irreg_double-x_irreg_cur)/_dx;
				double z_input = (z_irreg_double-z_irreg_cur)/_dz;

				_weight[gridPointIndexOffset + _nFilt2D*ix + iz] = boost::math::sinc_pi(M_PI*z_input)*boost::math::sinc_pi(M_PI*x_input);
				_gridPointIndex[gridPointIndexOffset + _nFilt2D*ix + iz] = _nz * (x_reg_int+ix-_nFilt+1) + (z_reg_int+iz-_nFilt+1);
				weight_sum+=_weight[gridPointIndexOffset + _nFilt2D*ix + iz];
			}
		}
		//normalize gaussian distribution
		for(int i=0; i<_nFiltTotal;i++){
			_weight[gridPointIndexOffset + i] = _weight[gridPointIndexOffset + i]/weight_sum;
		}
	}
}
void spaceInterpGpu::calcGaussWeights(){
	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {
		int gridPointIndexOffset = iDevice*_nFiltTotal;

		//find double index (x,y) value of current device
		double z_irreg_double = (*_zCoord->_mat)[iDevice];
		double x_irreg_double = (*_xCoord->_mat)[iDevice];
		double z_reg_double = ( z_irreg_double - _oz ) / _dz;
		double x_reg_double = ( x_irreg_double - _ox ) / _dx;
		//find int index (x,y) immediatly above and to the left of the irregular device (x,z) value
		int z_reg_int = z_reg_double;
		int x_reg_int = x_reg_double;

		boost::math::normal normal_dist_x(x_irreg_double,2*_dx);
		boost::math::normal normal_dist_z(z_irreg_double,2*_dz);

		double weight_sum=0;
		// loop over filter points surrounding device irregular (x,z) value
		for(int ix=0; ix<_nFilt2D; ix++){
			for(int iz=0; iz<_nFilt2D; iz++){
				double x_irreg_cur = (x_reg_int+ix-_nFilt+1)*_dx+_ox;
				double z_irreg_cur = (z_reg_int+iz-_nFilt+1)*_dz+_oz;

				double x_input = (x_irreg_double-x_irreg_cur)/_dx;
				double z_input = (z_irreg_double-z_irreg_cur)/_dz;

				_weight[gridPointIndexOffset + _nFilt2D*ix + iz] = boost::math::pdf(normal_dist_x, x_irreg_cur)*boost::math::pdf(normal_dist_z, z_irreg_cur);
				_gridPointIndex[gridPointIndexOffset + _nFilt2D*ix + iz] = _nz * (x_reg_int+ix-_nFilt+1) + (z_reg_int+iz-_nFilt+1);
				weight_sum+=_weight[gridPointIndexOffset + _nFilt2D*ix + iz];
			}
		}
		//normalize gaussian distribution
		for(int i=0; i<_nFiltTotal;i++){
			_weight[gridPointIndexOffset + i] = _weight[gridPointIndexOffset + i]/weight_sum;
		}
	}
}
//
// // Constructor #2
// spaceInterpGpu::spaceInterpGpu(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<SEP::hypercube> elasticParamHypercube, int &nt) {
//
// 	_elasticParamHypercube = elasticParamHypercube;
// 	_nt = nt;
// 	_nDeviceIrreg = zGridVector.size(); // Nb of device
// 	int _nz = _elasticParamHypercube->getAxis(1).n;
// 	checkOutOfBounds(zGridVector, xGridVector); // Make sure no device is on the edge of the domain
// 	_gridPointIndex = new int[4*_nDeviceIrreg]; // All the neighboring points of each device (non-unique)
// 	_weight = new double[4*_nDeviceIrreg]; // Weights for spatial interpolation
//
// 	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {
//
// 		int i1 = iDevice * 4;
//
// 		// Top left
// 		_gridPointIndex[i1] = xGridVector[iDevice] * _nz + zGridVector[iDevice];
// 		_weight[i1] = 1.0;
//
// 		// Bottom left
// 		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
// 		_weight[i1+1] = 0.0;
//
// 		// Top right
// 		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
// 		_weight[i1+2] = 0.0;
//
// 		// Bottom right
// 		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
// 		_weight[i1+3] = 0.0;
// 	}
// 	convertIrregToReg();
// }
//
// // Constructor #3
// spaceInterpGpu::spaceInterpGpu(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<SEP::hypercube> elasticParamHypercube, int &nt){
//
// 	_elasticParamHypercube = elasticParamHypercube;
// 	_nt = nt;
// 	_nDeviceIrreg = nzDevice * nxDevice; // Nb of devices on irregular grid
// 	int _nz = _elasticParamHypercube->getAxis(1).n;
// 	checkOutOfBounds(nzDevice, ozDevice, dzDevice , nxDevice, oxDevice, dxDevice);
// 	_gridPointIndex = new int[4*_nDeviceIrreg]; // All the neighboring points of each device (non-unique)
// 	_weight = new double[4*_nDeviceIrreg]; // Weights for spatial interpolation
//
// 	int iDevice = -1;
// 	for (int ix = 0; ix < nxDevice; ix++) {
// 		int ixDevice = oxDevice + ix * dxDevice; // x-position of device on FD grid
// 		for (int iz = 0; iz < nzDevice; iz++) {
// 			int izDevice = ozDevice + iz * dzDevice; // z-position of device on FD grid
// 			iDevice++;
// 			int i1 = iDevice * 4;
//
// 			// Top left
// 			_gridPointIndex[i1] = ixDevice * _nz + izDevice;
// 			_weight[i1] = 1.0;
//
// 			// Bottom left
// 			_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
// 			_weight[i1+1] = 0.0;
//
// 			// Top right
// 			_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
// 			_weight[i1+2] = 0.0;
//
// 			// Bottom right
// 			_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
// 			_weight[i1+3] = 0.0;
//
// 		}
// 	}
// 	convertIrregToReg();
// }

void spaceInterpGpu::convertIrregToReg() {

	/* (1) Create map where:
		- Key = excited grid point index (points are unique)
		- Value = signal trace number
		(2) Create a vector containing the indices of the excited grid points
	*/

	_nDeviceReg = 0; // Initialize the number of regular devices to zero
	_gridPointIndexUnique.clear(); // Initialize to empty vector

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over gridPointIndex array
		for (int ifilt = 0; ifilt < _nFiltTotal; ifilt++){
			int i1 = iDevice * _nFiltTotal + ifilt;

			// If the grid point is not already in the list
			if (_indexMap.count(_gridPointIndex[i1]) == 0) {
				_nDeviceReg++; // Increment the number of (unique) grid points excited by the signal
				_indexMap[_gridPointIndex[i1]] = _nDeviceReg - 1; // Add the pair to the map
				_gridPointIndexUnique.push_back(_gridPointIndex[i1]); // Append vector containing all unique grid point index
			}
		}
	}
}

void spaceInterpGpu::forward(const bool add, const std::shared_ptr<double2DReg> signalReg, std::shared_ptr<double2DReg> signalIrreg) const {

	// /* FORWARD: Go from REGULAR grid -> IRREGULAR grid */
	// if (!add) signalIrreg->scale(0.0);
	//
	// std::shared_ptr<double2D> d = signalIrreg->_mat;
	// std::shared_ptr<double2D> m = signalReg->_mat;
	// for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
	// 	for (int iCorner = 0; iCorner < 4; iCorner++){ // Loop over neighboring points on regular grid
	// 		int i1 = iDevice * 4 + iCorner;
	// 		int i2 = _indexMap.find(_gridPointIndex[i1])->second;
	// 		for (int it = 0; it < _nt; it++){
	// 			(*d)[iDevice][it] += _weight[i1] * (*m)[i2][it];
	// 		}
	// 	}
	// }
	//assert(checkDomainRange(signalReg,signalIrreg));

	if (!add) signalIrreg->scale(0.0);
	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;
	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (int ifilt = 0; ifilt < _nFiltTotal; ifilt++){ // Loop over neighboring points on regular grid
			int i1 = iDevice * _nFiltTotal + ifilt;
			int i2 = _indexMap.find(_gridPointIndex[i1])->second;
			for (int it = 0; it < _nt; it++){
				(*d)[iDevice][it] += _weight[i1] * (*m)[i2][it];
			}
		}
	}
}

void spaceInterpGpu::adjoint(const bool add, std::shared_ptr<double2DReg> signalReg, const std::shared_ptr<double2DReg> signalIrreg) const {

	//assert(checkDomainRange(signalReg,signalIrreg));

	if (!add) signalReg->scale(0.0);
	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (int ifilt = 0; ifilt < _nFiltTotal; ifilt++){ // Loop over neighboring points on regular grid
			int i1 = iDevice * _nFiltTotal + ifilt;  // Grid point index
			int i2 = _indexMap.find(_gridPointIndex[i1])->second; // Get trace number for signalReg
			for (int it = 0; it < _nt; it++){
				(*m)[i2][it] += _weight[i1] * (*d)[iDevice][it];
			}
		}
	}
	// /* ADJOINT: Go from IRREGULAR grid -> REGULAR grid */
	// if (!add) signalReg->scale(0.0);
	// std::shared_ptr<double2D> d = signalIrreg->_mat;
	// std::shared_ptr<double2D> m = signalReg->_mat;
	//
	// for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
	// 	for (int iCorner = 0; iCorner < 4; iCorner++){ // Loop over neighboring points on regular grid
	// 		int i1 = iDevice * 4 + iCorner; // Grid point index
	// 		int i2 = _indexMap.find(_gridPointIndex[i1])->second; // Get trace number for signalReg
	// 		for (int it = 0; it < _nt; it++){
	// 			(*m)[i2][it] += _weight[i1] * (*d)[iDevice][it];
	// 		}
	// 	}
	// }
}

void spaceInterpGpu::checkOutOfBounds(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord){

	int nDevice = zCoord->getHyper()->getAxis(1).n;
	double zMin = _elasticParamHypercube->getAxis(1).o;
	double xMin = _elasticParamHypercube->getAxis(2).o;
	double zMax = zMin + (_elasticParamHypercube->getAxis(1).n - 1) * _elasticParamHypercube->getAxis(1).d;
	double xMax = xMin + (_elasticParamHypercube->getAxis(2).n - 1) * _elasticParamHypercube->getAxis(2).d;
	double zBuffer = _elasticParamHypercube->getAxis(1).d * _nFilt;
	double xBuffer = _elasticParamHypercube->getAxis(2).d * _nFilt;
	for (int iDevice = 0; iDevice < nDevice; iDevice++){
		if ( ((*zCoord->_mat)[iDevice] >= zMax-zBuffer) || ((*zCoord->_mat)[iDevice] <= zMin+zBuffer) ){
			std::cerr << "**** ERROR: One of the device is out of bounds in the z direction ****" << std::endl;
			std::cerr << "((*zCoord->_mat)[iDevice]= " << (*zCoord->_mat)[iDevice] << std::endl;
			std::cerr << "zMax-zBuffer= " << zMax-zBuffer << std::endl;
			std::cerr << "zMin+zBuffer= " << zMin+zBuffer << std::endl;
			throw std::runtime_error("");
		}
		if( ((*xCoord->_mat)[iDevice] >= xMax-xBuffer) || ((*xCoord->_mat)[iDevice] <= xMin+xBuffer)){
			std::cerr << "**** ERROR: One of the device is out of bounds in the x direction ****" << std::endl;
			std::cerr << "((*xCoord->_mat)[iDevice]= " << (*xCoord->_mat)[iDevice] << std::endl;
			std::cerr << "xMax-xBuffer= " << xMax-xBuffer << std::endl;
			std::cerr << "xMin+xBuffer= " << xMin+xBuffer << std::endl;
			throw std::runtime_error("");
		}
	}
}

void spaceInterpGpu::checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector){

	double zIntMax = *max_element(zGridVector.begin(), zGridVector.end());
	double xIntMax = *max_element(xGridVector.begin(), xGridVector.end());
	if ( (zIntMax >= _elasticParamHypercube->getAxis(1).n) || (xIntMax >= _elasticParamHypercube->getAxis(2).n) ){
		std::cout << "**** ERROR: One of the device is out of bounds ****" << std::endl;
		throw std::runtime_error("");
	}
}

void spaceInterpGpu::checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice){

	double zIntMax = ozDevice + (nzDevice - 1) * dzDevice;
	double xIntMax = oxDevice + (nxDevice - 1) * dxDevice;
	if ( (zIntMax >= _elasticParamHypercube->getAxis(1).n) || (xIntMax >= _elasticParamHypercube->getAxis(2).n) ){
		std::cout << "**** ERROR: One of the device is out of bounds ****" << std::endl;
		throw std::runtime_error("");
	}
}

void spaceInterpGpu::getInfo(){

	std::cerr << "---------- Space Interp Info ----------\n";
	std::cerr << "_nDeviceIrreg= " << _nDeviceIrreg << "\n";
	std::cerr << "_nDeviceRreg= " << _nDeviceReg << "\n";
	for(int i = 0; i < _nDeviceIrreg; i++){
		std::cerr << "Irreg Device " << i << "\n";
		std::cerr << "\t top left weight " << _weight[i*4] << "\n";
		std::cerr << "\t bottom left weight " << _weight[i*4+1] << "\n";
		std::cerr << "\t top right weight " << _weight[i*4+2] << "\n";
		std::cerr << "\t bottom right weight " << _weight[i*4+3] << "\n";
	}
}
