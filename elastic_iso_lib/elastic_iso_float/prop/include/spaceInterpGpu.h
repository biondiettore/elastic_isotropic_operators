#ifndef DEVICE_ELASTIC_GPU_H
#define DEVICE_ELASTIC_GPU_H 1

#include "ioModes.h"
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "hypercube.h"
#include "operator.h"
#include <vector>

using namespace SEP;
//! This class transforms the data on an irregular space grid (positions of the receivers for example) into data on a regular grid.
/*!
 Once the data is transformed into data on a regular grid, you can pass it to the gpu function and there will be no race condition when you inject
*/
class spaceInterpGpu : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

		/* Spatial interpolation */
		//std::shared_ptr<float2DReg> _elasticPar; /** for dimension checking*/
		std::shared_ptr<SEP::hypercube> _elasticParamHypercube;
		std::shared_ptr<float1DReg> _zCoord, _xCoord; /** Detailed description after the member */
		std::vector<int> _gridPointIndexUnique; /** Array containing all the positions of the excited grid points - each grid point is unique */
		std::map<int, int> _indexMap;
		std::map<int, int>::iterator _iteratorIndexMap;
		float *_weight;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz, _nFilt,_nFilt2D,_nFiltTotal;
		int _dipole;
		float _zDipoleShift, _xDipoleShift;
		float _ox,_oz,_dx,_dz;
		std::string _interpMethod;

	public:

		/* Overloaded constructors */
		spaceInterpGpu(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<SEP::hypercube> elasticParamHyper, int &nt, std::string interpMethod, int nFilt, int dipole, float zDipoleShift, float xDipoleShift);
		// spaceInterpGpu(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<SEP::hypercube> elasticParamHyper, int &nt);
		// spaceInterpGpu(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<SEP::hypercube> elasticParamHyper, int &nt);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const;

		// Destructor
		~spaceInterpGpu(){};

		// Other functions
		void checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord); // For constructor #1
		void checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector); // For constructor #2
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice); // For constructor #3
		void convertIrregToReg();

		int *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
		int *getRegPos(){ return _gridPointIndex; }
		int getNt(){ return _nt; }
		int getNDeviceReg(){ return _nDeviceReg; }
		int getNDeviceIrreg(){ return _nDeviceIrreg; }
		float * getWeights() { return _weight; }
		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
		void getInfo();
		void calcLinearWeights();
		void calcSincWeights();
		void calcGaussWeights();
};

#endif
