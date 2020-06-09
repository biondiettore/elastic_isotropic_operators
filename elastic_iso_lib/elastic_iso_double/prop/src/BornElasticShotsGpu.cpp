#include <vector>
#include <omp.h>
#include "BornElasticShotsGpu.h"
#include "BornElasticGpu.h"
#include <stagger.h>
#include <ctime>

BornElasticShotsGpu::BornElasticShotsGpu(std::shared_ptr<SEP::double3DReg> elasticParam,
                                         std::shared_ptr<paramObj> par,
                                         std::vector<std::shared_ptr<SEP::double3DReg>> sourcesSignalsVector,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorCenterGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorZGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXZGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorCenterGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorZGrid,
                                         std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXZGrid){

    // Setup parameters
  	_par = par;
    _sourcesSignalsVector = sourcesSignalsVector;
  	_elasticParam = elasticParam;
  	_nExp = par->getInt("nExp");
    _useStreams = par->getInt("useStreams", 0);
		createGpuIdList();
  	_info = par->getInt("info", 0);
  	_deviceNumberInfo = par->getInt("deviceNumberInfo", 0);
  	assert(getGpuInfo(_gpuList, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs
  	_saveWavefield = _par->getInt("saveWavefield", 0);
  	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
  	if (_info == 1 && _saveWavefield == 1){std::cerr << "Saving wavefield(s) for shot # " << _wavefieldShotNumber << std::endl;}
  	_sourcesVectorCenterGrid = sourcesVectorCenterGrid;
    _sourcesVectorXGrid = sourcesVectorXGrid;
    _sourcesVectorZGrid = sourcesVectorZGrid;
    _sourcesVectorXZGrid = sourcesVectorXZGrid;

    _receiversVectorCenterGrid = receiversVectorCenterGrid;
    _receiversVectorXGrid = receiversVectorXGrid;
    _receiversVectorZGrid = receiversVectorZGrid;
    _receiversVectorXZGrid = receiversVectorXZGrid;

    _fdParamElastic = std::make_shared<fdParamElastic>(_elasticParam, _par);


  }

void BornElasticShotsGpu::createGpuIdList(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);
	std::vector<int> dummyVector;
	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR: Please provide a list of GPUs to be used ****" << std::endl; assert(1==2);}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		std::sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR: Please make sure there are no duplicates in the GPU Id list ****" << std::endl; assert(1==2);
		}
	}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}


void BornElasticShotsGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double4DReg> data) const{
    if (!add) data->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    // Check whether we use the same source signals for all shots
    if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom = 1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), SEP::axis(5))); //This model slice contains the staggered and scaled model perturbations (drhox, drhoz, dlame, dmu, dmuxz)
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<double3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<double3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<BornElasticGpu>> BornObjectVector;

    //Staggering and scaling input model perturbations
    std::shared_ptr<double2DReg> temp(new double2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));
    std::shared_ptr<double2DReg> temp1(new double2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));
    std::shared_ptr<SEP::double3DReg> modelSlice(new SEP::double3DReg(hyperModelSlices));

    //stagger 2d density, mu
    std::shared_ptr<staggerX> staggerXop(new staggerX(temp,temp1));
    std::shared_ptr<staggerZ> staggerZop(new staggerZ(temp,temp1));

    unsigned long long nz = _fdParamElastic->_nz;
    unsigned long long nx = _fdParamElastic->_nx;

    //DENSITY PERTURBATION
    //D_RHOX
    std::memcpy( temp->getVals(), model->getVals(), nx*nz*sizeof(double) );
    staggerXop->adjoint(false, temp1, temp);
    std::memcpy( modelSlice->getVals(), temp1->getVals(), nx*nz*sizeof(double) );
    //D_RHOZ
    staggerZop->adjoint(false, temp1, temp);
    std::memcpy( modelSlice->getVals()+nx*nz, temp1->getVals(), nx*nz*sizeof(double) );

    //D_LAME
    std::memcpy( modelSlice->getVals()+2*nx*nz, model->getVals()+nx*nz, nx*nz*sizeof(double) );

    //SHEAR MODULUS PERTURBATION
    //D_MU
    std::memcpy( modelSlice->getVals()+3*nx*nz, model->getVals()+2*nx*nz, nx*nz*sizeof(double) );
    //D_MUXZ
    std::memcpy( temp->getVals(), model->getVals()+2*nx*nz, nx*nz*sizeof(double) );
    staggerXop->adjoint(false, temp1, temp);
    staggerZop->adjoint(false, temp, temp1);
    std::memcpy( modelSlice->getVals()+4*nx*nz, temp->getVals(), nx*nz*sizeof(double) );

    //Scaling of the perturbations
    #pragma omp for collapse(2)
    for (long long ix = 0; ix < nx; ix++){
    	for (long long iz = 0; iz < nz; iz++) {
    		(*modelSlice->_mat)[0][ix][iz] *= (*_fdParamElastic->_rhoxDtwReg->_mat)[ix][iz];
    		(*modelSlice->_mat)[1][ix][iz] *= (*_fdParamElastic->_rhozDtwReg->_mat)[ix][iz];
    		(*modelSlice->_mat)[2][ix][iz] *= 2.0*_fdParamElastic->_dtw;
    		(*modelSlice->_mat)[3][ix][iz] *= 2.0*_fdParamElastic->_dtw;
    		(*modelSlice->_mat)[4][ix][iz] *= 2.0*_fdParamElastic->_dtw;
    	}
    }

    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Nonlinear propagator object
  		std::shared_ptr<BornElasticGpu> BornGpuObject(new BornElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		BornObjectVector.push_back(BornGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			BornGpuObject->getFdParam()->getInfo();
  		}

  		// Model slice
  		modelSlicesVector.push_back(modelSlice);

  		// Data slice
  		std::shared_ptr<SEP::double3DReg> dataSlices(new SEP::double3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);

  	}

    // Launch Born forward

    //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
    #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
    for (int iExp=0; iExp<_nExp; iExp++){

      int iGpu = omp_get_thread_num();
  	  int iGpuId = _gpuList[iGpu];

      // Set acquisition geometry
  	  if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
            modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }
  	  if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
            modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }
  	  if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
            modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }
  	  if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
            modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }

      // Set GPU number for propagator object
      BornObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

      //Test dot-product of single shot operator
      // BornObjectVector[iGpu]->dotTest(true);
      // exit(0);

      //Launch modeling
      BornObjectVector[iGpu]->forward(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

      // Store dataSlice into data
      #pragma omp parallel for collapse(3)
      for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
        for (int iReceiver=0; iReceiver<hyperDataSlices->getAxis(2).n; iReceiver++){
          for (int its=0; its<hyperDataSlices->getAxis(1).n; its++){
            (*data->_mat)[iExp][iwc][iReceiver][its] += (*dataSlicesVector[iGpu]->_mat)[iwc][iReceiver][its];
          }
        }
      }
    }

    // Deallocate memory on device
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
      deallocateBornElasticGpu(iGpu,_gpuList[iGpu],_useStreams);
    }

}



void BornElasticShotsGpu::adjoint(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double4DReg> data) const{
    if (!add) model->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    // Check whether we use the same source signals for all shots
    if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), SEP::axis(5))); //This model slice contains the staggered and scaled model perturbations (drhox, drhoz, dlame, dmu, dmuxz)
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<double3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<double3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<BornElasticGpu>> BornObjectVector;

    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

        // Nonlinear propagator object
  		std::shared_ptr<BornElasticGpu> BornGpuObject(new BornElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		BornObjectVector.push_back(BornGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			BornGpuObject->getFdParam()->getInfo();
  		}

  		// Model slice
  		std::shared_ptr<SEP::double3DReg> modelSlices(new SEP::double3DReg(hyperModelSlices));
      modelSlices->scale(0.0); // Initialize each model slices vector to zero
  		modelSlicesVector.push_back(modelSlices);

  		// Data slice
  		std::shared_ptr<SEP::double3DReg> dataSlices(new SEP::double3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);
    }

    //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
    #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
    for (int iExp=0; iExp<_nExp; iExp++){

      int iGpu = omp_get_thread_num();
  	  int iGpuId = _gpuList[iGpu];

			// Copy data slice
      memcpy(dataSlicesVector[iGpu]->getVals(), &(data->getVals()[iExp*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n]), sizeof(double)*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n);

      // Set acquisition geometry
		  if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
	          BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
	          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
		  }
		  if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
	          BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
	          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
		  }
		  if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
	          BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
	          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
		  }
		  if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
	          BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesSignalsVector[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
	          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
		  }

      // Set GPU number for propagator object
      BornObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

      //Launch modeling
      BornObjectVector[iGpu]->adjoint(true, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

	  }

		unsigned long long nz = _fdParamElastic->_nz;
    unsigned long long nx = _fdParamElastic->_nx;

    // Stack models computed by each GPU
		for (int iGpu=1; iGpu<_nGpu; iGpu++){
			#pragma omp parallel for collapse(3)
			for (int iComp=0; iComp<hyperModelSlices->getAxis(3).n; iComp++){
				for (int ix=0; ix<nx; ix++){
					for (int iz=0; iz<nz; iz++){
						(*modelSlicesVector[0]->_mat)[iComp][ix][iz] += (*modelSlicesVector[iGpu]->_mat)[iComp][ix][iz];
					}
				}
			}
		}



    //Scaling of the perturbations
		#pragma omp for collapse(2)
		for (long long ix = 0; ix < nx; ix++){
			for (long long iz = 0; iz < nz; iz++) {
				(*modelSlicesVector[0]->_mat)[0][ix][iz] *= (*_fdParamElastic->_rhoxDtwReg->_mat)[ix][iz];
				(*modelSlicesVector[0]->_mat)[1][ix][iz] *= (*_fdParamElastic->_rhozDtwReg->_mat)[ix][iz];
				(*modelSlicesVector[0]->_mat)[2][ix][iz] *= 2.0*_fdParamElastic->_dtw;
				(*modelSlicesVector[0]->_mat)[3][ix][iz] *= 2.0*_fdParamElastic->_dtw;
				(*modelSlicesVector[0]->_mat)[4][ix][iz] *= 2.0*_fdParamElastic->_dtw;
			}
		}

		//Staggering and scaling input model perturbations
    std::shared_ptr<double2DReg> temp(new double2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));
    std::shared_ptr<double2DReg> temp1(new double2DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2)));

    //stagger 2d density, mu
		std::shared_ptr<staggerX> staggerXop(new staggerX(temp,temp1));
		std::shared_ptr<staggerZ> staggerZop(new staggerZ(temp,temp1));

    //DENSITY PERTURBATION
    //D_RHOX
    std::memcpy( temp1->getVals(), model->getVals(), nx*nz*sizeof(double) );
		std::memcpy( temp->getVals(), modelSlicesVector[0]->getVals(), nx*nz*sizeof(double) );
		staggerXop->forward(true, temp, temp1);
    //D_RHOZ
    std::memcpy( temp->getVals(), modelSlicesVector[0]->getVals()+nx*nz, nx*nz*sizeof(double) );
		staggerZop->forward(true, temp, temp1);
    std::memcpy( model->getVals(), temp1->getVals(), nx*nz*sizeof(double) );

		//UNSTAGGERING D_MUXZ
    std::memcpy( temp->getVals(), modelSlicesVector[0]->getVals()+4*nx*nz, nx*nz*sizeof(double) );
		staggerZop->forward(false, temp, temp1);
    staggerXop->forward(false, temp1, temp);

		//LAME AND MU PERTUBRATION
		#pragma omp for collapse(2)
		for (long long ix = 0; ix < nx; ix++){
			for (long long iz = 0; iz < nz; iz++) {
				//D_LAME
				(*model->_mat)[1][ix][iz] += (*modelSlicesVector[0]->_mat)[2][ix][iz];
				//D_MU
				(*model->_mat)[2][ix][iz] += (*modelSlicesVector[0]->_mat)[3][ix][iz] + (*temp->_mat)[ix][iz];
			}
		}

    // Deallocate memory on device
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
      deallocateBornElasticGpu(iGpu,_gpuList[iGpu],_useStreams);
    }

}
