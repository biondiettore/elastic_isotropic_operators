#include <vector>
#include <omp.h>
#include "nonlinearPropElasticShotsGpu.h"
#include "nonlinearPropElasticGpu.h"
#include <ctime>

nonlinearPropElasticShotsGpu::nonlinearPropElasticShotsGpu(std::shared_ptr<SEP::float3DReg> elasticParam,
                              std::shared_ptr<paramObj> par,
                              std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorCenterGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorZGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorCenterGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorZGrid,
                              std::vector<std::shared_ptr<spaceInterpGpu>> receiversVectorXZGrid){

    // Setup parameters
  	_par = par;
  	_elasticParam = elasticParam;
  	_nExp = par->getInt("nExp");
  	// _nGpu = par->getInt("nGpu");
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

void nonlinearPropElasticShotsGpu::createGpuIdList(){

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

// Forward
void nonlinearPropElasticShotsGpu::forward(const bool add,
                                            const std::shared_ptr<float4DReg> model,
                                            std::shared_ptr<float4DReg> data) const{

    if (!add) data->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    //check that we have five wavefield componenets
    if (model->getHyper()->getAxis(3).n != 5) {assert(1==2);}

    // Check whether we use the same source signals for all shots
    if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<nonlinearPropElasticGpu>> propObjectVector;

    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Nonlinear propagator object
  		std::shared_ptr<nonlinearPropElasticGpu> propGpuObject(new nonlinearPropElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		propObjectVector.push_back(propGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			propGpuObject->getFdParam()->getInfo();
  		}

  		// Model slice
  		std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
  		modelSlicesVector.push_back(modelSlices);

  		// Data slice
  		std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);

  	}

  // Launch nonlinear forward

  //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
  #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
  for (int iExp=0; iExp<_nExp; iExp++){

    int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

    // Copy model slice
    if(constantSrcSignal == 1) {
      memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(float)*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n);
    } else {
      memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[iExp*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n]), sizeof(float)*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n);
    }
    // Set acquisition geometry
    if (constantRecGeom == 1) {
      propObjectVector[iGpu]->setAcquisition(
        _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
        _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
        modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
    } else {
      propObjectVector[iGpu]->setAcquisition(
        _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
        _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
        modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
    }

    // Set GPU number for propagator object
    propObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

    //Launch modeling
    propObjectVector[iGpu]->forward(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

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
    deallocateNonlinearElasticGpu(iGpu,_gpuList[iGpu]);
  }
}
void nonlinearPropElasticShotsGpu::forwardWavefield(const bool add,
                                            const std::shared_ptr<float4DReg> model,
                                            std::shared_ptr<float4DReg> data){

    if (!add) data->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    //check that we have five wavefield componenets
    if (model->getHyper()->getAxis(3).n != 5) {assert(1==2);}

    // Check whether we use the same source signals for all shots
    if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<nonlinearPropElasticGpu>> propObjectVector;

    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Nonlinear propagator object
  		std::shared_ptr<nonlinearPropElasticGpu> propGpuObject(new nonlinearPropElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		propObjectVector.push_back(propGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			propGpuObject->getFdParam()->getInfo();
  		}

  		// Model slice
  		std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
  		modelSlicesVector.push_back(modelSlices);
      //std::cerr << "pushed model slice " << iGpu << '\n';

  		// Data slice
  		std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);
      //std::cerr << "pushed data slice " << iGpu << '\n';

  	}

  // Launch nonlinear forward

  //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
  #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
  for (int iExp=0; iExp<_nExp; iExp++){

    int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

    // Change the wavefield flag
		if (iExp == _wavefieldShotNumber && _saveWavefield !=0 ) {
			propObjectVector[iGpu]->setAllWavefields(1);
		} else {
			propObjectVector[iGpu]->setAllWavefields(0);
		}

    // Copy model slice
    if(constantSrcSignal == 1) {
      memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(float)*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n);
    } else {
      memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[iExp*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n]), sizeof(float)*hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n*hyperModelSlices->getAxis(3).n);
    }
    // Set acquisition geometry
    if (constantRecGeom == 1) {
      propObjectVector[iGpu]->setAcquisition(
        _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
        _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
        modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
    } else {
      propObjectVector[iGpu]->setAcquisition(
        _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
        _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
        modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
    }

    // Set GPU number for propagator object
    propObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

    //Launch modeling
    propObjectVector[iGpu]->forward(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

    // Store dataSlice into data
    #pragma omp parallel for collapse(3)
    for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
      for (int iReceiver=0; iReceiver<hyperDataSlices->getAxis(2).n; iReceiver++){
        for (int its=0; its<hyperDataSlices->getAxis(1).n; its++){
          (*data->_mat)[iExp][iwc][iReceiver][its] += (*dataSlicesVector[iGpu]->_mat)[iwc][iReceiver][its];
        }
      }
    }


    // Get the wavefield
    if (iExp == _wavefieldShotNumber && _saveWavefield !=0 ) {
      _wavefield = propObjectVector[iGpu]->getWavefield();

    }
  }



  // Deallocate memory on device
  for (int iGpu=0; iGpu<_nGpu; iGpu++){
    deallocateNonlinearElasticGpu(iGpu,_gpuList[iGpu]);
  }
}

// Adjoint
void nonlinearPropElasticShotsGpu::adjoint(const bool add,
                std::shared_ptr<SEP::float4DReg> model,
                const std::shared_ptr<SEP::float4DReg> data) const{

  	if (!add) model->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    //check that we have five wavefield componenets
    if (model->getHyper()->getAxis(3).n != 5) {assert(1==2);}

    // Check whether we use the same source signals for all shots
    if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<nonlinearPropElasticGpu>> propObjectVector;

    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Nonlinear propagator object
  		std::shared_ptr<nonlinearPropElasticGpu> propGpuObject(new nonlinearPropElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		propObjectVector.push_back(propGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			propGpuObject->getFdParam()->getInfo();
  		}


  		// Model slice
  		std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
  		modelSlicesVector.push_back(modelSlices);
        modelSlicesVector[iGpu]->scale(0.0); // Initialize each model slices vector to zero


  		// Data slice
  		std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);
    }

    // Launch nonlinear adjoint

    //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
    #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
    for (int iExp=0; iExp<_nExp; iExp++){

      int iGpu = omp_get_thread_num();
			int iGpuId = _gpuList[iGpu];

      // Copy data slice
      memcpy(dataSlicesVector[iGpu]->getVals(), &(data->getVals()[iExp*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n]), sizeof(float)*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n);

      // Set acquisition geometry
      if (constantRecGeom == 1) {
        propObjectVector[iGpu]->setAcquisition(
          _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
          _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      } else {
        propObjectVector[iGpu]->setAcquisition(
          _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
          _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      }

      // Set GPU number for propagator object
      propObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

      // Launch modeling
      if (constantSrcSignal == 1){
      	// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
        propObjectVector[iGpu]->adjoint(true, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      }
      else {
        // Copy the shot into model slice --> Is there a way to parallelize this?
        propObjectVector[iGpu]->adjoint(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
        #pragma omp parallel for collapse(3)
        for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
          for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
            for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
            	(*model->_mat)[iExp][iwc][iSource][its] += (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
            }
          }
        }
      }

    }

    // If same sources for all shots, stack all shots from all iGpus
    if (constantSrcSignal == 1){
      #pragma omp parallel for collapse(4)
      for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
      	for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
      		for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
      			for (int iGpu=0; iGpu<_nGpu; iGpu++){
      				(*model->_mat)[0][iwc][iSource][its]	+= (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
      			}
      		}
      	}
      }
    }
    // Deallocate memory on device
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
  		deallocateNonlinearElasticGpu(iGpu,_gpuList[iGpu]);
  	}
}
// Adjoint with wavefield saving
void nonlinearPropElasticShotsGpu::adjointWavefield(const bool add, std::shared_ptr<float4DReg> model, const std::shared_ptr<float4DReg> data){

  	if (!add) model->scale(0.0);

    // Variable declaration
  	int omp_get_thread_num();
  	int constantSrcSignal, constantRecGeom;

    //check that we have five wavefield componenets
    if (model->getHyper()->getAxis(3).n != 5) {assert(1==2);}

    // Check whether we use the same source signals for all shots
    if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
    else {constantRecGeom=0;}

    // Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
    std::vector<std::shared_ptr<nonlinearPropElasticGpu>> propObjectVector;
    // Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Nonlinear propagator object
  		std::shared_ptr<nonlinearPropElasticGpu> propGpuObject(new nonlinearPropElasticGpu(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		propObjectVector.push_back(propGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			propGpuObject->getFdParam()->getInfo();
  		}


  		// Model slice
  		std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
  		modelSlicesVector.push_back(modelSlices);
      modelSlicesVector[iGpu]->scale(0.0); // Initialize each model slices vector to zero


  		// Data slice
  		std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);
    }
    std::cerr << "BEGIN PROP" << '\n';

    // Launch nonlinear adjoint

    //will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
    #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
    for (int iExp=0; iExp<_nExp; iExp++){

      int iGpu = omp_get_thread_num();
			int iGpuId = _gpuList[iGpu];

      // Change the saveWavefield flag
      if (iExp == _wavefieldShotNumber) { propObjectVector[iGpu]->setAllWavefields(1);}
      std::cerr << "RESET WAVEFIELD" << '\n';

      // Copy data slice
      memcpy(dataSlicesVector[iGpu]->getVals(), &(data->getVals()[iExp*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n]), sizeof(float)*hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n*hyperDataSlices->getAxis(3).n);

      std::cerr << "COPIED DATA" << '\n';

      // Set acquisition geometry
      if (constantRecGeom == 1) {
        propObjectVector[iGpu]->setAcquisition(
          _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
          _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0],
          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      } else {
        propObjectVector[iGpu]->setAcquisition(
          _sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp],
          _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp],
          modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      }
        std::cerr << "SET ACQ GEOM" << '\n';

      // Set GPU number for propagator object
      propObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

      // Launch modeling
      if (constantSrcSignal == 1){
      	// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
        propObjectVector[iGpu]->adjoint(true, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
      }
      else {
        // Copy the shot into model slice --> Is there a way to parallelize this?
        propObjectVector[iGpu]->adjoint(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
        #pragma omp parallel for collapse(3)
        for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
          for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
            for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
            	(*model->_mat)[iExp][iwc][iSource][its] += (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
            }
          }
        }
      }
      std::cerr << "PROPED" << '\n';
      //std::cerr << "modelSlicesVector[" << iGpu << "]->norm()= " << modelSlicesVector[iGpu]->norm() << std::endl;
      // Get the wavefield
      if (iExp == _wavefieldShotNumber) {
        _wavefield = propObjectVector[iGpu]->getWavefield();
      }
    }



    // If same sources for all shots, stack all shots from all iGpus
    if (constantSrcSignal == 1){
      #pragma omp parallel for collapse(4)
      for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
      	for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
      		for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
      			for (int iGpu=0; iGpu<_nGpu; iGpu++){
      				(*model->_mat)[0][iwc][iSource][its]	+= (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
      			}
      		}
      	}
      }
      std::cerr << "COPIED WFLD" << '\n';

    }
    // Deallocate memory on device
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
  		deallocateNonlinearElasticGpu(iGpu,_gpuList[iGpu]);
  	}
}
