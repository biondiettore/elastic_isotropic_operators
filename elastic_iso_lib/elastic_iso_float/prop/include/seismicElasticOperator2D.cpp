template <class V1, class V2>
void seismicElasticOperator2D <V1, V2>::setSources(std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid){
	_sourcesCenterGrid = sourcesCenterGrid;
	_sourcesXGrid = sourcesXGrid;
	_sourcesZGrid = sourcesZGrid;
	_sourcesXZGrid = sourcesXZGrid;

	_nSourcesRegCenterGrid = _sourcesCenterGrid->getNDeviceReg();
	_nSourcesRegXGrid = _sourcesXGrid->getNDeviceReg();
	_nSourcesRegZGrid = _sourcesZGrid->getNDeviceReg();
	_nSourcesRegXZGrid = _sourcesXZGrid->getNDeviceReg();

  	_nSourcesIrregCenterGrid = _sourcesCenterGrid->getNDeviceIrreg();
	_nSourcesIrregXGrid = _sourcesXGrid->getNDeviceIrreg();
	_nSourcesIrregZGrid = _sourcesZGrid->getNDeviceIrreg();
	_nSourcesIrregXZGrid = _sourcesXZGrid->getNDeviceIrreg();

	_sourcesPositionRegCenterGrid = _sourcesCenterGrid->getRegPosUnique();
	_sourcesPositionRegXGrid = _sourcesXGrid->getRegPosUnique();
	_sourcesPositionRegZGrid = _sourcesZGrid->getRegPosUnique();
	_sourcesPositionRegXZGrid = _sourcesXZGrid->getRegPosUnique();

}

// Sources setup for Born
template <class V1, class V2>
void seismicElasticOperator2D <V1, V2>::setSources(std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid, std::shared_ptr<V2> sourcesSignals){

	_sourcesCenterGrid = sourcesCenterGrid;
	_sourcesXGrid = sourcesXGrid;
	_sourcesZGrid = sourcesZGrid;
	_sourcesXZGrid = sourcesXZGrid;

	_nSourcesRegCenterGrid = _sourcesCenterGrid->getNDeviceReg();
	_nSourcesRegXGrid = _sourcesXGrid->getNDeviceReg();
	_nSourcesRegZGrid = _sourcesZGrid->getNDeviceReg();
	_nSourcesRegXZGrid = _sourcesXZGrid->getNDeviceReg();

    _nSourcesIrregCenterGrid = _sourcesCenterGrid->getNDeviceIrreg();
	_nSourcesIrregXGrid = _sourcesXGrid->getNDeviceIrreg();
	_nSourcesIrregZGrid = _sourcesZGrid->getNDeviceIrreg();
	_nSourcesIrregXZGrid = _sourcesXZGrid->getNDeviceIrreg();

	_sourcesPositionRegCenterGrid = _sourcesCenterGrid->getRegPosUnique();
	_sourcesPositionRegXGrid = _sourcesXGrid->getRegPosUnique();
	_sourcesPositionRegZGrid = _sourcesZGrid->getRegPosUnique();
	_sourcesPositionRegXZGrid = _sourcesXZGrid->getRegPosUnique();

	//Constructing source term
	_sourcesSignals = sourcesSignals->clone();

	//Memory allocation
	std::shared_ptr<float2DReg> sourceTemp_vx(new float2DReg(_fdParamElastic->_nts, _nSourcesIrregXGrid));
	std::shared_ptr<float2DReg> sourceTemp_vz(new float2DReg(_fdParamElastic->_nts, _nSourcesIrregZGrid));
	std::shared_ptr<float2DReg> sourceTemp_sigmaxx(new float2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<float2DReg> sourceTemp_sigmazz(new float2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<float2DReg> sourceTemp_sigmaxz(new float2DReg(_fdParamElastic->_nts, _nSourcesIrregXZGrid));

	std::shared_ptr<float2DReg> sourceRegDts_vx(new float2DReg(_fdParamElastic->_nts, _nSourcesRegXGrid));
	std::shared_ptr<float2DReg> sourceRegDts_vz(new float2DReg(_fdParamElastic->_nts, _nSourcesRegZGrid));
	std::shared_ptr<float2DReg> sourceRegDts_sigmaxx(new float2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
	std::shared_ptr<float2DReg> sourceRegDts_sigmazz(new float2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
	std::shared_ptr<float2DReg> sourceRegDts_sigmaxz(new float2DReg(_fdParamElastic->_nts, _nSourcesRegXZGrid));

	_sourceRegDtw_vx = std::make_shared<float2DReg>(_fdParamElastic->_ntw, _nSourcesRegXGrid);
	_sourceRegDtw_vz = std::make_shared<float2DReg>(_fdParamElastic->_ntw, _nSourcesRegZGrid);
	_sourceRegDtw_sigmaxx = std::make_shared<float2DReg>(_fdParamElastic->_ntw, _nSourcesRegCenterGrid);
	_sourceRegDtw_sigmazz = std::make_shared<float2DReg>(_fdParamElastic->_ntw, _nSourcesRegCenterGrid);
	_sourceRegDtw_sigmaxz = std::make_shared<float2DReg>(_fdParamElastic->_ntw, _nSourcesRegXZGrid);

	/* Copy from 3d source to respective 2d source components */
	std::memcpy( sourceTemp_vx->getVals(), _sourcesSignals->getVals(), _nSourcesIrregXGrid*_fdParamElastic->_nts*sizeof(float) );
	std::memcpy( sourceTemp_vz->getVals(), _sourcesSignals->getVals()+_nSourcesIrregXGrid*_fdParamElastic->_nts, _nSourcesIrregZGrid*_fdParamElastic->_nts*sizeof(float) );
	std::memcpy( sourceTemp_sigmaxx->getVals(), _sourcesSignals->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(float) );
	std::memcpy( sourceTemp_sigmazz->getVals(), _sourcesSignals->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(float) );
	std::memcpy( sourceTemp_sigmaxz->getVals(), _sourcesSignals->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+2*_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregXZGrid*_fdParamElastic->_nts*sizeof(float) );

	/* Interpolate source (seismic source) to regular grid */
	_sourcesXGrid->adjoint(false, sourceRegDts_vx, sourceTemp_vx);
	_sourcesZGrid->adjoint(false, sourceRegDts_vz, sourceTemp_vz);
	_sourcesCenterGrid->adjoint(false, sourceRegDts_sigmaxx, sourceTemp_sigmaxx);
	_sourcesCenterGrid->adjoint(false, sourceRegDts_sigmazz, sourceTemp_sigmazz);
	_sourcesXZGrid->adjoint(false, sourceRegDts_sigmaxz, sourceTemp_sigmaxz);

	/* Scale source signals */
	sourceRegDts_sigmaxx->scale(2.0*_fdParamElastic->_dtw);
	sourceRegDts_sigmazz->scale(2.0*_fdParamElastic->_dtw);
	#pragma omp parallel for collapse(2)
	for(int is = 0; is < _nSourcesRegXGrid; is++){ //loop over number of reg sources x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*sourceRegDts_vx->_mat)[is][it] *= _fdParamElastic->_rhoxDtw[(_sourcesXGrid->getRegPosUnique())[is]];
		}
	}
	#pragma omp parallel for collapse(2)
	for(int is = 0; is < _nSourcesRegZGrid; is++){ //loop over number of reg sources z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			(*sourceRegDts_vz->_mat)[is][it] *= _fdParamElastic->_rhozDtw[(_sourcesZGrid->getRegPosUnique())[is]];
		}
	}
	sourceRegDts_sigmaxz->scale(2.0*_fdParamElastic->_dtw);

	/*Scaling by the inverse of the space discretization*/
	float area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dz);
	sourceRegDts_sigmaxx->scale(area_scale);
	sourceRegDts_sigmazz->scale(area_scale);
	sourceRegDts_vx->scale(area_scale);
	sourceRegDts_vz->scale(area_scale);
	sourceRegDts_sigmaxz->scale(area_scale);

	/* Interpolate to fine time-sampling */
	_timeInterp->forward(false, sourceRegDts_vx, _sourceRegDtw_vx);
	_timeInterp->forward(false, sourceRegDts_vz, _sourceRegDtw_vz);
	_timeInterp->forward(false, sourceRegDts_sigmaxx, _sourceRegDtw_sigmaxx);
	_timeInterp->forward(false, sourceRegDts_sigmazz, _sourceRegDtw_sigmazz);
	_timeInterp->forward(false, sourceRegDts_sigmaxz, _sourceRegDtw_sigmaxz);



}

// Receivers setup for Nonlinear modeling, Born and Tomo
template <class V1, class V2>
void seismicElasticOperator2D <V1, V2>::setReceivers(std::shared_ptr<spaceInterpGpu> receiversCenterGrid, std::shared_ptr<spaceInterpGpu> receiversXGrid, std::shared_ptr<spaceInterpGpu> receiversZGrid, std::shared_ptr<spaceInterpGpu> receiversXZGrid){
	_receiversCenterGrid = receiversCenterGrid;
	_receiversXGrid = receiversXGrid;
	_receiversZGrid = receiversZGrid;
	_receiversXZGrid = receiversXZGrid;

	_nReceiversRegCenterGrid = _receiversCenterGrid->getNDeviceReg();
	_nReceiversRegXGrid = _receiversXGrid->getNDeviceReg();
	_nReceiversRegZGrid = _receiversZGrid->getNDeviceReg();
	_nReceiversRegXZGrid = _receiversXZGrid->getNDeviceReg();

	_nReceiversIrregCenterGrid = _receiversCenterGrid->getNDeviceIrreg();
	_nReceiversIrregXGrid = _receiversXGrid->getNDeviceIrreg();
	_nReceiversIrregZGrid = _receiversZGrid->getNDeviceIrreg();
	_nReceiversIrregXZGrid = _receiversXZGrid->getNDeviceIrreg();

	_receiversPositionRegCenterGrid = _receiversCenterGrid->getRegPosUnique();
	_receiversPositionRegXGrid = _receiversXGrid->getRegPosUnique();
	_receiversPositionRegZGrid = _receiversZGrid->getRegPosUnique();
	_receiversPositionRegXZGrid = _receiversXZGrid->getRegPosUnique();
}

// Set acquisiton for Nonlinear modeling
template <class V1, class V2>
void seismicElasticOperator2D <V1, V2>::setAcquisition(
	std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid,
	std::shared_ptr<spaceInterpGpu> receiversCenterGrid, std::shared_ptr<spaceInterpGpu> receiversXGrid, std::shared_ptr<spaceInterpGpu> receiversZGrid, std::shared_ptr<spaceInterpGpu> receiversXZGrid,
	const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources(sourcesCenterGrid, sourcesXGrid, sourcesZGrid, sourcesXZGrid);
	setReceivers(receiversCenterGrid, receiversXGrid, receiversZGrid, receiversXZGrid);
	this->setDomainRange(model, data);
	if ( not checkParfileConsistency(model, data)){
		throw std::runtime_error("");
	}
}

// Set acquisiton for Nonlinear modeling
template <class V1, class V2>
void seismicElasticOperator2D <V1, V2>::setAcquisition(
	std::shared_ptr<spaceInterpGpu> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu> sourcesXGrid, std::shared_ptr<spaceInterpGpu> sourcesZGrid, std::shared_ptr<spaceInterpGpu> sourcesXZGrid, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<spaceInterpGpu> receiversCenterGrid, std::shared_ptr<spaceInterpGpu> receiversXGrid, std::shared_ptr<spaceInterpGpu> receiversZGrid, std::shared_ptr<spaceInterpGpu> receiversXZGrid, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources(sourcesCenterGrid, sourcesXGrid, sourcesZGrid, sourcesXZGrid, sourcesSignals);
	setReceivers(receiversCenterGrid, receiversXGrid, receiversZGrid, receiversXZGrid);
	this->setDomainRange(model, data);
	if (not checkParfileConsistency(model, data)){
		throw std::runtime_error("");
	};
}


// Wavefield setup
template <class V1, class V2>
std::shared_ptr<float4DReg> seismicElasticOperator2D <V1, V2>:: setWavefield(int wavefieldFlag){

	_saveWavefield = wavefieldFlag;

	std::shared_ptr<float4DReg> wavefield;
	if (wavefieldFlag == 1) {
		wavefield = std::make_shared<float4DReg>(_fdParamElastic->_zAxis, _fdParamElastic->_xAxis, _fdParamElastic->_wavefieldCompAxis, _fdParamElastic->_timeAxisCoarse);
		unsigned long long int wavefieldSize = _fdParamElastic->_wavefieldCompAxis.n * _fdParamElastic->_zAxis.n * _fdParamElastic->_xAxis.n;
		wavefieldSize *= _fdParamElastic->_nts*sizeof(float);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
	else {
		wavefield = std::make_shared<float4DReg>(1, 1, 1, 1);
		unsigned long long int wavefieldSize = 1*sizeof(float);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
}
