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

// Sources setup for Born and Tomo
// template <class V1, class V2>
// void seismicElasticOperator2D <V1, V2>::setSources(std::shared_ptr<spaceInterpGpu> sourcesDevices, std::shared_ptr<V2> sourcesSignals){
//
// 	// Set source devices
// 	_sources = sourcesDevices;
// 	_nSourcesReg = _sources->getNDeviceReg();
// 	_sourcesPositionReg = _sources->getRegPosUnique();
//
// 	// Set source signals
// 	_sourcesSignals = sourcesSignals->clone(); // Source signal read from the input file (raw)
// 	_sourcesSignalsRegDts = std::make_shared<V2>(_fdParam->_nts, _nSourcesReg, 5); // Source signal interpolated to the regular grid
// 	_sourcesSignalsRegDtsDt2 = std::make_shared<V2>(_fdParam->_nts, _nSourcesReg, 5); // Source signal with second-order time derivative
// 	_sourcesSignalsRegDtwDt2 = std::make_shared<V2>(_fdParam->_ntw, _nSourcesReg, 5); // Source signal with second-order time derivative on fine time-sampling grid
// 	_sourcesSignalsRegDtw = std::make_shared<V2>(_fdParam->_ntw, _nSourcesReg, 5); // Source signal on fine time-sampling grid
//
// 	// Interpolate spatially to regular grid
// 	_sources->adjoint(false, _sourcesSignalsRegDts, _sourcesSignals); // Interpolate sources signals to regular grid
//
// 	// Apply second time derivative to sources signals
// 	_secTimeDer->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2);
//
// 	// Scale seismic source
// 	scaleSeismicSource(_sources, _sourcesSignalsRegDtsDt2, _fdParam); // Scale sources signals by dtw^2 * vel^2
// 	scaleSeismicSource(_sources, _sourcesSignalsRegDts, _fdParam); // Scale sources signals by dtw^2 * vel^2
//
// 	// Interpolate to fine time-sampling
// 	_timeInterp->forward(false, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2); // Interpolate sources signals to fine time-sampling
// 	_timeInterp->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtw); // Interpolate sources signals to fine time-sampling
//
// }

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
	assert(checkParfileConsistency(model, data));
}

// // Set acquisiton for Born and Tomo
// template <class V1, class V2>
// void seismicOperator2D <V1, V2>::setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<deviceGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
// 	setSources(sources, sourcesSignals);
// 	setReceivers(receivers);
// 	this->setDomainRange(model, data);
// 	assert(checkParfileConsistency(model, data));
// }

// // Scale seismic source
// template <class V1, class V2>
// void seismicElasticOperator2D <V1, V2>::scaleSeismicSource(const std::shared_ptr<spaceInterpGpu> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParamElastic> parObj) const{
//
// 	std::shared_ptr<double3D> sig = signal->_mat;
// 	double *elastic = _fdParam->_elasticParam->getVals();
// 	int *pos = seismicSource->getRegPosUnique();
//
// 	#pragma omp parallel for
// 	for (int iGridPoint = 0; iGridPoint < seismicSource->getNDeviceReg(); iGridPoint++){
//     double scale_vx
//     double scale_vz
//     double scale_sigmaxx
//     double scale_sigmazz
//     double scale_sigmaxz
// 		double scale = _fdParam->_dtw * _fdParam->_dtw * v[pos[iGridPoint]]*v[pos[iGridPoint]];
// 		for (int it = 0; it < signal->getHyper()->getAxis(1).n; it++){
// 			(*sig)[iGridPoint][it] = (*sig)[iGridPoint][it] * scale;
// 		}
// 	}
// }

// Wavefield setup
template <class V1, class V2>
std::shared_ptr<double4DReg> seismicElasticOperator2D <V1, V2>:: setWavefield(int wavefieldFlag){

	_saveWavefield = wavefieldFlag;

	std::shared_ptr<double4DReg> wavefield;
	if (wavefieldFlag == 1) {
		wavefield = std::make_shared<double4DReg>(_fdParamElastic->_zAxis, _fdParamElastic->_xAxis, _fdParamElastic->_wavefieldCompAxis, _fdParamElastic->_timeAxisCoarse);
		unsigned long long int wavefieldSize = _fdParamElastic->_wavefieldCompAxis.n * _fdParamElastic->_zAxis.n * _fdParamElastic->_xAxis.n;
		wavefieldSize *= _fdParamElastic->_nts*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
	else {
		wavefield = std::make_shared<double4DReg>(1, 1, 1, 1);
		unsigned long long int wavefieldSize = 1*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
}
