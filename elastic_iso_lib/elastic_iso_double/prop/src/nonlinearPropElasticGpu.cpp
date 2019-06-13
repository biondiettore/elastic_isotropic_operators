#include <vector>
#include <ctime>
#include "nonlinearPropElasticGpu.h"
#include <cstring>

nonlinearPropElasticGpu::nonlinearPropElasticGpu(std::shared_ptr<fdParamElastic> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParamElastic = fdParamElastic;
	//_fdParamElastic = std::make_shared<fdParamElastic>(elasticParam, par);
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParamElastic->_nts, _fdParamElastic->_dts, _fdParamElastic->_ots, _fdParamElastic->_sub);
	//setAllWavefields(par->getInt("saveWavefield", 0));
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_saveWavefield = par->getInt("saveWavefield", 0);
	_useStreams = par->getInt("useStreams", 0); //Flag whether to use streams to save the wavefield

	// Initialize GPU
	initNonlinearElasticGpu(_fdParamElastic->_dz, _fdParamElastic->_dx, _fdParamElastic->_nz, _fdParamElastic->_nx, _fdParamElastic->_nts, _fdParamElastic->_dts, _fdParamElastic->_sub, _fdParamElastic->_minPad, _fdParamElastic->_blockSize, _fdParamElastic->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	/// Alocate on GPUs
	allocateNonlinearElasticGpu(_fdParamElastic->_rhoxDtw,
														  _fdParamElastic->_rhozDtw,
															_fdParamElastic->_lamb2MuDtw,
															_fdParamElastic->_lambDtw,
															_fdParamElastic->_muxzDtw,
															_iGpu, iGpuId);
	setAllWavefields(0); // By default, do not record the scattered wavefields
}

void nonlinearPropElasticGpu::setAllWavefields(int wavefieldFlag){
	_wavefield = setWavefield(wavefieldFlag);
}

bool nonlinearPropElasticGpu::checkParfileConsistency(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{

	if (_fdParamElastic->checkParfileConsistencyTime(data, 1) != true) {return false;} // Check data time axis
	if (_fdParamElastic->checkParfileConsistencyTime(model,1) != true) {return false;}; // Check model time axis

	return true;
}

// model is seismic sources and data are receiver recordings
void nonlinearPropElasticGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

  /* Allocation */
  std::shared_ptr<double2DReg> modelTemp_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXGrid));
  std::shared_ptr<double2DReg> modelTemp_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregZGrid));
  std::shared_ptr<double2DReg> modelTemp_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> modelTemp_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> modelTemp_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXZGrid));

  std::shared_ptr<double2DReg> modelRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXGrid));
  std::shared_ptr<double2DReg> modelRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegZGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXZGrid));

  std::shared_ptr<double2DReg> modelRegDtw_vx(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegXGrid));
  std::shared_ptr<double2DReg> modelRegDtw_vz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegZGrid));
  std::shared_ptr<double2DReg> modelRegDtw_sigmaxx(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDtw_sigmazz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDtw_sigmaxz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegXZGrid));

  std::shared_ptr<double2DReg> dataRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXGrid));
  std::shared_ptr<double2DReg> dataRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegZGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXZGrid));

  std::shared_ptr<double2DReg> dataTemp_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXGrid));
  std::shared_ptr<double2DReg> dataTemp_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregZGrid));
  std::shared_ptr<double2DReg> dataTemp_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
  std::shared_ptr<double2DReg> dataTemp_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
  std::shared_ptr<double2DReg> dataTemp_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXZGrid));

  if (!add){
	  data->scale(0.0);
  } else {
	  /* Copy the data to the temporary array */
	  std::memcpy(dataTemp_vx->getVals(),data->getVals(), _nReceiversIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
	  std::memcpy(dataTemp_vz->getVals(), data->getVals()+_nReceiversIrregXGrid*_fdParamElastic->_nts, _nReceiversIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
	  std::memcpy(dataTemp_sigmaxx->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid)*_fdParamElastic->_nts, _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	  std::memcpy(dataTemp_sigmazz->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	  std::memcpy(dataTemp_sigmaxz->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+2*_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, _nReceiversIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );

  }

  /* Copy from 3d model to respective 2d model components */
  std::memcpy( modelTemp_vx->getVals(), model->getVals(), _nSourcesIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy( modelTemp_vz->getVals(), model->getVals()+_nSourcesIrregXGrid*_fdParamElastic->_nts, _nSourcesIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy( modelTemp_sigmaxx->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy( modelTemp_sigmazz->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy( modelTemp_sigmaxz->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+2*_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );

  /* Interpolate model (seismic source) to regular grid */
  _sourcesXGrid->adjoint(false, modelRegDts_vx, modelTemp_vx);
  _sourcesZGrid->adjoint(false, modelRegDts_vz, modelTemp_vz);
  _sourcesCenterGrid->adjoint(false, modelRegDts_sigmaxx, modelTemp_sigmaxx);
  _sourcesCenterGrid->adjoint(false, modelRegDts_sigmazz, modelTemp_sigmazz);
  _sourcesXZGrid->adjoint(false, modelRegDts_sigmaxz, modelTemp_sigmaxz);
  /* Scale source signals model */
  modelRegDts_sigmaxx->scale(2.0*_fdParamElastic->_dtw);
  modelRegDts_sigmazz->scale(2.0*_fdParamElastic->_dtw);
  #pragma omp parallel for collapse(2)
  for(int is = 0; is < _nSourcesRegXGrid; is++){ //loop over number of reg sources x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_vx->_mat)[is][it] *= _fdParamElastic->_rhoxDtw[(_sourcesXGrid->getRegPosUnique())[is]];
		}
  }
  #pragma omp parallel for collapse(2)
  for(int is = 0; is < _nSourcesRegZGrid; is++){ //loop over number of reg sources z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_vz->_mat)[is][it] *= _fdParamElastic->_rhozDtw[(_sourcesZGrid->getRegPosUnique())[is]];
		}
  }
  modelRegDts_sigmaxz->scale(2.0*_fdParamElastic->_dtw);

  /*Scaling by the inverse of the space discretization*/
  double area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dz);
  modelRegDts_sigmaxx->scale(area_scale);
  modelRegDts_sigmazz->scale(area_scale);
  modelRegDts_vx->scale(area_scale);
  modelRegDts_vz->scale(area_scale);
  modelRegDts_sigmaxz->scale(area_scale);

  /* Interpolate to fine time-sampling */
  _timeInterp->forward(false, modelRegDts_vx, modelRegDtw_vx);
  _timeInterp->forward(false, modelRegDts_vz, modelRegDtw_vz);
  _timeInterp->forward(false, modelRegDts_sigmaxx, modelRegDtw_sigmaxx);
  _timeInterp->forward(false, modelRegDts_sigmazz, modelRegDtw_sigmazz);
  _timeInterp->forward(false, modelRegDts_sigmaxz, modelRegDtw_sigmaxz);

	// /* Propagate */
	if (_saveWavefield == 0) {
		propShotsElasticFwdGpu(modelRegDtw_vx->getVals(),
													modelRegDtw_vz->getVals(),
													modelRegDtw_sigmaxx->getVals(),
													modelRegDtw_sigmazz->getVals(),
													modelRegDtw_sigmaxz->getVals(),
													dataRegDts_vx->getVals(),
													dataRegDts_vz->getVals(),
													dataRegDts_sigmaxx->getVals(),
													dataRegDts_sigmazz->getVals(),
													dataRegDts_sigmaxz->getVals(),
													_sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid,
													_sourcesPositionRegXGrid, _nSourcesRegXGrid,
													_sourcesPositionRegZGrid, _nSourcesRegZGrid,
													_sourcesPositionRegXZGrid, _nSourcesRegXZGrid,
													_receiversPositionRegCenterGrid, _nReceiversRegCenterGrid,
													_receiversPositionRegXGrid, _nReceiversRegXGrid,
													_receiversPositionRegZGrid, _nReceiversRegZGrid,
													_receiversPositionRegXZGrid, _nReceiversRegXZGrid,
													 _iGpu, _iGpuId, _fdParamElastic->_surfaceCondition);
	} else {
			//Saving wavefield with or w/o streams
			if(_useStreams == 0){
				propShotsElasticFwdGpuWavefield(modelRegDtw_vx->getVals(),
															modelRegDtw_vz->getVals(),
															modelRegDtw_sigmaxx->getVals(),
															modelRegDtw_sigmazz->getVals(),
															modelRegDtw_sigmaxz->getVals(),
															dataRegDts_vx->getVals(),
															dataRegDts_vz->getVals(),
															dataRegDts_sigmaxx->getVals(),
															dataRegDts_sigmazz->getVals(),
															dataRegDts_sigmaxz->getVals(),
															_sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid,
															_sourcesPositionRegXGrid, _nSourcesRegXGrid,
															_sourcesPositionRegZGrid, _nSourcesRegZGrid,
															_sourcesPositionRegXZGrid, _nSourcesRegXZGrid,
															_receiversPositionRegCenterGrid, _nReceiversRegCenterGrid,
															_receiversPositionRegXGrid, _nReceiversRegXGrid,
															_receiversPositionRegZGrid, _nReceiversRegZGrid,
															_receiversPositionRegXZGrid, _nReceiversRegXZGrid,
															_wavefield->getVals(),
															 _iGpu, _iGpuId);
			} else {
				propShotsElasticFwdGpuWavefieldStreams(modelRegDtw_vx->getVals(),
															modelRegDtw_vz->getVals(),
															modelRegDtw_sigmaxx->getVals(),
															modelRegDtw_sigmazz->getVals(),
															modelRegDtw_sigmaxz->getVals(),
															dataRegDts_vx->getVals(),
															dataRegDts_vz->getVals(),
															dataRegDts_sigmaxx->getVals(),
															dataRegDts_sigmazz->getVals(),
															dataRegDts_sigmaxz->getVals(),
															_sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid,
															_sourcesPositionRegXGrid, _nSourcesRegXGrid,
															_sourcesPositionRegZGrid, _nSourcesRegZGrid,
															_sourcesPositionRegXZGrid, _nSourcesRegXZGrid,
															_receiversPositionRegCenterGrid, _nReceiversRegCenterGrid,
															_receiversPositionRegXGrid, _nReceiversRegXGrid,
															_receiversPositionRegZGrid, _nReceiversRegZGrid,
															_receiversPositionRegXZGrid, _nReceiversRegXZGrid,
															_wavefield->getVals(),
															 _iGpu, _iGpuId);
			}
	}


  /* Interpolate to irregular grid */
  _receiversXGrid->forward(true, dataRegDts_vx, dataTemp_vx);
  _receiversZGrid->forward(true, dataRegDts_vz, dataTemp_vz);
  _receiversCenterGrid->forward(true, dataRegDts_sigmaxx, dataTemp_sigmaxx);
  _receiversCenterGrid->forward(true, dataRegDts_sigmazz, dataTemp_sigmazz);
  _receiversXZGrid->forward(true, dataRegDts_sigmaxz, dataTemp_sigmaxz);

  /* Copy each component data into one cube */
  std::memcpy(data->getVals(), dataTemp_vx->getVals(), _nReceiversIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy(data->getVals()+_nReceiversIrregXGrid*_fdParamElastic->_nts, dataTemp_vz->getVals(), _nReceiversIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy(data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid)*_fdParamElastic->_nts, dataTemp_sigmaxx->getVals(), _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy(data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, dataTemp_sigmazz->getVals(), _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
  std::memcpy(data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+2*_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, dataTemp_sigmaxz->getVals(), _nReceiversIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );
}

void nonlinearPropElasticGpu::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const {
	// 	/* Allocation */
	std::shared_ptr<double2DReg> modelTemp_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXGrid));
	std::shared_ptr<double2DReg> modelTemp_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregZGrid));
	std::shared_ptr<double2DReg> modelTemp_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<double2DReg> modelTemp_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<double2DReg> modelTemp_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXZGrid));

	std::shared_ptr<double2DReg> modelRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXGrid));
	std::shared_ptr<double2DReg> modelRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegZGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXZGrid));

	std::shared_ptr<double2DReg> modelRegDtw_vx(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegXGrid));
	std::shared_ptr<double2DReg> modelRegDtw_vz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegZGrid));
	std::shared_ptr<double2DReg> modelRegDtw_sigmaxx(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegCenterGrid));
	std::shared_ptr<double2DReg> modelRegDtw_sigmazz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegCenterGrid));
	std::shared_ptr<double2DReg> modelRegDtw_sigmaxz(new double2DReg(_fdParamElastic->_ntw, _nSourcesRegXZGrid));

	std::shared_ptr<double2DReg> dataRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXGrid));
	std::shared_ptr<double2DReg> dataRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegZGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXZGrid));

	std::shared_ptr<double2DReg> dataTemp_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXGrid));
	std::shared_ptr<double2DReg> dataTemp_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregZGrid));
	std::shared_ptr<double2DReg> dataTemp_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
	std::shared_ptr<double2DReg> dataTemp_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
	std::shared_ptr<double2DReg> dataTemp_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXZGrid));

	if (!add) {
		model->scale(0.0);
		modelTemp_vx -> scale(0.0);
		modelTemp_vz -> scale(0.0);
		modelTemp_sigmaxx -> scale(0.0);
		modelTemp_sigmazz -> scale(0.0);
		modelTemp_sigmaxz -> scale(0.0);
	} else {
		/* Copy each source into one cube */
		std::memcpy(modelTemp_vx->getVals(), model->getVals(), _nSourcesIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
		std::memcpy(modelTemp_vz->getVals(), model->getVals()+_nSourcesIrregXGrid*_fdParamElastic->_nts, _nSourcesIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
		std::memcpy(modelTemp_sigmaxx->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
		std::memcpy(modelTemp_sigmazz->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
		std::memcpy(modelTemp_sigmaxz->getVals(), model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+2*_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, _nSourcesIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );
	}

	/* Copy from 3d model to respective 2d components*/
	std::memcpy( dataTemp_vx->getVals(), data->getVals(), _nReceiversIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy( dataTemp_vz->getVals(), data->getVals()+_nReceiversIrregXGrid*_fdParamElastic->_nts, _nReceiversIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy( dataTemp_sigmaxx->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid)*_fdParamElastic->_nts, _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy( dataTemp_sigmazz->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, _nReceiversIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy( dataTemp_sigmaxz->getVals(), data->getVals()+(_nReceiversIrregXGrid+_nReceiversIrregZGrid+2*_nReceiversIrregCenterGrid)*_fdParamElastic->_nts, _nReceiversIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );

	/* Interpolate data to regular grid */
	_receiversXGrid->adjoint(false, dataRegDts_vx, dataTemp_vx);
	_receiversZGrid->adjoint(false, dataRegDts_vz, dataTemp_vz);
	_receiversCenterGrid->adjoint(false, dataRegDts_sigmaxx, dataTemp_sigmaxx);
	_receiversCenterGrid->adjoint(false, dataRegDts_sigmazz, dataTemp_sigmazz);
	_receiversXZGrid->adjoint(false, dataRegDts_sigmaxz, dataTemp_sigmaxz);

	/* Propagate */
	if (_saveWavefield == 0) {
		propShotsElasticAdjGpu(modelRegDtw_vx->getVals(),
													modelRegDtw_vz->getVals(),
													modelRegDtw_sigmaxx->getVals(),
													modelRegDtw_sigmazz->getVals(),
													modelRegDtw_sigmaxz->getVals(),
													dataRegDts_vx->getVals(),
													dataRegDts_vz->getVals(),
													dataRegDts_sigmaxx->getVals(),
													dataRegDts_sigmazz->getVals(),
													dataRegDts_sigmaxz->getVals(),
													_sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid,
													_sourcesPositionRegXGrid, _nSourcesRegXGrid,
													_sourcesPositionRegZGrid, _nSourcesRegZGrid,
													_sourcesPositionRegXZGrid, _nSourcesRegXZGrid,
													_receiversPositionRegCenterGrid, _nReceiversRegCenterGrid,
													_receiversPositionRegXGrid, _nReceiversRegXGrid,
													_receiversPositionRegZGrid, _nReceiversRegZGrid,
													_receiversPositionRegXZGrid, _nReceiversRegXZGrid,
													 _iGpu, _iGpuId, _fdParamElastic->_surfaceCondition);
	} else {
			propShotsElasticAdjGpuWavefield(modelRegDtw_vx->getVals(),
														modelRegDtw_vz->getVals(),
														modelRegDtw_sigmaxx->getVals(),
														modelRegDtw_sigmazz->getVals(),
														modelRegDtw_sigmaxz->getVals(),
														dataRegDts_vx->getVals(),
														dataRegDts_vz->getVals(),
														dataRegDts_sigmaxx->getVals(),
														dataRegDts_sigmazz->getVals(),
														dataRegDts_sigmaxz->getVals(),
														_sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid,
														_sourcesPositionRegXGrid, _nSourcesRegXGrid,
														_sourcesPositionRegZGrid, _nSourcesRegZGrid,
														_sourcesPositionRegXZGrid, _nSourcesRegXZGrid,
														_receiversPositionRegCenterGrid, _nReceiversRegCenterGrid,
														_receiversPositionRegXGrid, _nReceiversRegXGrid,
														_receiversPositionRegZGrid, _nReceiversRegZGrid,
														_receiversPositionRegXZGrid, _nReceiversRegXZGrid,
														_wavefield->getVals(),
														_iGpu, _iGpuId);
	}

	/* Interpolate to coarse time-sampling */
	_timeInterp->adjoint(false, modelRegDts_vx, modelRegDtw_vx);
	_timeInterp->adjoint(false, modelRegDts_vz, modelRegDtw_vz);
	_timeInterp->adjoint(false, modelRegDts_sigmaxx, modelRegDtw_sigmaxx);
	_timeInterp->adjoint(false, modelRegDts_sigmazz, modelRegDtw_sigmazz);
	_timeInterp->adjoint(false, modelRegDts_sigmaxz, modelRegDtw_sigmaxz);

	/*Scaling by the inverse of the space discretization*/
	double area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dz);
	modelRegDts_sigmaxx->scale(area_scale);
	modelRegDts_sigmazz->scale(area_scale);
	modelRegDts_vx->scale(area_scale);
	modelRegDts_vz->scale(area_scale);
	modelRegDts_sigmaxz->scale(area_scale);
	/* Scale model  */
	modelRegDts_sigmaxx->scale(2.0*_fdParamElastic->_dtw);
	modelRegDts_sigmazz->scale(2.0*_fdParamElastic->_dtw);
	#pragma omp parallel for collapse(2)
	for(int is = 0; is < _nSourcesRegXGrid; is++){ //loop over number of reg sources x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			(*modelRegDts_vx->_mat)[is][it] *= _fdParamElastic->_rhoxDtw[(_sourcesXGrid->getRegPosUnique())[is]];
		}
	}
	#pragma omp parallel for collapse(2)
	for(int is = 0; is < _nSourcesRegZGrid; is++){ //loop over number of reg sources z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			(*modelRegDts_vz->_mat)[is][it] *= _fdParamElastic->_rhozDtw[(_sourcesZGrid->getRegPosUnique())[is]];
		}
	}
	modelRegDts_sigmaxz->scale(2.0*_fdParamElastic->_dtw);

	/* Scale adjoint wavefield */
	if (_saveWavefield == 1){
		// scale wavefield
		#pragma omp parallel for collapse(3)
		for(int it=0; it < _fdParamElastic->_timeAxisCoarse.n; it++){
			for(int ix=0; ix < _fdParamElastic->_xAxis.n; ix++){
				for(int iz=0; iz < _fdParamElastic->_zAxis.n; iz++){
					(*_wavefield->_mat)[it][0][ix][iz]*=_fdParamElastic->_rhoxDtw[ix * _fdParamElastic->_zAxis.n + iz]*area_scale;
					(*_wavefield->_mat)[it][1][ix][iz]*=_fdParamElastic->_rhozDtw[ix * _fdParamElastic->_zAxis.n + iz]*area_scale;
					(*_wavefield->_mat)[it][2][ix][iz]*=_fdParamElastic->_dtw*area_scale;
					(*_wavefield->_mat)[it][3][ix][iz]*=_fdParamElastic->_dtw*area_scale;
					(*_wavefield->_mat)[it][4][ix][iz]*=_fdParamElastic->_dtw*area_scale;
				}
			}
		}

	}

	/* Interpolate to irregular grid */
	_sourcesXGrid->forward(true, modelRegDts_vx, modelTemp_vx);
	_sourcesZGrid->forward(true, modelRegDts_vz, modelTemp_vz);
	_sourcesCenterGrid->forward(true, modelRegDts_sigmaxx, modelTemp_sigmaxx);
	_sourcesCenterGrid->forward(true, modelRegDts_sigmazz,modelTemp_sigmazz);
	_sourcesXZGrid->forward(true, modelRegDts_sigmaxz, modelTemp_sigmaxz);

	/* Copy each source into one cube */
	std::memcpy(model->getVals(), modelTemp_vx->getVals(), _nSourcesIrregXGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy(model->getVals()+_nSourcesIrregXGrid*_fdParamElastic->_nts, modelTemp_vz->getVals(), _nSourcesIrregZGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy(model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid)*_fdParamElastic->_nts, modelTemp_sigmaxx->getVals(), _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy(model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, modelTemp_sigmazz->getVals(), _nSourcesIrregCenterGrid*_fdParamElastic->_nts*sizeof(double) );
	std::memcpy(model->getVals()+(_nSourcesIrregXGrid+_nSourcesIrregZGrid+2*_nSourcesIrregCenterGrid)*_fdParamElastic->_nts, modelTemp_sigmaxz->getVals(), _nSourcesIrregXZGrid*_fdParamElastic->_nts*sizeof(double) );
}
