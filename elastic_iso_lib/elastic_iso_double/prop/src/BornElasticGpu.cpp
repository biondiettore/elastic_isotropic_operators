#include <vector>
#include <ctime>
#include "BornElasticGpu.h"
#include <cstring>
#include <stdexcept>


BornElasticGpu::BornElasticGpu(std::shared_ptr<fdParamElastic> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

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
	initBornGpu(_fdParamElastic->_dz, _fdParamElastic->_dx, _fdParamElastic->_nz, _fdParamElastic->_nx, _fdParamElastic->_nts, _fdParamElastic->_dts, _fdParamElastic->_sub, _fdParamElastic->_minPad, _fdParamElastic->_blockSize, _fdParamElastic->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	/// Alocate on GPUs
	allocateBornElasticGpu(_fdParamElastic->_rhoxDtw,
                         _fdParamElastic->_rhozDtw,
												 _fdParamElastic->_lamb2MuDtw,
												 _fdParamElastic->_lambDtw,
												 _fdParamElastic->_muxzDtw,
												 _iGpu, _iGpuId, _useStreams);
	setAllWavefields(0); // By default, do not record the scattered wavefields
}

bool BornElasticGpu::checkParfileConsistency(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{

	if (_fdParamElastic->checkParfileConsistencyTime(data, 1) != true) {return false;} // Check data time axis
	if (_fdParamElastic->checkParfileConsistencyTime(_sourcesSignals, 1) != true) {return false;} // Check source time axis
	if (_fdParamElastic->checkParfileConsistencySpace(model) != true) {return false;}; // Check model axis

	return true;
}

void BornElasticGpu::setAllWavefields(int wavefieldFlag){
	_srcWavefield = setWavefield(wavefieldFlag);
	_secWavefield = setWavefield(wavefieldFlag);
}

void BornElasticGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

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

	//Getting already staggered model perturbations
	double *drhox_in = model->getVals();
	double *drhoz_in = model->getVals()+_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dlame_in = model->getVals()+2*_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dmu_in   = model->getVals()+3*_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dmuxz_in = model->getVals()+4*_fdParamElastic->_nz*_fdParamElastic->_nx;

	// /* Propagate */
	if (_saveWavefield == 0) {
		BornShotsFwdGpu(_sourceRegDtw_vx->getVals(),
										_sourceRegDtw_vz->getVals(),
										_sourceRegDtw_sigmaxx->getVals(),
										_sourceRegDtw_sigmazz->getVals(),
										_sourceRegDtw_sigmaxz->getVals(),
										drhox_in,
										drhoz_in,
										dlame_in,
										dmu_in,
										dmuxz_in,
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
										_iGpu, _iGpuId, _fdParamElastic->_surfaceCondition, _useStreams);
	} else {
		throw std::logic_error( "Error! Born forward operator w/ wavefield saving not implemented yet!" );
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

void BornElasticGpu::adjoint(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) {
		model->scale(0.0);
	}

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

	//Getting model perturbations pointers
	double *drhox_in = model->getVals();
	double *drhoz_in = model->getVals()+_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dlame_in = model->getVals()+2*_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dmu_in   = model->getVals()+3*_fdParamElastic->_nz*_fdParamElastic->_nx;
	double *dmuxz_in = model->getVals()+4*_fdParamElastic->_nz*_fdParamElastic->_nx;

	// /* Propagate */
	if (_saveWavefield == 0) {
		BornShotsAdjGpu(_sourceRegDtw_vx->getVals(),
										_sourceRegDtw_vz->getVals(),
										_sourceRegDtw_sigmaxx->getVals(),
										_sourceRegDtw_sigmazz->getVals(),
										_sourceRegDtw_sigmaxz->getVals(),
										drhox_in,
										drhoz_in,
										dlame_in,
										dmu_in,
										dmuxz_in,
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
										_iGpu, _iGpuId, _fdParamElastic->_surfaceCondition, _useStreams);
	} else {
		throw std::logic_error( "Error! Born forward operator w/ wavefield saving not implemented yet!" );
	}
}
