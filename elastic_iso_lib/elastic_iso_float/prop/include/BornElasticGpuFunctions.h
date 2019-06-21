#ifndef BORN_ELASTIC_GPU_FUNCTIONS_H
#define BORN_ELASTIC_GPU_FUNCTIONS_H 1
#include <vector>

/* Parameter settings */
/*********************************** Initialization **************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornElasticGpu(float *rhoxDtw, float *rhozDtw, float *lamb2MuDt, float *lambDtw, float *muxzDt, int iGpu, int iGpuId, int useStreams);
void deallocateBornElasticGpu(int iGpu, int iGpuId, int useStreams);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu(float *sourceRegDtw_vx,
										float *sourceRegDtw_vz,
										float *sourceRegDtw_sigmaxx,
										float *sourceRegDtw_sigmazz,
										float *sourceRegDtw_sigmaxz,
										float *drhox,
										float *drhoz,
										float *dlame,
										float *dmu,
										float *dmuxz,
										float *dataRegDts_vx,
										float *dataRegDts_vz,
										float *dataRegDts_sigmaxx,
										float *dataRegDts_sigmazz,
										float *dataRegDts_sigmaxz,
										int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid,
										int *sourcesPositionRegXGrid, int nSourcesRegXGrid,
										int *sourcesPositionRegZGrid, int nSourcesRegZGrid,
										int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid,
										int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid,
										int *receiversPositionRegXGrid, int nReceiversRegXGrid,
										int *receiversPositionRegZGrid, int nReceiversRegZGrid,
										int *receiversPositionRegXZGrid, int nReceiversRegXZGrid,
										int iGpu, int iGpuId, int surfaceCondition, int useStreams);
// void BornShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu(float *sourceRegDtw_vx,
										float *sourceRegDtw_vz,
										float *sourceRegDtw_sigmaxx,
										float *sourceRegDtw_sigmazz,
										float *sourceRegDtw_sigmaxz,
										float *drhox,
										float *drhoz,
										float *dlame,
										float *dmu,
										float *dmuxz,
										float *dataRegDts_vx,
										float *dataRegDts_vz,
										float *dataRegDts_sigmaxx,
										float *dataRegDts_sigmazz,
										float *dataRegDts_sigmaxz,
										int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid,
										int *sourcesPositionRegXGrid, int nSourcesRegXGrid,
										int *sourcesPositionRegZGrid, int nSourcesRegZGrid,
										int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid,
										int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid,
										int *receiversPositionRegXGrid, int nReceiversRegXGrid,
										int *receiversPositionRegZGrid, int nReceiversRegZGrid,
										int *receiversPositionRegXZGrid, int nReceiversRegXZGrid,
										int iGpu, int iGpuId, int surfaceCondition, int useStreams);
// void BornShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);

#endif
