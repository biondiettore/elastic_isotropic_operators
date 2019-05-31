#ifndef BORN_ELASTIC_GPU_FUNCTIONS_H
#define BORN_ELASTIC_GPU_FUNCTIONS_H 1
#include <vector>

/* Parameter settings */
/*********************************** Initialization **************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornElasticGpu(double *rhoxDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDt, int iGpu, int iGpuId, int useStreams);
void deallocateBornElasticGpu(int iGpu, int iGpuId, int useStreams);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu(double *sourceRegDtw_vx,
										double *sourceRegDtw_vz,
										double *sourceRegDtw_sigmaxx,
										double *sourceRegDtw_sigmazz,
										double *sourceRegDtw_sigmaxz,
										double *drhox,
										double *drhoz,
										double *dlame,
										double *dmu,
										double *dmuxz,
										double *dataRegDts_vx,
										double *dataRegDts_vz,
										double *dataRegDts_sigmaxx,
										double *dataRegDts_sigmazz,
										double *dataRegDts_sigmaxz,
										int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid,
										int *sourcesPositionRegXGrid, int nSourcesRegXGrid,
										int *sourcesPositionRegZGrid, int nSourcesRegZGrid,
										int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid,
										int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid,
										int *receiversPositionRegXGrid, int nReceiversRegXGrid,
										int *receiversPositionRegZGrid, int nReceiversRegZGrid,
										int *receiversPositionRegXZGrid, int nReceiversRegXZGrid,
										int iGpu, int iGpuId, int surfaceCondition, int useStreams);
// void BornShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu(double *sourceRegDtw_vx,
										double *sourceRegDtw_vz,
										double *sourceRegDtw_sigmaxx,
										double *sourceRegDtw_sigmazz,
										double *sourceRegDtw_sigmaxz,
										double *drhox,
										double *drhoz,
										double *dlame,
										double *dmu,
										double *dmuxz,
										double *dataRegDts_vx,
										double *dataRegDts_vz,
										double *dataRegDts_sigmaxx,
										double *dataRegDts_sigmazz,
										double *dataRegDts_sigmaxz,
										int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid,
										int *sourcesPositionRegXGrid, int nSourcesRegXGrid,
										int *sourcesPositionRegZGrid, int nSourcesRegZGrid,
										int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid,
										int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid,
										int *receiversPositionRegXGrid, int nReceiversRegXGrid,
										int *receiversPositionRegZGrid, int nReceiversRegZGrid,
										int *receiversPositionRegXZGrid, int nReceiversRegXZGrid,
										int iGpu, int iGpuId, int surfaceCondition, int useStreams);
// void BornShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

#endif
