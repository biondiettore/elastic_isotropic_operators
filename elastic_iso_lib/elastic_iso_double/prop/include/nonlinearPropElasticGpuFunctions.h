#ifndef NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_H
#define NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearElasticGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearElasticGpu(double *rhoxDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDt,int iGpu, int iGpuId);
void deallocateNonlinearElasticGpu(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD/ADJ **************************************/
// void switchPointers(int iGpu);
void propShotsElasticAdjGpu(double *modelRegDtw_vx,
														double *modelRegDtw_vz,
														double *modelRegDtw_sigmaxx,
														double *modelRegDtw_sigmazz,
														double *modelRegDtw_sigmaxz,
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
														int iGpu, int iGpuId, int _surfaceCondition);
void propShotsElasticFwdGpu(double *modelRegDtw_vx,
														double *modelRegDtw_vz,
														double *modelRegDtw_sigmaxx,
														double *modelRegDtw_sigmazz,
														double *modelRegDtw_sigmaxz,
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
														 int iGpu, int iGpuId, int _surfaceCondition);
void propShotsElasticFwdGpuWavefield(double *modelRegDtw_vx,
														double *modelRegDtw_vz,
														double *modelRegDtw_sigmaxx,
														double *modelRegDtw_sigmazz,
														double *modelRegDtw_sigmaxz,
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
														double* wavefield,
														int iGpu, int iGpuId);
void propShotsElasticAdjGpuWavefield(double *modelRegDtw_vx,
														double *modelRegDtw_vz,
														double *modelRegDtw_sigmaxx,
														double *modelRegDtw_sigmazz,
														double *modelRegDtw_sigmaxz,
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
														double* wavefield,
														int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/
/* Adjoint propagation -- Data recorded at fine scale */
// void propShotsAdjGpu(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);
// void propShotsAdjGpuWavefield(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

void get_dev_zCoeff(double *hp);
void get_dev_xCoeff(double *hp);

#endif
