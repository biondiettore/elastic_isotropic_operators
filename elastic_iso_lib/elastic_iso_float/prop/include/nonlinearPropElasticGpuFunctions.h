#ifndef NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_H
#define NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearElasticGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearElasticGpu(float *rhoxDtw, float *rhozDtw, float *lamb2MuDt, float *lambDtw, float *muxzDt,int iGpu, int iGpuId);
void deallocateNonlinearElasticGpu(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
void propShotsElasticFwdGpu(float *modelRegDtw_vx,
							float *modelRegDtw_vz,
							float *modelRegDtw_sigmaxx,
							float *modelRegDtw_sigmazz,
							float *modelRegDtw_sigmaxz,
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
							int iGpu, int iGpuId, int surfaceCondition);
void propShotsElasticFwdGpuWavefield(float *modelRegDtw_vx,
									 float *modelRegDtw_vz,
									 float *modelRegDtw_sigmaxx,
									 float *modelRegDtw_sigmazz,
									 float *modelRegDtw_sigmaxz,
									 float *dataRegDts_vx,
									 float *dataRegDts_vz,
									 float *dataRegDts_sigmaxx,
									 float *dataRegDts_sigmazz,
									 float *dataRegDts_sigmaxz,
									 int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid,
									 int *sourcesPositionRegXGrid, int nSourcesRegXGrid,
									 int *sourcesPositionRegZGrid, int nSourcesRegZGrid,
									 int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid,
									 int *receiversPositionRegCenterGrid, int  nReceiversRegCenterGrid,
									 int *receiversPositionRegXGrid, int nReceiversRegXGrid,
									 int *receiversPositionRegZGrid, int nReceiversRegZGrid,
									 int *receiversPositionRegXZGrid, int nReceiversRegXZGrid,
									 float* wavefield,
									 int iGpu, int iGpuId, int surfaceCondition);
void propShotsElasticFwdGpuWavefieldStreams(float *modelRegDtw_vx,
											float *modelRegDtw_vz,
											float *modelRegDtw_sigmaxx,
											float *modelRegDtw_sigmazz,
											float *modelRegDtw_sigmaxz,
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
											float* wavefield,
											int iGpu, int iGpuId, int surfaceCondition);

/*********************************** Nonlinear ADJ **************************************/
void propShotsElasticAdjGpu(float *modelRegDtw_vx,
							float *modelRegDtw_vz,
							float *modelRegDtw_sigmaxx,
							float *modelRegDtw_sigmazz,
							float *modelRegDtw_sigmaxz,
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
							int iGpu, int iGpuId, int surfaceCondition);
void propShotsElasticAdjGpuWavefield(float *modelRegDtw_vx,
														float *modelRegDtw_vz,
														float *modelRegDtw_sigmaxx,
														float *modelRegDtw_sigmazz,
														float *modelRegDtw_sigmaxz,
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
														float* wavefield,
														int iGpu, int iGpuId);
void propShotsElasticAdjGpuWavefieldStreams(float *modelRegDtw_vx,
														float *modelRegDtw_vz,
														float *modelRegDtw_sigmaxx,
														float *modelRegDtw_sigmazz,
														float *modelRegDtw_sigmaxz,
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
														float* wavefield,
														int iGpu, int iGpuId);

void get_dev_zCoeff(float *hp);
void get_dev_xCoeff(float *hp);

#endif
