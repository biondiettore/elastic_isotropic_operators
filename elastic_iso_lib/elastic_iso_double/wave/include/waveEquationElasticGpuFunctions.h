#ifndef WAVE_EQUATION_ELASTIC_GPU_FUNCTIONS_H
#define WAVE_EQUATION_ELASTIC_GPU_FUNCTIONS_H 1

/*********************************** Initialization **************************************/
//bool getGpuInfo(int nGpu, int info, int deviceNumber);
void initWaveEquationElasticGpu(double dz, double dx, int nz, int nx, int nts, double dts, int minPad, int blockSize, int nGpu, int iGpuId, int iGpuAlloc);
void allocateWaveEquationElasticGpu(double *rhox, double *rhoz, double *lamb2Mu, double *lamb, double *muxz, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void deallocateWaveEquationElasticGpu();
void waveEquationElasticFwdGpu(double *model,double *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void waveEquationElasticAdjGpu(double *model,double *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
bool getGpuInfo(int nGpu, int info, int deviceNumber);
float getTotalGlobalMem(int nGpu, int info, int deviceNumber); //return GB of total global mem

#endif
