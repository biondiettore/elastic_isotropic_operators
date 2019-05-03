#ifndef WAVE_EQUATION_ELASTIC_GPU_FUNCTIONS_H
#define WAVE_EQUATION_ELASTIC_GPU_FUNCTIONS_H 1

/*********************************** Initialization **************************************/
//bool getGpuInfo(int nGpu, int info, int deviceNumber);
void initWaveEquationElasticGpu(float dz, float dx, int nz, int nx, int nts, float dts, int minPad, int blockSize, int nGpu, int iGpuId, int iGpuAlloc);
void allocateWaveEquationElasticGpu(float *rhox, float *rhoz, float *lamb2Mu, float *lamb, float *muxz, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void deallocateWaveEquationElasticGpu();
void waveEquationElasticFwdGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void waveEquationElasticAdjGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
bool getGpuInfo(int nGpu, int info, int deviceNumber);
float getTotalGlobalMem(int nGpu, int info, int deviceNumber); //return GB of total global mem

#endif
