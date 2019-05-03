#include <cstring>
#include <iostream>
#include "waveEquationElasticGpuFunctions.h"
#include "varDeclareWaveEquation.h"
#include "kernelsGpuWaveEquationElastic.cu"
#include "cudaErrors.cu"
#include <cstring>

void initWaveEquationElasticGpu(double dz, double dx, int nz, int nx, int nts, double dts, int minPad, int blockSize, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
  cudaSetDevice(iGpuId);

  // Host variables
	host_nz = nz;
	host_nx = nx;
	host_dz = dz;
	host_dx = dx;
	host_nts = nts;
	host_dts = dts;

	// /**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		// Array of pointers to wavefields
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];

		// Scaled earth parameters velocity
		dev_rhox = new double*[nGpu];
		dev_rhoz = new double*[nGpu];
		dev_lamb2Mu = new double*[nGpu];
		dev_lamb = new double*[nGpu];
		dev_muxz = new double*[nGpu];
  }

  /**************************** COMPUTE DERIVATIVE COEFFICIENTS ************************/
  double zCoeff[COEFF_SIZE];
	double xCoeff[COEFF_SIZE];

	zCoeff[0] = 1.196289062541883 / dz;
	zCoeff[1] = -0.079752604188901 / dz;
	zCoeff[2] = 0.009570312506634 / dz;
	zCoeff[3] = -6.975446437140719e-04 / dz;

	xCoeff[0] = 1.196289062541883 / dx;
	xCoeff[1] = -0.079752604188901 / dx;
	xCoeff[2] = 0.009570312506634 / dx;
	xCoeff[3] = -6.975446437140719e-04 / dx;

  /************************** COMPUTE COSINE DAMPING COEFFICIENTS **********************/
  // Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy derivative coefficients to device
	cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));

	// // Cosine damping parameters
	// cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
	// cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
	// cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_nw, &host_nw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	// cuda_call(cudaMemcpyToSymbol(dev_dx, &host_dx, sizeof(float), 0, cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpyToSymbol(dev_dz, &host_dz, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_dts, &host_dts, sizeof(double), 0, cudaMemcpyHostToDevice));
}

void allocateWaveEquationElasticGpu(double *rhox, double *rhoz, double *lamb2Mu, double *lamb, double *muxz, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);

	// Allocate scaled elastic parameters to device
	cuda_call(cudaMalloc((void**) &dev_rhox[iGpu], host_nz*host_nx*sizeof(double)));
  cuda_call(cudaMalloc((void**) &dev_rhoz[iGpu], host_nz*host_nx*sizeof(double)));
  cuda_call(cudaMalloc((void**) &dev_lamb2Mu[iGpu], host_nz*host_nx*sizeof(double)));
  cuda_call(cudaMalloc((void**) &dev_lamb[iGpu], host_nz*host_nx*sizeof(double)));
  cuda_call(cudaMalloc((void**) &dev_muxz[iGpu], host_nz*host_nx*sizeof(double)));

  // Copy scaled elastic parameters to device
	cuda_call(cudaMemcpy(dev_rhox[iGpu], rhox, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
  cuda_call(cudaMemcpy(dev_rhoz[iGpu], rhoz, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
  cuda_call(cudaMemcpy(dev_lamb2Mu[iGpu], lamb2Mu, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
  cuda_call(cudaMemcpy(dev_lamb[iGpu], lamb, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
  cuda_call(cudaMemcpy(dev_muxz[iGpu], muxz, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate wavefields on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double)));
  cuda_call(cudaMalloc((void**) &dev_p1[iGpu], (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double)));

}

void waveEquationElasticFwdGpu(double *model,double *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);

	//copy model to gpu
	cuda_call(cudaMemcpy(dev_p1[iGpu], model+firstTimeSample*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyHostToDevice));

	//copy data to gpu
	cuda_call(cudaMemcpy(dev_p0[iGpu], data+firstTimeSample*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyHostToDevice));

	//call fwd gpu kernel
	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockz = ((lastTimeSample-firstTimeSample+1)+BLOCK_SIZE) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky, nblockz);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

	kernel_exec(ker_we_fwd<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_rhox[iGpu], dev_rhoz[iGpu], dev_lamb2Mu[iGpu], dev_lamb[iGpu], dev_muxz[iGpu],firstTimeSample,lastTimeSample));

	//first, last, and other gpus have different copies
	if(firstTimeSample==0 && lastTimeSample==host_nts-1) {
		//include first and last sample in block
		cuda_call(cudaMemcpy(data, dev_p0[iGpu], host_nts*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else if(firstTimeSample==0){
		//exclude last sample | include first sample in block
		cuda_call(cudaMemcpy(data, dev_p0[iGpu], (lastTimeSample-firstTimeSample)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else if(lastTimeSample==host_nts-1){
		//exclude first sample | include last sample in block
		cuda_call(cudaMemcpy(data+(firstTimeSample+1)*host_nz*host_nx*host_nw, dev_p0[iGpu]+1*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else{
		//exclude first and last sample in block
		cuda_call(cudaMemcpy(data+(firstTimeSample+1)*host_nz*host_nx*host_nw, dev_p0[iGpu]+1*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample-1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
}
void waveEquationElasticAdjGpu(double *model,double *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);

	//copy data to gpu
	cuda_call(cudaMemcpy(dev_p1[iGpu], data+firstTimeSample*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyHostToDevice));

	//copy model to gpu
	cuda_call(cudaMemcpy(dev_p0[iGpu], model+firstTimeSample*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyHostToDevice));

	//call adj gpu kernel
	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockz = ((lastTimeSample-firstTimeSample+1)+BLOCK_SIZE) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky, nblockz);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

	kernel_exec(ker_we_adj<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_rhox[iGpu], dev_rhoz[iGpu], dev_lamb2Mu[iGpu], dev_lamb[iGpu], dev_muxz[iGpu],firstTimeSample,lastTimeSample));

	//copy model from gpu
	//first, last, and other gpus have different copies
	if(firstTimeSample==0 && lastTimeSample==host_nts-1) {
		//include first and last sample in block
		cuda_call(cudaMemcpy(model, dev_p0[iGpu], host_nts*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else if(firstTimeSample==0){
		//exclude last sample | include first sample in block
		cuda_call(cudaMemcpy(model, dev_p0[iGpu], (lastTimeSample-firstTimeSample)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else if(lastTimeSample==host_nts-1){
		//exclude first sample | include last sample in block
		cuda_call(cudaMemcpy(model+(firstTimeSample+1)*host_nz*host_nx*host_nw, dev_p0[iGpu]+1*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
	else{
		//exclude first and last sample in block
		cuda_call(cudaMemcpy(model+(firstTimeSample+1)*host_nz*host_nx*host_nw, dev_p0[iGpu]+1*host_nz*host_nx*host_nw, (lastTimeSample-firstTimeSample-1)*host_nz*host_nx*host_nw*sizeof(double), cudaMemcpyDeviceToHost));
	}
}

// void deallocateWaveEquationElasticGpu(){
//   // Deallocate scaled elastic params
//   cuda_call(cudaFree(dev_rhox));
//   cuda_call(cudaFree(dev_rhoz));
//   cuda_call(cudaFree(dev_lamb2Muw));
//   cuda_call(cudaFree(dev_lamb));
//   cuda_call(cudaFree(dev_muxzw));
//
//   // Deallocate wavefields
// 	cuda_call(cudaFree(dev_p0));
//   cuda_call(cudaFree(dev_p1));


// check gpu info
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << "************************ INFO FOR GPU# " << deviceNumberInfo << " *************************" << std::endl;
		std::cout << "*******************************************************************" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << nGpu << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;

		// Driver version
		std::cout << "Cuda driver version: " << cudaDriverGetVersion(&driver) << std::endl; // Driver

		// Get properties
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop,deviceNumberInfo);

		// Display
		std::cout << "Name: " << dprop.name << std::endl;
		std::cout << "Total global memory: " << dprop.totalGlobalMem/(1024*1024*1024) << " [GB] " << std::endl;
		std::cout << "Shared memory per block: " << dprop.sharedMemPerBlock/1024 << " [kB]" << std::endl;
		std::cout << "Number of register per block: " << dprop.regsPerBlock << std::endl;
		std::cout << "Warp size: " << dprop.warpSize << " [threads]" << std::endl;
		std::cout << "Maximum pitch allowed for memory copies in bytes: " << dprop.memPitch/(1024*1024*1024) << " [GB]" << std::endl;
		std::cout << "Maximum threads per block: " << dprop.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum block dimensions: " << "(" << dprop.maxThreadsDim[0] << ", " << dprop.maxThreadsDim[1] << ", " << dprop.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum grid dimensions: " << "(" << dprop.maxGridSize[0] << ", " << dprop.maxGridSize[1] << ", " << dprop.maxGridSize[2] << ")" << std::endl;
		std::cout << "Total constant memory: " << dprop.totalConstMem/1024 << " [kB]" << std::endl;
		std::cout << "Number of streaming multiprocessors on device: " << dprop.multiProcessorCount << std::endl;
		if (dprop.deviceOverlap == 1) {std::cout << "Device can simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.deviceOverlap != 1) {std::cout << "Device cannot simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.canMapHostMemory == 1) { std::cout << "Device can map host memory" << std::endl; }
		if (dprop.canMapHostMemory != 1) { std::cout << "Device cannot map host memory" << std::endl; }
		if (dprop.concurrentKernels == 1) {std::cout << "Device can support concurrent kernel" << std::endl;}
		if (dprop.concurrentKernels != 1) {std::cout << "Device cannot support concurrent kernel execution" << std::endl;}

		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

  	if (nGpu<nDevice+1) {return true;}
  	else {std::cout << "Number of requested GPU greater than available GPUs" << std::endl; return false;}
}

float getTotalGlobalMem(int nGpu, int info, int deviceNumberInfo){
	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);
	// Get properties
	cudaDeviceProp dprop;
	cudaGetDeviceProperties(&dprop,deviceNumberInfo);
	return dprop.totalGlobalMem/(1024*1024*1024);
}
