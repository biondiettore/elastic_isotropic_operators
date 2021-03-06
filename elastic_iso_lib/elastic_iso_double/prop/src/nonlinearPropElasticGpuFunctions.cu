#include <cstring>
#include <iostream>
#include "nonlinearPropElasticGpuFunctions.h"
#include "kernelsGpuElastic.cu"
#include "cudaErrors.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo){

		int nDevice, driver;
		cudaGetDeviceCount(&nDevice);

		if (info == 1){

				std::cout << " " << std::endl;
				std::cout << "-------------------------------------------------------------------" << std::endl;
				std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
				std::cout << "-------------------------------------------------------------------" << std::endl;

				// Number of devices
				std::cout << "Number of requested GPUs: " << gpuList.size() << std::endl;
				std::cout << "Number of available GPUs: " << nDevice << std::endl;
				std::cout << "Id of requested GPUs: ";
				for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
					if (iGpu<gpuList.size()-1){std::cout << gpuList[iGpu] << ", ";}
					else{ std::cout << gpuList[iGpu] << std::endl;}
				}

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

		// Check that the number of requested GPU is less or equal to the total number of available GPUs
		if (gpuList.size()>nDevice) {
			std::cout << "**** ERROR [getGpuInfo]: Number of requested GPU greater than available GPUs ****" << std::endl;
			return false;
		}

		// Check that the GPU numbers in the list are between 0 and nGpu-1
		for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
			if (gpuList[iGpu]<0 || gpuList[iGpu]>nDevice-1){
				std::cout << "**** ERROR [getGpuInfo]: One of the element of the GPU Id list is not a valid GPU Id number ****" << std::endl;
				return false;
			}
		}

		return true;
}

void initNonlinearElasticGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

		// Set GPU
		cudaSetDevice(iGpuId);

		// Host variables
		host_nz = nz;
		host_nx = nx;
		host_nts = nts;
		host_sub = sub;
		host_ntw = (nts - 1) * sub + 1;

		/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
		// Only one GPU will perform the following
		if (iGpuId == iGpuAlloc) {

				// Time slices for FD stepping for each wavefield
				dev_p0_vx       = new double*[nGpu];
				dev_p0_vz       = new double*[nGpu];
				dev_p0_sigmaxx  = new double*[nGpu];
				dev_p0_sigmazz  = new double*[nGpu];
				dev_p0_sigmaxz  = new double*[nGpu];

				dev_p1_vx       = new double*[nGpu];
				dev_p1_vz       = new double*[nGpu];
				dev_p1_sigmaxx  = new double*[nGpu];
				dev_p1_sigmazz  = new double*[nGpu];
				dev_p1_sigmaxz  = new double*[nGpu];

				dev_temp1    = new double*[nGpu];

				// Data and model
				dev_modelRegDtw_vx = new double*[nGpu];
				dev_modelRegDtw_vz = new double*[nGpu];
				dev_modelRegDtw_sigmaxx = new double*[nGpu];
				dev_modelRegDtw_sigmazz = new double*[nGpu];
				dev_modelRegDtw_sigmaxz = new double*[nGpu];
				dev_dataRegDts_vx = new double*[nGpu];
				dev_dataRegDts_vz = new double*[nGpu];
				dev_dataRegDts_sigmaxx = new double*[nGpu];
				dev_dataRegDts_sigmazz = new double*[nGpu];
				dev_dataRegDts_sigmaxz = new double*[nGpu];

				// Source and receivers
				dev_sourcesPositionRegCenterGrid = new int*[nGpu];
				dev_sourcesPositionRegXGrid = new int*[nGpu];
				dev_sourcesPositionRegZGrid = new int*[nGpu];
				dev_sourcesPositionRegXZGrid = new int*[nGpu];
				dev_receiversPositionRegCenterGrid = new int*[nGpu];
				dev_receiversPositionRegXGrid = new int*[nGpu];
				dev_receiversPositionRegZGrid = new int*[nGpu];
				dev_receiversPositionRegXZGrid = new int*[nGpu];

				// Scaled velocity
				dev_rhoxDtw = new double*[nGpu]; // Precomputed scaling dtw / rho_x
				dev_rhozDtw = new double*[nGpu]; // Precomputed scaling dtw / rho_z
				dev_lamb2MuDtw = new double*[nGpu]; // Precomputed scaling (lambda + 2*mu) * dtw
				dev_lambDtw = new double*[nGpu]; // Precomputed scaling lambda * dtw
				dev_muxzDtw = new double*[nGpu]; // Precomputed scaling mu_xz * dtw

				// Streams for saving the wavefield and time slices
				compStream = new cudaStream_t[nGpu];
				transferStream = new cudaStream_t[nGpu];
				pin_wavefieldSlice = new double*[nGpu];
				dev_wavefieldDts_left = new double*[nGpu];
				dev_wavefieldDts_right = new double*[nGpu];
				dev_pStream = new double*[nGpu];

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


		/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
		// Time interpolation filter length/half length
		int hInterpFilter = host_sub + 1;
		int nInterpFilter = 2 * hInterpFilter;

		// Check the subsampling coefficient is smaller than the maximum allowed
		if (sub>=SUB_MAX){
				std::cout << "**** ERROR: Subsampling parameter too high ****" << std::endl;
				throw std::runtime_error("");
		}

		// Allocate and fill interpolation filter
		double interpFilter[nInterpFilter];
		for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
				interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
				interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
				interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
				interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		}

		/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
		if (minPad>=PAD_MAX){
				std::cout << "**** ERROR: Padding value is too high ****" << std::endl;
				throw std::runtime_error("");
		}
		double cosDampingCoeff[minPad];

		// Cosine padding
		for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
				double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
				arg = alphaCos + (1.0-alphaCos) * cos(arg);
				cosDampingCoeff[iFilter-FAT] = arg;
		}

		// Check that the block size is consistent between parfile and "varDeclare.h"
		if (blockSize != BLOCK_SIZE) {
				std::cout << "**** ERROR: Block size for time stepper is not consistent with parfile ****" << std::endl;
				throw std::runtime_error("");
		}

		/**************************** COPY TO CONSTANT MEMORY *******************************/
		// Finite-difference coefficients
		cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy derivative coefficients to device
		cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));

		// Time interpolation filter
		cuda_call(cudaMemcpyToSymbol(dev_nInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
		cuda_call(cudaMemcpyToSymbol(dev_hInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
		cuda_call(cudaMemcpyToSymbol(dev_interpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

		// Cosine damping parameters
		cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
		cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
		cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

		// FD parameters
		cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
		cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
		cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}
void allocateNonlinearElasticGpu(double *rhoxDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDt,int iGpu, int iGpuId){

		// Get GPU number
		cudaSetDevice(iGpuId);

		// Allocate scaled elastic parameters to device
		cuda_call(cudaMalloc((void**) &dev_rhoxDtw[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_rhozDtw[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_lamb2MuDtw[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_lambDtw[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_muxzDtw[iGpu], host_nz*host_nx*sizeof(double)));

		// Copy scaled elastic parameters to device
		cuda_call(cudaMemcpy(dev_rhoxDtw[iGpu], rhoxDtw, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_rhozDtw[iGpu], rhozDtw, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_lamb2MuDtw[iGpu], lamb2MuDt, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_lambDtw[iGpu], lambDtw, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_muxzDtw[iGpu], muxzDt, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));

		// Allocate wavefield time slices on device (for the stepper)
		cuda_call(cudaMalloc((void**) &dev_p0_vx[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_vz[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmaxx[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmazz[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmaxz[iGpu], host_nz*host_nx*sizeof(double)));

		cuda_call(cudaMalloc((void**) &dev_p1_vx[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_vz[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmaxx[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmazz[iGpu], host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmaxz[iGpu], host_nz*host_nx*sizeof(double)));

}
void deallocateNonlinearElasticGpu(int iGpu, int iGpuId){
		cudaSetDevice(iGpuId); // Set device number on GPU cluster

		// Deallocate scaled elastic params
		cuda_call(cudaFree(dev_rhoxDtw[iGpu]));
		cuda_call(cudaFree(dev_rhozDtw[iGpu]));
		cuda_call(cudaFree(dev_lamb2MuDtw[iGpu]));
		cuda_call(cudaFree(dev_lambDtw[iGpu]));
		cuda_call(cudaFree(dev_muxzDtw[iGpu]));

		// Deallocate wavefields
		cuda_call(cudaFree(dev_p0_vx[iGpu]));
		cuda_call(cudaFree(dev_p0_vz[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_p1_vx[iGpu]));
		cuda_call(cudaFree(dev_p1_vz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxz[iGpu]));
}
void srcAllocateAndCopyToGpu(int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int iGpu){
		// Sources geometry
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegCenterGrid[iGpu], sourcesPositionRegCenterGrid, nSourcesRegCenterGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegXGrid[iGpu], sourcesPositionRegXGrid, nSourcesRegXGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegZGrid[iGpu], sourcesPositionRegZGrid, nSourcesRegZGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegXZGrid[iGpu], sourcesPositionRegXZGrid, nSourcesRegXZGrid*sizeof(int), cudaMemcpyHostToDevice));
}
void recAllocateAndCopyToGpu(int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu){
		// Receivers geometry
		cuda_call(cudaMemcpyToSymbol(dev_nReceiversRegCenterGrid, &nReceiversRegCenterGrid, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegCenterGrid[iGpu], receiversPositionRegCenterGrid, nReceiversRegCenterGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_nReceiversRegXGrid, &nReceiversRegXGrid, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegXGrid[iGpu], receiversPositionRegXGrid, nReceiversRegXGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_nReceiversRegZGrid, &nReceiversRegZGrid, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegZGrid[iGpu], receiversPositionRegZGrid, nReceiversRegZGrid*sizeof(int), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_nReceiversRegXZGrid, &nReceiversRegXZGrid, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid*sizeof(int)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegXZGrid[iGpu], receiversPositionRegXZGrid, nReceiversRegXZGrid*sizeof(int), cudaMemcpyHostToDevice));
}
void srcRecAllocateAndCopyToGpu(int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu){

		srcAllocateAndCopyToGpu(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, iGpu);
		recAllocateAndCopyToGpu(receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu);
}

//allocate model on device
void modelAllocateGpu(int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int iGpu){
		cuda_call(cudaMalloc((void**) &dev_modelRegDtw_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double))); // Allocate input on device
		cuda_call(cudaMalloc((void**) &dev_modelRegDtw_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double))); // Allocate input on device
		cuda_call(cudaMalloc((void**) &dev_modelRegDtw_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double))); // Allocate input on device
		cuda_call(cudaMalloc((void**) &dev_modelRegDtw_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double))); // Allocate input on device
		cuda_call(cudaMalloc((void**) &dev_modelRegDtw_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double))); // Allocate input on device
}

//copy model from host to device
void modelCopyToGpu(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int iGpu){
		cuda_call(cudaMemcpy(dev_modelRegDtw_vx[iGpu], modelRegDtw_vx, nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device
		cuda_call(cudaMemcpy(dev_modelRegDtw_vz[iGpu], modelRegDtw_vz, nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device
		cuda_call(cudaMemcpy(dev_modelRegDtw_sigmaxx[iGpu], modelRegDtw_sigmaxx, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device
		cuda_call(cudaMemcpy(dev_modelRegDtw_sigmazz[iGpu], modelRegDtw_sigmazz, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device
		cuda_call(cudaMemcpy(dev_modelRegDtw_sigmaxz[iGpu], modelRegDtw_sigmaxz, nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device
}
//initialize model values to
void modelInitializeOnGpu(int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int iGpu){
		cuda_call(cudaMemset(dev_modelRegDtw_vx[iGpu], 0, nSourcesRegXGrid*host_nts*sizeof(double))); // Copy input signals on device
		cuda_call(cudaMemset(dev_modelRegDtw_vz[iGpu], 0, nSourcesRegZGrid*host_nts*sizeof(double))); // Copy input signals on device
		cuda_call(cudaMemset(dev_modelRegDtw_sigmaxx[iGpu], 0, nSourcesRegCenterGrid*host_nts*sizeof(double))); // Copy input signals on device
		cuda_call(cudaMemset(dev_modelRegDtw_sigmazz[iGpu], 0, nSourcesRegCenterGrid*host_nts*sizeof(double))); // Copy input signals on device
		cuda_call(cudaMemset(dev_modelRegDtw_sigmaxz[iGpu], 0, nSourcesRegXZGrid*host_nts*sizeof(double))); // Copy input signals on device
}
//allocate model on device
void dataAllocateGpu(int nReceiversRegCenterGrid, int nReceiversRegXGrid, int nReceiversRegZGrid, int nReceiversRegXZGrid, int iGpu){
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double))); // Allocate output on device
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double))); // Allocate output on device
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double))); // Allocate output on device
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double))); // Allocate output on device
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double))); // Allocate output on device
}
void dataCopyToGpu(double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int nReceiversRegCenterGrid, int nReceiversRegXGrid, int nReceiversRegZGrid, int nReceiversRegXZGrid, int iGpu){
		cuda_call(cudaMemcpy(dev_dataRegDts_vx[iGpu], dataRegDts_vx, nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
		cuda_call(cudaMemcpy(dev_dataRegDts_vz[iGpu], dataRegDts_vz, nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxx[iGpu], dataRegDts_sigmaxx, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmazz[iGpu], dataRegDts_sigmazz, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxz[iGpu], dataRegDts_sigmaxz, nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
}
void dataInitializeOnGpu(int nReceiversRegCenterGrid, int nReceiversRegXGrid, int nReceiversRegZGrid, int nReceiversRegXZGrid, int iGpu){
		cuda_call(cudaMemset(dev_dataRegDts_vx[iGpu], 0, nReceiversRegXGrid*host_nts*sizeof(double))); // Initialize output on device
		cuda_call(cudaMemset(dev_dataRegDts_vz[iGpu], 0, nReceiversRegZGrid*host_nts*sizeof(double))); // Initialize output on device
		cuda_call(cudaMemset(dev_dataRegDts_sigmaxx[iGpu], 0, nReceiversRegCenterGrid*host_nts*sizeof(double))); // Initialize output on device
		cuda_call(cudaMemset(dev_dataRegDts_sigmazz[iGpu], 0, nReceiversRegCenterGrid*host_nts*sizeof(double))); // Initialize output on device
		cuda_call(cudaMemset(dev_dataRegDts_sigmaxz[iGpu], 0, nReceiversRegXZGrid*host_nts*sizeof(double))); // Initialize output on device
}
void wavefieldInitializeOnGpu(int iGpu){
		// Time slices
		cuda_call(cudaMemset(dev_p0_vx[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_vz[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmaxx[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmazz[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmaxz[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_vx[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_vz[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmaxx[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmazz[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmaxz[iGpu], 0, host_nz*host_nx*sizeof(double)));
}
//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and copy model (arrays for sources for each wavefield) to device
//             c) allocate and copy data (recevier recordings arrays) to device
//             d) allocate and copy wavefield time slices to gpu
void setupFwdGpu(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

	//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu);

		// Model - wavelets for each wavefield component. Allocate and copy to gpu
		modelAllocateGpu(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);
		modelCopyToGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);
		dataInitializeOnGpu(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);

		//initailize wavefield slices to zero
		wavefieldInitializeOnGpu(iGpu);

}

//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
//             c) allocate and copy data (recevier recordings arrays) to device
//             d) allocate and copy wavefield time slices to gpu
void setupAdjGpu(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

	//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu);

		// Model - wavelets for each wavefield component. Allocate and initilaize to 0 on gpu
		modelAllocateGpu(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);
		modelInitializeOnGpu(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);
		dataCopyToGpu(dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);

		//initailize wavefield slices to zero
		wavefieldInitializeOnGpu(iGpu);
}

void launchFwdStepKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(ker_step_fwd<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}

void launchFwdStepKernelsStreams(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_stream_exec(ker_step_fwd<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}

void launchFwdStepFreeSurfaceKernels(int nblocksSurface, int threadsInBlockSurface, dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(ker_step_fwd_surface_top<<<nblocksSurface, threadsInBlockSurface>>>(dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu]));
		kernel_exec(ker_step_fwd_surface_body<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}
void launchFwdStepFreeSurfaceKernelsStreams(int nblocksSurface, int threadsInBlockSurface, dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_stream_exec(ker_step_fwd_surface_top<<<nblocksSurface, threadsInBlockSurface, 0, compStream[iGpu]>>>(dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu]));
		kernel_stream_exec(ker_step_fwd_surface_body<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}
void launchFwdInjectSourceKernels(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouZGrid, int nblockSouXZGrid, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int its, int it2, int iGpu){

		kernel_exec(ker_inject_interp_source_centerGrid<<<nblockSouCenterGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDtw_sigmaxx[iGpu], dev_modelRegDtw_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		kernel_exec(ker_inject_interp_source_stagGrid<<<nblockSouXGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDtw_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		kernel_exec(ker_inject_interp_source_stagGrid<<<nblockSouZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDtw_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		kernel_exec(ker_inject_interp_source_stagGrid<<<nblockSouXZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDtw_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));

}
void launchFwdInjectSourceKernelsStreams(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouZGrid, int nblockSouXZGrid, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int its, int it2, int iGpu){

		kernel_stream_exec(ker_inject_interp_source_centerGrid<<<nblockSouCenterGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_modelRegDtw_sigmaxx[iGpu], dev_modelRegDtw_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		kernel_stream_exec(ker_inject_interp_source_stagGrid<<<nblockSouXGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_modelRegDtw_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		kernel_stream_exec(ker_inject_interp_source_stagGrid<<<nblockSouZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_modelRegDtw_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		kernel_stream_exec(ker_inject_interp_source_stagGrid<<<nblockSouXZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_modelRegDtw_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));

}
void launchDampCosineEdgeKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu]));
}
void launchDampCosineEdgeKernelsStreams(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_stream_exec(dampCosineEdge<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu]));
}
void launchDampCosineEdgeFreeSurfaceKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
	kernel_exec(dampCosineEdge_freesurface<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu]));
}
void launchDampCosineEdgeFreeSurfaceKernelsStreams(dim3 dimGrid, dim3 dimBlock, int iGpu){
	kernel_stream_exec(dampCosineEdge_freesurface<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu]));
}
void launchFwdRecordInterpDataKernels(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataZGrid, int nblockDataXZGrid, int its, int it2, int iGpu){
		kernel_exec(ker_record_interp_data_centerGrid<<<nblockDataCenterGrid, BLOCK_SIZE_DATA>>>( dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu]));
		kernel_exec(ker_record_interp_data_xGrid<<<nblockDataXGrid, BLOCK_SIZE_DATA>>>( dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu]));

		kernel_exec(ker_record_interp_data_zGrid<<<nblockDataZGrid, BLOCK_SIZE_DATA>>>( dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu]));

		kernel_exec(ker_record_interp_data_xzGrid<<<nblockDataXZGrid, BLOCK_SIZE_DATA>>>( dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu]));
}
void launchFwdRecordInterpDataKernelsStreams(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataZGrid, int nblockDataXZGrid, int its, int it2, int iGpu){
		kernel_stream_exec(ker_record_interp_data_centerGrid<<<nblockDataCenterGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>( dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu]));
		kernel_stream_exec(ker_record_interp_data_xGrid<<<nblockDataXGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>( dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu]));

		kernel_stream_exec(ker_record_interp_data_zGrid<<<nblockDataZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>( dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu]));

		kernel_stream_exec(ker_record_interp_data_xzGrid<<<nblockDataXZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>( dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu]));
}
void launchAdjStepKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(ker_step_adj<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}
void launchAdjStepKernelsStreams(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_stream_exec(ker_step_adj<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}
void launchAdjStepFreeSurfaceKernels(dim3 dimGridTop, dim3 dimGridBody, dim3 dimBlock, int iGpu){
		kernel_exec(ker_step_adj_surface_top<<<dimGridTop, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
		kernel_exec(ker_step_adj_surface_body<<<dimGridBody, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}
void launchAdjInterpInjectDataKernels(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataZGrid, int nblockDataXZGrid, int its, int it2, int iGpu){
		kernel_exec(ker_interp_inject_data_centerGrid<<<nblockDataCenterGrid, BLOCK_SIZE_DATA>>>( dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu]));

		kernel_exec(ker_interp_inject_data_xGrid<<<nblockDataXGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu]));


		kernel_exec(ker_interp_inject_data_zGrid<<<nblockDataZGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu]));
		kernel_exec(ker_interp_inject_data_xzGrid<<<nblockDataXZGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu]));

}
void launchAdjInterpInjectDataKernelsStreams(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataZGrid, int nblockDataXZGrid, int its, int it2, int iGpu){
		kernel_stream_exec(ker_interp_inject_data_centerGrid<<<nblockDataCenterGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>( dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu]));

		kernel_stream_exec(ker_interp_inject_data_xGrid<<<nblockDataXGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu]));


		kernel_stream_exec(ker_interp_inject_data_zGrid<<<nblockDataZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu]));
		kernel_stream_exec(ker_interp_inject_data_xzGrid<<<nblockDataXZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu]));

}
void launchAdjExtractSource(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouZGrid, int nblockSouXZGrid, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int its, int it2, int iGpu){

		kernel_exec(ker_record_interp_source_centerGrid<<<nblockSouCenterGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_modelRegDtw_sigmaxx[iGpu], dev_modelRegDtw_sigmazz[iGpu], its, it2, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		kernel_exec(ker_record_interp_source_stagGrid<<<nblockSouXGrid, BLOCK_SIZE_DATA>>>(dev_p0_vx[iGpu], dev_modelRegDtw_vx[iGpu], its, it2, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		kernel_exec(ker_record_interp_source_stagGrid<<<nblockSouZGrid, BLOCK_SIZE_DATA>>>(dev_p0_vz[iGpu], dev_modelRegDtw_vz[iGpu], its, it2, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		kernel_exec(ker_record_interp_source_stagGrid<<<nblockSouXZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxz[iGpu], dev_modelRegDtw_sigmaxz[iGpu], its, it2, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));

}
void launchAdjExtractSourceStreams(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouZGrid, int nblockSouXZGrid, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int its, int it2, int iGpu){

		kernel_stream_exec(ker_record_interp_source_centerGrid<<<nblockSouCenterGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_modelRegDtw_sigmaxx[iGpu], dev_modelRegDtw_sigmazz[iGpu], its, it2, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		kernel_stream_exec(ker_record_interp_source_stagGrid<<<nblockSouXGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0_vx[iGpu], dev_modelRegDtw_vx[iGpu], its, it2, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		kernel_stream_exec(ker_record_interp_source_stagGrid<<<nblockSouZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0_vz[iGpu], dev_modelRegDtw_vz[iGpu], its, it2, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		kernel_stream_exec(ker_record_interp_source_stagGrid<<<nblockSouXZGrid, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0_sigmaxz[iGpu], dev_modelRegDtw_sigmaxz[iGpu], its, it2, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));

}
void switchPointers(int iGpu){
		dev_temp1[iGpu] = dev_p0_vx[iGpu];
		dev_p0_vx[iGpu] = dev_p1_vx[iGpu];
		dev_p1_vx[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = dev_p0_vz[iGpu];
		dev_p0_vz[iGpu] = dev_p1_vz[iGpu];
		dev_p1_vz[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = dev_p0_sigmaxx[iGpu];
		dev_p0_sigmaxx[iGpu] = dev_p1_sigmaxx[iGpu];
		dev_p1_sigmaxx[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = dev_p0_sigmazz[iGpu];
		dev_p0_sigmazz[iGpu] = dev_p1_sigmazz[iGpu];
		dev_p1_sigmazz[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = dev_p0_sigmaxz[iGpu];
		dev_p0_sigmaxz[iGpu] = dev_p1_sigmaxz[iGpu];
		dev_p1_sigmaxz[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = NULL;
}

/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propShotsElasticFwdGpu(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId, int surfaceCondition) {


		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and copy model (arrays for sources for each wavefield) to device
		//                      c) allocate and initialize(0) data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu
		setupFwdGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);

		//Finite-difference grid and blocks
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		// Extraction grid size
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Injection grid size
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// Start propagation
		for (int its = 0; its < host_nts-1; its++){
				for (int it2 = 1; it2 < host_sub+1; it2++){

						if(surfaceCondition==0){
							// Step forward
							launchFwdStepKernels(dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernels(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);
							// Damp wavefields
							launchDampCosineEdgeKernels(dimGrid, dimBlock, iGpu);
							// Extract and interpolate data
							launchFwdRecordInterpDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
							// Switch pointers
							switchPointers(iGpu);
						}
						else if(surfaceCondition==1){
							// free surface fwd step kernels
							launchFwdStepFreeSurfaceKernels(nblocky, BLOCK_SIZE, dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernels(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);
							// free surface damp wfld kernels
							launchDampCosineEdgeFreeSurfaceKernels(dimGrid, dimBlock, iGpu);
							// Extract and interpolate data
							launchFwdRecordInterpDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
							// Switch pointers
							switchPointers(iGpu);
						}

				}
		}

		// Copy data back to host
		cuda_call(cudaMemcpy(dataRegDts_vx, dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vz, dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxx, dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmazz, dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxz, dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

}
void propShotsElasticFwdGpuWavefield(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, double* wavefieldDts, int iGpu, int iGpuId, int surfaceCondition){


		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and copy model (arrays for sources for each wavefield) to device
		//                      c) allocate and initialize(0) data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu



		setupFwdGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);

		//wavefield allocation and setup
		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_all, 5*host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
		cuda_call(cudaMemset(dev_wavefieldDts_all, 0, 5*host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

		//Finite-difference grid and blocks
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		// Extraction grid size
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Injection grid size
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// Start propagation
		for (int its = 0; its < host_nts-1; its++){
				for (int it2 = 1; it2 < host_sub+1; it2++){

						if(surfaceCondition==0){
							// Step forward
							launchFwdStepKernels(dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernels(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);

							// Damp wavefields
							launchDampCosineEdgeKernels(dimGrid, dimBlock, iGpu);

							// Extract wavefield
							kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_wavefieldDts_all, dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

							// Extract and interpolate data
							launchFwdRecordInterpDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);

							// Switch pointers
							switchPointers(iGpu);
						} else {

							// free surface fwd step kernels
							launchFwdStepFreeSurfaceKernels(nblocky, BLOCK_SIZE, dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernels(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);
							// free surface damp wfld kernels
							launchDampCosineEdgeFreeSurfaceKernels(dimGrid, dimBlock, iGpu);

							// Extract wavefield
							kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_wavefieldDts_all, dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

							// Extract and interpolate data
							launchFwdRecordInterpDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
							switchPointers(iGpu);
						}
				}
		}

		// Copy data back to host
		cuda_call(cudaMemcpy(dataRegDts_vx, dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vz, dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxx, dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmazz, dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxz, dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));



		// Copy wavefield back to host
		cuda_call(cudaMemcpy(wavefieldDts, dev_wavefieldDts_all, 5*host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

		cuda_call(cudaFree(dev_wavefieldDts_all));

}
/********************************************/
/*******Saving wavefield using streams*******/
/********************************************/
void propShotsElasticFwdGpuWavefieldStreams(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, double* wavefieldDts, int iGpu, int iGpuId, int surfaceCondition){


		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and copy model (arrays for sources for each wavefield) to device
		//                      c) allocate and initialize(0) data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu



		setupFwdGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);

		// Create streams for saving the wavefield
		cudaStreamCreate(&compStream[iGpu]);
		cudaStreamCreate(&transferStream[iGpu]);

		//wavefield allocation and setup
		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_left[iGpu], 5*host_nz*host_nx*sizeof(double))); // Allocate on device
		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_right[iGpu], 5*host_nz*host_nx*sizeof(double))); // Allocate on device
		cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], 5*host_nz*host_nx*sizeof(double))); // Allocate on device
		cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice[iGpu], 5*host_nz*host_nx*sizeof(double), cudaHostAllocDefault)); //Pinned memory on the Host
		//Initialize wavefield slides
		cuda_call(cudaMemset(dev_wavefieldDts_left[iGpu], 0, 5*host_nz*host_nx*sizeof(double))); // Initialize left slide on device
		cuda_call(cudaMemset(dev_wavefieldDts_right[iGpu], 0, 5*host_nz*host_nx*sizeof(double))); // Initialize right slide on device
		cuda_call(cudaMemset(dev_pStream[iGpu], 0, 5*host_nz*host_nx*sizeof(double))); // Initialize temporary slice on device
		cudaMemset(pin_wavefieldSlice[iGpu], 0, 5*host_nz*host_nx*sizeof(double)); // Initialize pinned memory

		// Finite-difference grid and blocks
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		// Extraction grid size
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Injection grid size
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		unsigned long long int time_slice = 5*host_nz*host_nx;

		// Start propagation
		for (int its = 0; its < host_nts-1; its++){
				for (int it2 = 1; it2 < host_sub+1; it2++){

						if(surfaceCondition==0){

							// Step forward
							launchFwdStepKernelsStreams(dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernelsStreams(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);

							// Damp wavefields
							launchDampCosineEdgeKernelsStreams(dimGrid, dimBlock, iGpu);

							// Extract wavefield
							kernel_stream_exec(interpWavefieldStreams<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_wavefieldDts_left[iGpu],dev_wavefieldDts_right[iGpu],dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

							// Extract and interpolate data
							launchFwdRecordInterpDataKernelsStreams(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);

							// Switch pointers
							switchPointers(iGpu);
					} else {

						// free surface fwd step kernels
						launchFwdStepFreeSurfaceKernelsStreams(nblocky, BLOCK_SIZE, dimGrid, dimBlock, iGpu);
						// Inject source
						launchFwdInjectSourceKernelsStreams(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, its, it2, iGpu);
						// free surface damp wfld kernels
						launchDampCosineEdgeFreeSurfaceKernelsStreams(dimGrid, dimBlock, iGpu);

						// Extract wavefield
						kernel_stream_exec(interpWavefieldStreams<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_wavefieldDts_left[iGpu],dev_wavefieldDts_right[iGpu],dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

						// Extract and interpolate data
						launchFwdRecordInterpDataKernelsStreams(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
						switchPointers(iGpu);

					}
				}

				/* Note: At that point pLeft [its] is ready to be transfered back to host */

				// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
				cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

				// Asynchronous copy of dev_wavefieldDts_left => dev_pStream [its] [compute]
				cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_wavefieldDts_left[iGpu], 5*host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

				// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
				if (its>0) {
					// Copying pinned memory using standard library
					std::memcpy(wavefieldDts+(its-1)*time_slice, pin_wavefieldSlice[iGpu], 5*host_nz*host_nx*sizeof(double));

				}

				// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
				cuda_call(cudaStreamSynchronize(compStream[iGpu]));

				// Asynchronous transfer of pStream => pin [its] [transfer]
				// Launch the transfer while we compute the next coarse time sample
				cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], 5*host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

				// Switch pointers
				dev_temp1[iGpu] = dev_wavefieldDts_left[iGpu];
				dev_wavefieldDts_left[iGpu] = dev_wavefieldDts_right[iGpu];
				dev_wavefieldDts_right[iGpu] = dev_temp1[iGpu];
				dev_temp1[iGpu] = NULL;
				cuda_call(cudaMemsetAsync(dev_wavefieldDts_right[iGpu], 0, 5*host_nz*host_nx*sizeof(double), compStream[iGpu])); // Reinitialize dev_wavefieldDts_right to zero
		}

		// Note: At that point, dev_wavefieldDts_left contains the value of the wavefield at nts-1 (last time sample) and the wavefield only has values up to nts-3
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));	// [transfer]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_wavefieldDts_left[iGpu], 5*host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu])); // Copy last time sample [nts-1] to temporary array [comp]
		memcpy(wavefieldDts+(host_nts-2)*time_slice,pin_wavefieldSlice[iGpu], 5*host_nz*host_nx*sizeof(double)); // Copy pinned array to wavefield array [nts-2] [host]
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));	// Make sure dev_pLeft => dev_pStream is done
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], 5*host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu])); // Copy pStream to pinned memory on device
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
		memcpy(wavefieldDts+(host_nts-1)*time_slice,pin_wavefieldSlice[iGpu], 5*host_nz*host_nx*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]

		// Destroying streams
		cuda_call(cudaStreamDestroy(compStream[iGpu]));
		cuda_call(cudaStreamDestroy(transferStream[iGpu]));

		// Copy data back to host
		cuda_call(cudaMemcpy(dataRegDts_vx, dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vz, dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxx, dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmazz, dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxz, dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

		//Deallocating wavefield-related host and device memory
		cuda_call(cudaFree(dev_wavefieldDts_left[iGpu]));
		cuda_call(cudaFree(dev_wavefieldDts_right[iGpu]));
		cuda_call(cudaFree(dev_pStream[iGpu]));
		cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));

}

/****************************************************************************************/
/******************************* Nonlinear adjoint propagation **************************/
/****************************************************************************************/
void propShotsElasticAdjGpu(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId, int surfaceCondition) {

		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
		//                      c) allocate and copy data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu
		setupAdjGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);




		// Grid and block dimensions for stepper
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGridTop(1, nblocky);
		dim3 dimGridBody(nblockx-1, nblocky);

		// Grid and block dimensions for data injection
		// Injection grid size (data)
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Extraction grid size (source)
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// double *temp=new double[host_nz*host_nx];

		// Start propagation
		for (int its = host_nts-2; its > -1; its--){
				for (int it2 = host_sub-1; it2 > -1; it2--){

						if(surfaceCondition==0){
							// Step back in time
							launchAdjStepKernels(dimGrid, dimBlock, iGpu);
							// Inject data
							launchAdjInterpInjectDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
							// Damp wavefield
							launchDampCosineEdgeKernels(dimGrid, dimBlock, iGpu);
							// Extract model
							launchAdjExtractSource(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, its, it2, iGpu);
							// Switch pointers
							switchPointers(iGpu);
						}
						else if(surfaceCondition==1){
							// step back free surface  kernels
							launchAdjStepFreeSurfaceKernels(dimGridTop, dimGridBody, dimBlock, iGpu);
							// launchAdjStepKernels(dimGrid, dimBlock, iGpu);
							// Inject Data kernels
							launchAdjInterpInjectDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);
							// Damp wavefield free surface kernels
							launchDampCosineEdgeFreeSurfaceKernels(dimGrid, dimBlock, iGpu);
							// Extract model kernels
							launchAdjExtractSource(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, its, it2, iGpu);
							//switch pointers kernels
							switchPointers(iGpu);
						}

				}

		}

		// Copy model back to host
		cuda_call(cudaMemcpy(modelRegDtw_vx, dev_modelRegDtw_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_vz, dev_modelRegDtw_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxx, dev_modelRegDtw_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmazz, dev_modelRegDtw_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxz, dev_modelRegDtw_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

}
void propShotsElasticAdjGpuWavefield(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, double* wavefieldDts, int iGpu, int iGpuId){

		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
		//                      c) allocate and copy data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu
		setupAdjGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);
		// Grid and block dimensions for stepper
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_all, 5*host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
		cuda_call(cudaMemset(dev_wavefieldDts_all, 0, 5*host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

		// Grid and block dimensions for data injection
		// Injection grid size (data)
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Extraction grid size (source)
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// double *temp=new double[host_nz*host_nx];

		// Start propagation
		for (int its = host_nts-2; its > -1; its--){
				for (int it2 = host_sub-1; it2 > -1; it2--){

						// Step back in time
						launchAdjStepKernels(dimGrid, dimBlock, iGpu);

						// Inject data
						launchAdjInterpInjectDataKernels(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);

						// Damp wavefield
						launchDampCosineEdgeKernels(dimGrid, dimBlock, iGpu);

						// Extract wavefield
						kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_wavefieldDts_all, dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

						// Extract model
						launchAdjExtractSource(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, its, it2, iGpu);

						// Switch pointers
						switchPointers(iGpu);
				}

		}

		// Copy model back to host
		cuda_call(cudaMemcpy(modelRegDtw_vx, dev_modelRegDtw_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_vz, dev_modelRegDtw_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxx, dev_modelRegDtw_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmazz, dev_modelRegDtw_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxz, dev_modelRegDtw_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Copy wavefield back to host
		cuda_call(cudaMemcpy(wavefieldDts, dev_wavefieldDts_all, 5*host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

		cuda_call(cudaFree(dev_wavefieldDts_all));
}

/********************************************/
/*******Saving wavefield using streams*******/
/********************************************/
void propShotsElasticAdjGpuWavefieldStreams(double *modelRegDtw_vx, double *modelRegDtw_vz, double *modelRegDtw_sigmaxx, double *modelRegDtw_sigmazz, double *modelRegDtw_sigmaxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, double* wavefieldDts, int iGpu, int iGpuId){

		//setup:                a) src and receiver positions allocation and copying to device
		//                      b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
		//                      c) allocate and copy data (recevier recordings arrays) to device
		//                      d) allocate and copy wavefield time slices to gpu
		setupAdjGpu(modelRegDtw_vx, modelRegDtw_vz, modelRegDtw_sigmaxx, modelRegDtw_sigmazz, modelRegDtw_sigmaxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);
		// Grid and block dimensions for stepper
		int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		// Create streams for saving the wavefield
		cudaStreamCreate(&compStream[iGpu]);
		cudaStreamCreate(&transferStream[iGpu]);

		unsigned long long int time_slice = 5*host_nz*host_nx;

		//wavefield allocation and setup
		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_left[iGpu], time_slice*sizeof(double))); // Allocate on device
		cuda_call(cudaMalloc((void**) &dev_wavefieldDts_right[iGpu], time_slice*sizeof(double))); // Allocate on device
		cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], time_slice*sizeof(double))); // Allocate on device
		cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice[iGpu], time_slice*sizeof(double), cudaHostAllocDefault)); //Pinned memory on the Host
		//Initialize wavefield slides
		cuda_call(cudaMemset(dev_wavefieldDts_left[iGpu], 0, time_slice*sizeof(double))); // Initialize left slide on device
		cuda_call(cudaMemset(dev_wavefieldDts_right[iGpu], 0, time_slice*sizeof(double))); // Initialize right slide on device
		cuda_call(cudaMemset(dev_pStream[iGpu], 0, time_slice*sizeof(double))); // Initialize temporary slice on device
		cudaMemset(pin_wavefieldSlice[iGpu], 0, time_slice*sizeof(double)); // Initialize pinned memory

		// Grid and block dimensions for data injection
		// Injection grid size (data)
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Extraction grid size (source)
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// Start propagation
		for (int its = host_nts-2; its > -1; its--){
				for (int it2 = host_sub-1; it2 > -1; it2--){

						// Step back in time
						launchAdjStepKernelsStreams(dimGrid, dimBlock, iGpu);

						// Inject data
						launchAdjInterpInjectDataKernelsStreams(nblockDataCenterGrid, nblockDataXGrid, nblockDataZGrid, nblockDataXZGrid, its, it2, iGpu);

						// Damp wavefield
						launchDampCosineEdgeKernelsStreams(dimGrid, dimBlock, iGpu);

						// Extract wavefield
						kernel_stream_exec(interpWavefieldStreams<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_wavefieldDts_left[iGpu],dev_wavefieldDts_right[iGpu],dev_p0_vx[iGpu],dev_p0_vz[iGpu],dev_p0_sigmaxx[iGpu],dev_p0_sigmazz[iGpu],dev_p0_sigmaxz[iGpu], its, it2));

						// Extract model
						launchAdjExtractSourceStreams(nblockSouCenterGrid,nblockSouXGrid,nblockSouZGrid,nblockSouXZGrid,nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, its, it2, iGpu);

						// Switch pointers
						switchPointers(iGpu);
				}

				/* Note: At that point pLeft [its] is ready to be transfered back to host */

				// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
				cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

				// Asynchronous copy of dev_wavefieldDts_right => dev_pStream [its] [compute]
				cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_wavefieldDts_right[iGpu], time_slice*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

				// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
				if (its<host_nts-2) {
					// Copying pinned memory using standard library
					std::memcpy(wavefieldDts+(its+2)*time_slice, pin_wavefieldSlice[iGpu], time_slice*sizeof(double));
				}

				// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
				cuda_call(cudaStreamSynchronize(compStream[iGpu]));

				// Asynchronous transfer of pStream => pin [its] [transfer]
				// Launch the transfer while we compute the next coarse time sample
				cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], time_slice*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

				// Switch pointers
				dev_temp1[iGpu] = dev_wavefieldDts_left[iGpu];
				dev_wavefieldDts_left[iGpu] = dev_wavefieldDts_right[iGpu];
				dev_wavefieldDts_right[iGpu] = dev_temp1[iGpu];
				dev_temp1[iGpu] = NULL;
				cuda_call(cudaMemsetAsync(dev_wavefieldDts_left[iGpu], 0, time_slice*sizeof(double), compStream[iGpu])); // Reinitialize dev_wavefieldDts_right to zero

		}

		// Note: At that point, dev_wavefieldDts_left contains the value of the wavefield at nts-1 (last time sample) and the wavefield only has values up to nts-3
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));	// [transfer]
		std::memcpy(wavefieldDts+time_slice,pin_wavefieldSlice[iGpu], time_slice*sizeof(double)); // Copy pinned array to wavefield array [its=1] [host]
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));	// Make sure dev_pLeft => dev_pStream is done
		cuda_call(cudaMemcpyAsync(wavefieldDts, dev_wavefieldDts_right[iGpu], time_slice*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu])); // Copy pinned array to wavefield array for the last sample [its=0] [host]
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy model back to host
		cuda_call(cudaMemcpy(modelRegDtw_vx, dev_modelRegDtw_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_vz, dev_modelRegDtw_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxx, dev_modelRegDtw_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmazz, dev_modelRegDtw_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(modelRegDtw_sigmaxz, dev_modelRegDtw_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));


		// Destroying streams
		cuda_call(cudaStreamDestroy(compStream[iGpu]));
		cuda_call(cudaStreamDestroy(transferStream[iGpu]));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDtw_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDtw_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));

		//Deallocating wavefield-related host and device memory
		cuda_call(cudaFree(dev_wavefieldDts_left[iGpu]));
		cuda_call(cudaFree(dev_wavefieldDts_right[iGpu]));
		cuda_call(cudaFree(dev_pStream[iGpu]));
		cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));
}

/* #############################################################################
												FUNCTIONS FOR TESTING
############################################################################# */
void get_dev_zCoeff(double *hp){
	cuda_call(cudaMemcpyFromSymbol(hp, dev_zCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyDeviceToHost));
}
void get_dev_xCoeff(double *hp){
	cuda_call(cudaMemcpyFromSymbol(hp, dev_xCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyDeviceToHost));
}
