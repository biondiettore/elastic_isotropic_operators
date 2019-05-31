#include <cstring>
#include <iostream>
#include "BornElasticGpuFunctions.h"
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


void initBornGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

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
				dev_sourceRegDtw_vx = new double*[nGpu];
				dev_sourceRegDtw_vz = new double*[nGpu];
				dev_sourceRegDtw_sigmaxx = new double*[nGpu];
				dev_sourceRegDtw_sigmazz = new double*[nGpu];
				dev_sourceRegDtw_sigmaxz = new double*[nGpu];
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

                // Pointers specific to Born operator
                dev_ssVxLeft  = new double*[nGpu];
                dev_ssVxRight = new double*[nGpu];
                dev_ssVzLeft  = new double*[nGpu];
                dev_ssVzRight = new double*[nGpu];
                dev_ssSigmaxxLeft  = new double*[nGpu];
                dev_ssSigmaxxRight = new double*[nGpu];
                dev_ssSigmazzLeft  = new double*[nGpu];
                dev_ssSigmazzRight = new double*[nGpu];
                dev_ssSigmaxzLeft  = new double*[nGpu];
                dev_ssSigmaxzRight = new double*[nGpu];

                dev_drhox = new double*[nGpu];
                dev_drhoz = new double*[nGpu];
                dev_dlame = new double*[nGpu];
                dev_dmu   = new double*[nGpu];
                dev_dmuxz = new double*[nGpu];

                dev_wavefieldVx = new double*[nGpu];
                dev_wavefieldVz = new double*[nGpu];

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
				assert (1==2);
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
				assert (1==2);
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
				assert (1==2);
		}

		/**************************** COPY TO CONSTANT MEMORY *******************************/
		// Laplacian coefficients
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
        double inv_dts = 1.0/dts;
        cuda_call(cudaMemcpyToSymbol(dev_dts_inv, &inv_dts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Inverse of the time-source sampling

}


void allocateBornElasticGpu(double *rhoxDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDt, int iGpu, int iGpuId, int useStreams){

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

    //Allocating memory specific to Born operator
    cuda_call(cudaMalloc((void**) &dev_ssVxLeft[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssVxRight[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssVzLeft[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssVzRight[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmaxxLeft[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmaxxRight[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmazzLeft[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmazzRight[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmaxzLeft[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_ssSigmaxzRight[iGpu], host_nz*host_nx*sizeof(double)));

    cuda_call(cudaMalloc((void**) &dev_drhox[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_drhoz[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_dlame[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_dmu[iGpu], host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_dmuxz[iGpu], host_nz*host_nx*sizeof(double)));

    //If streams are used, allocate wavefield memory on the device
    if(useStreams == 1){
        cuda_call(cudaMalloc((void**) &dev_wavefieldVx[iGpu], host_nz*host_nx*host_nts*sizeof(double)));
        cuda_call(cudaMalloc((void**) &dev_wavefieldVz[iGpu], host_nz*host_nx*host_nts*sizeof(double)));
    }
}

void deallocateBornElasticGpu(int iGpu, int iGpuId, int useStreams){
		cudaSetDevice(iGpuId); // Set device number on GPU cluster

		// Deallocate scaled elastic params
		cuda_call(cudaFree(dev_rhoxDtw[iGpu]));
		cuda_call(cudaFree(dev_rhozDtw[iGpu]));
		cuda_call(cudaFree(dev_lamb2MuDtw[iGpu]));
		cuda_call(cudaFree(dev_lambDtw[iGpu]));
		cuda_call(cudaFree(dev_muxzDtw[iGpu]));

		// Deallocate wavefield slices
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

		//Deallocate memory specific to Born operator
		cuda_call(cudaFree(dev_ssVxLeft[iGpu]));
		cuda_call(cudaFree(dev_ssVxRight[iGpu]));
		cuda_call(cudaFree(dev_ssVzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssVzRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxxLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxxRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmazzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmazzRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxzRight[iGpu]));

		cuda_call(cudaFree(dev_drhox[iGpu]));
		cuda_call(cudaFree(dev_drhoz[iGpu]));
		cuda_call(cudaFree(dev_dlame[iGpu]));
		cuda_call(cudaFree(dev_dmu[iGpu]));
		cuda_call(cudaFree(dev_dmuxz[iGpu]));

    if(useStreams == 1){
        cuda_call(cudaFree(dev_wavefieldVx[iGpu]));
        cuda_call(cudaFree(dev_wavefieldVz[iGpu]));
    }
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

//allocate source terms on device
void sourceAllocateGpu(int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int iGpu){
		cuda_call(cudaMalloc((void**) &dev_sourceRegDtw_vx[iGpu], nSourcesRegXGrid*host_ntw*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_sourceRegDtw_vz[iGpu], nSourcesRegZGrid*host_ntw*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_sourceRegDtw_sigmaxx[iGpu], nSourcesRegCenterGrid*host_ntw*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_sourceRegDtw_sigmazz[iGpu], nSourcesRegCenterGrid*host_ntw*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_sourceRegDtw_sigmaxz[iGpu], nSourcesRegXZGrid*host_ntw*sizeof(double)));
}

//copy source terms from host to device
void sourceCopyToGpu(double *sourceRegDtw_vx, double *sourceRegDtw_vz, double *sourceRegDtw_sigmaxx, double *sourceRegDtw_sigmazz, double *sourceRegDtw_sigmaxz, int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int iGpu){
		cuda_call(cudaMemcpy(dev_sourceRegDtw_vx[iGpu], sourceRegDtw_vx, nSourcesRegXGrid*host_ntw*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_sourceRegDtw_vz[iGpu], sourceRegDtw_vz, nSourcesRegZGrid*host_ntw*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_sourceRegDtw_sigmaxx[iGpu], sourceRegDtw_sigmaxx, nSourcesRegCenterGrid*host_ntw*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_sourceRegDtw_sigmazz[iGpu], sourceRegDtw_sigmazz, nSourcesRegCenterGrid*host_ntw*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_sourceRegDtw_sigmaxz[iGpu], sourceRegDtw_sigmaxz, nSourcesRegXZGrid*host_ntw*sizeof(double), cudaMemcpyHostToDevice));
}
//allocate model on device
void dataAllocateGpu(int nReceiversRegCenterGrid, int nReceiversRegXGrid, int nReceiversRegZGrid, int nReceiversRegXZGrid, int iGpu){
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double))); // Allocate output on device
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double)));
}
void dataCopyToGpu(double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int nReceiversRegCenterGrid, int nReceiversRegXGrid, int nReceiversRegZGrid, int nReceiversRegXZGrid, int iGpu){
		cuda_call(cudaMemcpy(dev_dataRegDts_vx[iGpu], dataRegDts_vx, nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy rec signals on device
		cuda_call(cudaMemcpy(dev_dataRegDts_vz[iGpu], dataRegDts_vz, nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxx[iGpu], dataRegDts_sigmaxx, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmazz[iGpu], dataRegDts_sigmazz, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxz[iGpu], dataRegDts_sigmaxz, nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
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
		//Born specific slices
		cuda_call(cudaMemset(dev_ssVxLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssVxRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssVzLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssVzRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmaxxLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmaxxRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmazzLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmazzRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmaxzLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
		cuda_call(cudaMemset(dev_ssSigmaxzRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
}

void modelCopyToGpu(double *drhox, double *drhoz, double *dlame, double *dmu, double *dmuxz,int iGpu){
		cuda_call(cudaMemcpy(dev_drhox[iGpu], drhox, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_drhoz[iGpu], drhoz, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dlame[iGpu], dlame, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmu[iGpu], dmu, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmuxz[iGpu], dmuxz, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
}

void setupBornFwdGpu(double *sourceRegDtw_vx, double *sourceRegDtw_vz, double *sourceRegDtw_sigmaxx, double *sourceRegDtw_sigmazz, double *sourceRegDtw_sigmaxz, double *drhox, double *drhoz, double *dlame, double *dmu, double *dmuxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

	//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu);

		// Source - wavelets for each wavefield component. Allocate and copy to gpu
		sourceAllocateGpu(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);
		sourceCopyToGpu(sourceRegDtw_vx, sourceRegDtw_vz, sourceRegDtw_sigmaxx, sourceRegDtw_sigmazz, sourceRegDtw_sigmaxz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegZGrid, nSourcesRegXZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);
		dataInitializeOnGpu(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegZGrid, nReceiversRegXZGrid, iGpu);

		//Initialize wavefield slices to zero
		wavefieldInitializeOnGpu(iGpu);

		//Initialize model perturbations
    modelCopyToGpu(drhox,drhoz,dlame,dmu,dmuxz,iGpu);
}

void launchFwdStepKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(ker_step_fwd<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_vx[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_vx[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_rhoxDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu]));
}

void launchFwdInjectSourceKernels(int nSourcesRegCenterGrid, int nSourcesRegXGrid, int nSourcesRegZGrid, int nSourcesRegXZGrid, int itw, int iGpu){
		kernel_exec(ker_inject_source_centerGrid<<<1, nSourcesRegCenterGrid>>>(dev_modelRegDtw_sigmaxx[iGpu], dev_modelRegDtw_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], itw-1, dev_sourcesPositionRegCenterGrid[iGpu]));

		kernel_exec(ker_inject_source_xGrid<<<1, nSourcesRegXGrid>>>(dev_modelRegDtw_vx[iGpu], dev_p0_vx[iGpu], itw-1, dev_sourcesPositionRegXGrid[iGpu]));
		kernel_exec(ker_inject_source_zGrid<<<1, nSourcesRegZGrid>>>(dev_modelRegDtw_vz[iGpu], dev_p0_vz[iGpu], itw-1, dev_sourcesPositionRegZGrid[iGpu]));

		kernel_exec(ker_inject_source_xzGrid<<<1, nSourcesRegXZGrid>>>(dev_modelRegDtw_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], itw-1, dev_sourcesPositionRegXZGrid[iGpu]));
}

void launchDampCosineEdgeKernels(dim3 dimGrid, dim3 dimBlock, int iGpu){
		kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu]));
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
/*********************************** Born forward operator ******************************/
/****************************************************************************************/
void BornShotsFwdGpu(double *sourceRegDtw_vx, double *sourceRegDtw_vz, double *sourceRegDtw_sigmaxx, double *sourceRegDtw_sigmazz, double *sourceRegDtw_sigmaxz, double *drhox, double *drhoz, double *dlame, double *dmu, double *dmuxz, double *dataRegDts_vx, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, int *sourcesPositionRegCenterGrid, int nSourcesRegCenterGrid, int *sourcesPositionRegXGrid, int nSourcesRegXGrid, int *sourcesPositionRegZGrid, int nSourcesRegZGrid, int *sourcesPositionRegXZGrid, int nSourcesRegXZGrid, int *receiversPositionRegCenterGrid, int nReceiversRegCenterGrid, int *receiversPositionRegXGrid, int nReceiversRegXGrid, int *receiversPositionRegZGrid, int nReceiversRegZGrid, int *receiversPositionRegXZGrid, int nReceiversRegXZGrid, int iGpu, int iGpuId, int surfaceCondition, int useStreams){
    //setup:                a) src and receiver positions allocation and copying to device
    //                      b) allocate and copy model (arrays for sources for each wavefield) to device
    //                      c) allocate and initialize(0) data (recevier recordings arrays) to device
    //                      d) allocate and copy wavefield time slices to gpu
    setupBornFwdGpu(sourceRegDtw_vx, sourceRegDtw_vz, sourceRegDtw_sigmaxx, sourceRegDtw_sigmazz, sourceRegDtw_sigmaxz, drhox, drhoz, dlame, dmu, dmuxz, dataRegDts_vx, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmazz, dataRegDts_sigmaxz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, iGpu, iGpuId);

		//Finite-difference grid and blocks
		int nblockx;
		if(surfaceCondition==0){
			nblockx = (host_nz-5-FAT) / BLOCK_SIZE;
		}
		else if(surfaceCondition==1){
			nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
		}
		int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		if(useStreams == 0){
			//Born operator w/o the use of Streams
			/************************** Source wavefield computation ****************************/
			for (int its = 0; its < host_nts-1; its++){
					for (int it2 = 1; it2 < host_sub+1; it2++){
							// Compute fine time-step index
							int itw = its * host_sub + it2;

							// Step forward
							launchFwdStepKernels(dimGrid, dimBlock, iGpu);
							// Inject source
							launchFwdInjectSourceKernels(nSourcesRegCenterGrid,nSourcesRegXGrid,nSourcesRegZGrid,nSourcesRegXZGrid, itw, iGpu);

							// Damp wavefields
							launchDampCosineEdgeKernels(dimGrid, dimBlock, iGpu);

							// Extract wavefield components
							kernel_exec(interpWavefieldSingleComp<<<dimGrid, dimBlock>>>(dev_wavefieldVx[iGpu], dev_p0_vx[iGpu], its, it2));
							kernel_exec(interpWavefieldSingleComp<<<dimGrid, dimBlock>>>(dev_wavefieldVz[iGpu], dev_p0_vz[iGpu], its, it2));

							// Switch pointers
							switchPointers(iGpu);
					}
			}

		} else {
			//Born operator w/ the use of Streams

		}
}
