#include "varDeclare.h"
#include <stdio.h>
/****************************************************************************************/
/*************************************** Extraction *************************************/
/****************************************************************************************/
/* Extract and interpolate data on center grid */
__global__ void ker_record_interp_data_centerGrid(
  double *dev_newTimeSlice_sigmaxx,
  double *dev_newTimeSlice_sigmazz,
  double *dev_signalOut_sigmaxx,
  double *dev_signalOut_sigmazz,
  int its, int it2, int *dev_receiversPositionRegCenterGrid) {

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegCenterGrid) {
  // printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
  dev_signalOut_sigmaxx[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] * dev_interpFilter[it2];
  dev_signalOut_sigmaxx[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];
  dev_signalOut_sigmazz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] * dev_interpFilter[it2];
  dev_signalOut_sigmazz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];
  }
}

/* Extract and interpolate data on x shifted grid */
__global__ void ker_record_interp_data_xGrid(
  double *dev_newTimeSlice_vx,
  double *dev_signalOut_vx,
  int its, int it2, int *dev_receiversPositionRegXGrid) {

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegXGrid) {
  // printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
  dev_signalOut_vx[dev_nts*iThread+its]   += dev_newTimeSlice_vx[dev_receiversPositionRegXGrid[iThread]] * dev_interpFilter[it2];
  dev_signalOut_vx[dev_nts*iThread+its+1] += dev_newTimeSlice_vx[dev_receiversPositionRegXGrid[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];

  }
}

/* Extract and interpolate data on z shifted grid */
__global__ void ker_record_interp_data_zGrid(
  double *dev_newTimeSlice_vz,
  double *dev_signalOut_vz,
  int its, int it2, int *dev_receiversPositionRegZGrid) {

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegZGrid) {
  // printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
  dev_signalOut_vz[dev_nts*iThread+its]   += dev_newTimeSlice_vz[dev_receiversPositionRegZGrid[iThread]] * dev_interpFilter[it2];
  dev_signalOut_vz[dev_nts*iThread+its+1] += dev_newTimeSlice_vz[dev_receiversPositionRegZGrid[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];

  }
}

/* Extract and interpolate data on xz shifted grid */
__global__ void ker_record_interp_data_xzGrid(
  double *dev_newTimeSlice_sigmaxz,
  double *dev_signalOut_sigmaxz,
  int its, int it2, int *dev_receiversPositionRegXZGrid) {

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegXZGrid) {
  // printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
  dev_signalOut_sigmaxz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] * dev_interpFilter[it2];
  dev_signalOut_sigmaxz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];

  }
}
/*extract source thar are on center grid */
__global__ void ker_record_source_centerGrid(double *dev_newTimeSlice_sigmaxx, double *dev_newTimeSlice_sigmazz,
   double *dev_signalOut_sigmaxx, double *dev_signalOut_sigmazz,
   int itw, int *dev_sourcesPositionRegCenterGrid) {
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_signalOut_sigmaxx[dev_ntw*iThread + itw] += dev_newTimeSlice_sigmaxx[dev_sourcesPositionRegCenterGrid[iThread]];
  dev_signalOut_sigmazz[dev_ntw*iThread + itw] += dev_newTimeSlice_sigmazz[dev_sourcesPositionRegCenterGrid[iThread]];
}
/*extract source thar are on x grid */
__global__ void ker_record_source_XGrid(double *dev_newTimeSlice_vx,
   double *dev_signalOut_vx,
   int itw, int *dev_sourcesPositionRegXGrid) {
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_signalOut_vx[dev_ntw*iThread + itw] += dev_newTimeSlice_vx[dev_sourcesPositionRegXGrid[iThread]];
}
/*extract source thar are on z grid */
__global__ void ker_record_source_ZGrid(double *dev_newTimeSlice_vz,
   double *dev_signalOut_vz,
   int itw, int *dev_sourcesPositionRegZGrid) {
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_signalOut_vz[dev_ntw*iThread + itw] += dev_newTimeSlice_vz[dev_sourcesPositionRegZGrid[iThread]];
}
/*extract source thar are on xz grid */
__global__ void ker_record_source_XZGrid(double *dev_newTimeSlice_sigmaxz,
   double *dev_signalOut_sigmaxz,
   int itw, int *dev_sourcesPositionRegXZGrid) {
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_signalOut_sigmaxz[dev_ntw*iThread + itw] += dev_newTimeSlice_sigmaxz[dev_sourcesPositionRegXZGrid[iThread]];
}
/****************************************************************************************/
/***************************************** Injection ************************************/
/****************************************************************************************/
/* Inject source on center grid: no need for a "if" statement because the number of threads = nb devices */
__global__ void ker_inject_source_centerGrid(double *dev_signalIn_sigmaxx,
   double *dev_signalIn_sigmazz,
   double *dev_timeSlice_sigmaxx,
   double *dev_timeSlice_sigmazz,
   int itw, int *dev_sourcesPositionRegCenterGrid){

  //thread per source device
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_timeSlice_sigmaxx[dev_sourcesPositionRegCenterGrid[iThread]] += dev_signalIn_sigmaxx[iThread * dev_ntw + itw];
  dev_timeSlice_sigmazz[dev_sourcesPositionRegCenterGrid[iThread]] += dev_signalIn_sigmazz[iThread * dev_ntw + itw];
}

/* Inject source on x shifted grid: no need for a "if" statement because the number of threads = nb devices */
__global__ void ker_inject_source_xGrid(double *dev_signalIn_vx,
  double *dev_timeSlice_vx,
  int itw, int *dev_sourcesPositionRegXGrid){

  //thread per source device
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_timeSlice_vx[dev_sourcesPositionRegXGrid[iThread]] += dev_signalIn_vx[iThread * dev_ntw + itw]; // Time is the fast axis
}
/* Inject source on z shifted grid: no need for a "if" statement because the number of threads = nb devices */
__global__ void ker_inject_source_zGrid(double *dev_signalIn_vz,
  double *dev_timeSlice_vz,
  int itw, int *dev_sourcesPositionRegZGrid){

  //thread per source device
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_timeSlice_vz[dev_sourcesPositionRegZGrid[iThread]] += dev_signalIn_vz[iThread * dev_ntw + itw]; // Time is the fast axis
}
/* Inject source on xz shifted grid: no need for a "if" statement because the number of threads = nb devices */
__global__ void ker_inject_source_xzGrid(double *dev_signalIn_sigmaxz,
   double *dev_timeSlice_sigmaxz,
   int itw, int *dev_sourcesPositionRegXZGrid){

  //thread per source device
  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  dev_timeSlice_sigmaxz[dev_sourcesPositionRegXZGrid[iThread]] += dev_signalIn_sigmaxz[iThread * dev_ntw + itw]; // Time is the fast axis
}

__global__ void ker_interp_inject_data_centerGrid(double *dev_signalIn_sigmaxx,
  double *dev_signalIn_sigmazz,
  double *dev_timeSlice_sigmaxx,
  double *dev_timeSlice_sigmazz,
  int its, int it2, int *dev_receiversPositionRegCenterGrid){

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegCenterGrid) {
  dev_timeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] += dev_signalIn_sigmaxx[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn_sigmaxx[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
  dev_timeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] += dev_signalIn_sigmazz[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn_sigmazz[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
  }
}
__global__ void ker_interp_inject_data_xGrid(double *dev_signalIn_vx,
     double *dev_timeSlice_vx,
     int its, int it2, int *dev_receiversPositionRegXGrid){

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegXGrid) {
  dev_timeSlice_vx[dev_receiversPositionRegXGrid[iThread]] += dev_signalIn_vx[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn_vx[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
  }
}
__global__ void ker_interp_inject_data_zGrid(double *dev_signalIn_vz,
     double *dev_timeSlice_vz,
     int its, int it2, int *dev_receiversPositionRegZGrid){

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegZGrid) {
  dev_timeSlice_vz[dev_receiversPositionRegZGrid[iThread]] += dev_signalIn_vz[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn_vz[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
  }
}
__global__ void ker_interp_inject_data_xzGrid(double *dev_signalIn_sigmaxz,
     double *dev_timeSlice_sigmaxz,
     int its, int it2, int *dev_receiversPositionRegXZGrid){

  int iThread = blockIdx.x * blockDim.x + threadIdx.x;
  if (iThread < dev_nReceiversRegXZGrid) {
  dev_timeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] += dev_signalIn_sigmaxz[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn_sigmaxz[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
  }
}
/****************************************************************************************/
/*********************************** Forward steppers ***********************************/
/****************************************************************************************/
/* kernel to compute forward time step */
__global__ void ker_step_fwd(double* dev_o_vx, double* dev_o_vz, double* dev_o_sigmaxx, double* dev_o_sigmazz, double* dev_o_sigmaxz,
   double* dev_c_vx, double* dev_c_vz, double* dev_c_sigmaxx, double* dev_c_sigmazz, double* dev_c_sigmaxz,
   double* dev_n_vx, double* dev_n_vz, double* dev_n_sigmaxx, double* dev_n_sigmazz, double* dev_n_sigmaxz,
   double* dev_rhoxDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw){
   //double* dev_c_all,double* dev_n_all, double* dev_elastic_param_scaled) {

  // Allocate shared memory for each wavefield component
  __shared__ double shared_c_vx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vx
  __shared__ double shared_c_vz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vz
  __shared__ double shared_c_sigmaxx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmaxx
  __shared__ double shared_c_sigmazz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmazz
  __shared__ double shared_c_sigmaxz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmaxz

  // calculate global and local x/z coordinates
  int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
  int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Copy current slice from global to shared memory for each wavefield component -- center
  shared_c_vx[ixLocal][izLocal] = dev_c_vx[iGlobal]; // vx
  shared_c_vz[ixLocal][izLocal] = dev_c_vz[iGlobal]; // vz
  shared_c_sigmaxx[ixLocal][izLocal] = dev_c_sigmaxx[iGlobal]; // sigmaxx
  shared_c_sigmazz[ixLocal][izLocal] = dev_c_sigmazz[iGlobal]; // sigmaxz
  shared_c_sigmaxz[ixLocal][izLocal] = dev_c_sigmaxz[iGlobal]; // sigmazz

  // Copy current slice from global to shared memory for each wavefield component -- edges
  if (threadIdx.y < FAT) {
  // vx
  shared_c_vx[ixLocal-FAT][izLocal] = dev_c_vx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vx[ixLocal+BLOCK_SIZE][izLocal] = dev_c_vx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // vz
  shared_c_vz[ixLocal-FAT][izLocal] = dev_c_vz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_vz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxx
  shared_c_sigmaxx[ixLocal-FAT][izLocal] = dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxx[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmazz
  shared_c_sigmazz[ixLocal-FAT][izLocal] = dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmazz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxz
  shared_c_sigmaxz[ixLocal-FAT][izLocal] = dev_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  }
  if (threadIdx.x < FAT) {
  // vx
  shared_c_vx[ixLocal][izLocal-FAT] = dev_c_vx[iGlobal-FAT]; // Up
  shared_c_vx[ixLocal][izLocal+BLOCK_SIZE] = dev_c_vx[iGlobal+BLOCK_SIZE]; // Down
  // vz
  shared_c_vz[ixLocal][izLocal-FAT] = dev_c_vz[iGlobal-FAT]; // Up
  shared_c_vz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_vz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxx
  shared_c_sigmaxx[ixLocal][izLocal-FAT] = dev_c_sigmaxx[iGlobal-FAT]; // Up
  shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmaxx[iGlobal+BLOCK_SIZE]; // Down
  // sigmazz
  shared_c_sigmazz[ixLocal][izLocal-FAT] = dev_c_sigmazz[iGlobal-FAT]; // Up
  shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmazz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxz
  shared_c_sigmaxz[ixLocal][izLocal-FAT] = dev_c_sigmaxz[iGlobal-FAT]; // Up
  shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmaxz[iGlobal+BLOCK_SIZE]; // Down
  }
  __syncthreads(); // Synchronise all threads within each block -- look new sync options

  //new vx
  dev_n_vx[iGlobal] = //old vx
    dev_o_vx[iGlobal] +
    dev_rhoxDtw[iGlobal] * (
    //first derivative in negative x direction of current sigmaxx
   dev_xCoeff[0]*(shared_c_sigmaxx[ixLocal][izLocal]-shared_c_sigmaxx[ixLocal-1][izLocal])+
   dev_xCoeff[1]*(shared_c_sigmaxx[ixLocal+1][izLocal]-shared_c_sigmaxx[ixLocal-2][izLocal])+
   dev_xCoeff[2]*(shared_c_sigmaxx[ixLocal+2][izLocal]-shared_c_sigmaxx[ixLocal-3][izLocal])+
   dev_xCoeff[3]*(shared_c_sigmaxx[ixLocal+3][izLocal]-shared_c_sigmaxx[ixLocal-4][izLocal]) +
    //first derivative in positive z direction of current sigmaxz
   dev_zCoeff[0]*(shared_c_sigmaxz[ixLocal][izLocal+1]-shared_c_sigmaxz[ixLocal][izLocal])  +
   dev_zCoeff[1]*(shared_c_sigmaxz[ixLocal][izLocal+2]-shared_c_sigmaxz[ixLocal][izLocal-1])+
   dev_zCoeff[2]*(shared_c_sigmaxz[ixLocal][izLocal+3]-shared_c_sigmaxz[ixLocal][izLocal-2])+
   dev_zCoeff[3]*(shared_c_sigmaxz[ixLocal][izLocal+4]-shared_c_sigmaxz[ixLocal][izLocal-3])
    );
  //new vz
  dev_n_vz[iGlobal] = //old vz
    dev_o_vz[iGlobal] +
    dev_rhozDtw[iGlobal] * (
    //first derivative in negative z direction of current sigmazz
   dev_zCoeff[0]*(shared_c_sigmazz[ixLocal][izLocal]-shared_c_sigmazz[ixLocal][izLocal-1])  +
   dev_zCoeff[1]*(shared_c_sigmazz[ixLocal][izLocal+1]-shared_c_sigmazz[ixLocal][izLocal-2])+
   dev_zCoeff[2]*(shared_c_sigmazz[ixLocal][izLocal+2]-shared_c_sigmazz[ixLocal][izLocal-3])+
   dev_zCoeff[3]*(shared_c_sigmazz[ixLocal][izLocal+3]-shared_c_sigmazz[ixLocal][izLocal-4])+
    //first derivative in positive x direction of current sigmaxz
   dev_xCoeff[0]*(shared_c_sigmaxz[ixLocal+1][izLocal]-shared_c_sigmaxz[ixLocal][izLocal])  +
   dev_xCoeff[1]*(shared_c_sigmaxz[ixLocal+2][izLocal]-shared_c_sigmaxz[ixLocal-1][izLocal])+
   dev_xCoeff[2]*(shared_c_sigmaxz[ixLocal+3][izLocal]-shared_c_sigmaxz[ixLocal-2][izLocal])+
   dev_xCoeff[3]*(shared_c_sigmaxz[ixLocal+4][izLocal]-shared_c_sigmaxz[ixLocal-3][izLocal])
    );
  //new sigmaxx
  dev_n_sigmaxx[iGlobal] = //old sigmaxx
     dev_o_sigmaxx[iGlobal] +
     //first deriv in positive x direction of current vx
     dev_lamb2MuDtw[iGlobal] * (
  dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
  dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
  dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
  dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal])
     ) +
     //first deriv in positive z direction of current vz
     dev_lambDtw[iGlobal] * (
  dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
  dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
  dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
  dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3])
     );
  //new sigmazz
  dev_n_sigmazz[iGlobal] = //old sigmazz
     dev_o_sigmazz[iGlobal] +
     //first deriv in positive x direction of current vx
     dev_lambDtw[iGlobal] * (
  dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
  dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
  dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
  dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal])
     ) +
     //first deriv in positive z direction of current vz
     dev_lamb2MuDtw[iGlobal] * (
  dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
  dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
  dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
  dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3])
     );
  //new sigmaxz
  dev_n_sigmaxz[iGlobal] = //old sigmaxz
     dev_o_sigmaxz[iGlobal] +
     dev_muxzDtw[iGlobal] * (
     //first deriv in negative z direction of current vx
  dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal][izLocal-1])  +
  dev_zCoeff[1]*(shared_c_vx[ixLocal][izLocal+1]-shared_c_vx[ixLocal][izLocal-2])+
  dev_zCoeff[2]*(shared_c_vx[ixLocal][izLocal+2]-shared_c_vx[ixLocal][izLocal-3])+
  dev_zCoeff[3]*(shared_c_vx[ixLocal][izLocal+3]-shared_c_vx[ixLocal][izLocal-4])+
     //first deriv in negative x direction of current vz
  dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
  dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
  dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
  dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal])
     );
}

/* kernel to compute adjoint time step */
__global__ void ker_step_adj(double* dev_o_vx, double* dev_o_vz, double* dev_o_sigmaxx, double* dev_o_sigmazz, double* dev_o_sigmaxz,
   double* dev_c_vx, double* dev_c_vz, double* dev_c_sigmaxx, double* dev_c_sigmazz, double* dev_c_sigmaxz,
   double* dev_n_vx, double* dev_n_vz, double* dev_n_sigmaxx, double* dev_n_sigmazz, double* dev_n_sigmaxz,
   double* dev_rhoxDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw){

  // Allocate shared memory for each SCALED wavefield component
  __shared__ double shared_c_vx_rhodtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vx*dtw/rhox
  __shared__ double shared_c_vz_rhodtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vz*dtw/rhoz
  __shared__ double shared_c_sigmaxx_lamb2MuDtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmaxx*dtw*(lamb+2Mu)
  __shared__ double shared_c_sigmaxx_lambDtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmaxx*dtw*(lamb)
  __shared__ double shared_c_sigmazz_lamb2MuDtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmazz*dtw*(lamb+2Mu)
  __shared__ double shared_c_sigmazz_lambDtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmazz*dtw*(lamb)
  __shared__ double shared_c_sigmaxz_muxzDtw[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // sigmaxz*dtw*muxz

  // calculate global and local x/z coordinates
  int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
  int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Copy current slice from global to shared memory ans scale appropriately for each wavefield component -- center
  shared_c_vx_rhodtw[ixLocal][izLocal]          = dev_rhoxDtw[iGlobal]*dev_c_vx[iGlobal]; // vx*dtw/rhox
  shared_c_vz_rhodtw[ixLocal][izLocal]          = dev_rhozDtw[iGlobal]*dev_c_vz[iGlobal]; // vz*dtw/rhoz
  shared_c_sigmaxx_lamb2MuDtw[ixLocal][izLocal] = dev_lamb2MuDtw[iGlobal]*dev_c_sigmaxx[iGlobal]; // sigmaxx*dtw*(lamb+2Mu)
  shared_c_sigmaxx_lambDtw[ixLocal][izLocal]    = dev_lambDtw[iGlobal]*dev_c_sigmaxx[iGlobal]; // sigmaxx*dtw*(lamb)
  shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal] = dev_lamb2MuDtw[iGlobal]*dev_c_sigmazz[iGlobal]; // sigmazz*dtw*(lamb+2Mu)
  shared_c_sigmazz_lambDtw[ixLocal][izLocal]    = dev_lambDtw[iGlobal]*dev_c_sigmazz[iGlobal]; // sigmazz*dtw*(lamb)
  shared_c_sigmaxz_muxzDtw[ixLocal][izLocal]    = dev_muxzDtw[iGlobal]*dev_c_sigmaxz[iGlobal]; // sigmaxz*dtw*muxz

  // Copy current slice from global to shared memory for each wavefield component -- edges
  if (threadIdx.y < FAT) {
  // vx*rho*dtw
  shared_c_vx_rhodtw[ixLocal-FAT][izLocal]        = dev_rhoxDtw[iGlobal-dev_nz*FAT]*dev_c_vx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vx_rhodtw[ixLocal+BLOCK_SIZE][izLocal] = dev_rhoxDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_vx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // vz*rho*dtw
  shared_c_vz_rhodtw[ixLocal-FAT][izLocal]         = dev_rhozDtw[iGlobal-dev_nz*FAT]*dev_c_vz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vz_rhodtw[ixLocal+BLOCK_SIZE][izLocal]  = dev_rhozDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_vz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxx*dtw*(lamb+2Mu)
  shared_c_sigmaxx_lamb2MuDtw[ixLocal-FAT][izLocal]         = dev_lamb2MuDtw[iGlobal-dev_nz*FAT]*dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxx_lamb2MuDtw[ixLocal+BLOCK_SIZE][izLocal]  = dev_lamb2MuDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxx*dtw*(lamb)
  shared_c_sigmaxx_lambDtw[ixLocal-FAT][izLocal]         = dev_lambDtw[iGlobal-dev_nz*FAT]*dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxx_lambDtw[ixLocal+BLOCK_SIZE][izLocal]  = dev_lambDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmazz*dtw*(lamb+2Mu)
  shared_c_sigmazz_lamb2MuDtw[ixLocal-FAT][izLocal]         = dev_lamb2MuDtw[iGlobal-dev_nz*FAT]*dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmazz_lamb2MuDtw[ixLocal+BLOCK_SIZE][izLocal]  = dev_lamb2MuDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmazz*dtw*(lamb)
  shared_c_sigmazz_lambDtw[ixLocal-FAT][izLocal]         = dev_lambDtw[iGlobal-dev_nz*FAT]*dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmazz_lambDtw[ixLocal+BLOCK_SIZE][izLocal]  = dev_lambDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxz
  shared_c_sigmaxz_muxzDtw[ixLocal-FAT][izLocal]        = dev_muxzDtw[iGlobal-dev_nz*FAT]*dev_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxz_muxzDtw[ixLocal+BLOCK_SIZE][izLocal] = dev_muxzDtw[iGlobal+dev_nz*BLOCK_SIZE]*dev_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE]; // Right side
  }
  if (threadIdx.x < FAT) {
  // vx*rho*dtw
  shared_c_vx_rhodtw[ixLocal][izLocal-FAT]        = dev_rhoxDtw[iGlobal-FAT]*dev_c_vx[iGlobal-FAT]; // Up
  shared_c_vx_rhodtw[ixLocal][izLocal+BLOCK_SIZE] = dev_rhoxDtw[iGlobal+BLOCK_SIZE]*dev_c_vx[iGlobal+BLOCK_SIZE]; // Down
  // vz*rho*dtw
  shared_c_vz_rhodtw[ixLocal][izLocal-FAT]        = dev_rhozDtw[iGlobal-FAT]*dev_c_vz[iGlobal-FAT]; // Up
  shared_c_vz_rhodtw[ixLocal][izLocal+BLOCK_SIZE] = dev_rhozDtw[iGlobal+BLOCK_SIZE]*dev_c_vz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxx*dtw*(lamb+2Mu)
  shared_c_sigmaxx_lamb2MuDtw[ixLocal][izLocal-FAT]         = dev_lamb2MuDtw[iGlobal-FAT]*dev_c_sigmaxx[iGlobal-FAT]; // Up
  shared_c_sigmaxx_lamb2MuDtw[ixLocal][izLocal+BLOCK_SIZE]  = dev_lamb2MuDtw[iGlobal+BLOCK_SIZE]*dev_c_sigmaxx[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxx*dtw*(lamb)
  shared_c_sigmaxx_lambDtw[ixLocal][izLocal-FAT]         = dev_lambDtw[iGlobal-FAT]*dev_c_sigmaxx[iGlobal-FAT]; // Up
  shared_c_sigmaxx_lambDtw[ixLocal][izLocal+BLOCK_SIZE]  = dev_lambDtw[iGlobal+BLOCK_SIZE]*dev_c_sigmaxx[iGlobal+BLOCK_SIZE]; // Down
  // sigmazz*dtw*(lamb+2Mu)
  shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal-FAT]         = dev_lamb2MuDtw[iGlobal-FAT]*dev_c_sigmazz[iGlobal-FAT]; // Up
  shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal+BLOCK_SIZE]  = dev_lamb2MuDtw[iGlobal+BLOCK_SIZE]*dev_c_sigmazz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxx*dtw*(lamb)
  shared_c_sigmazz_lambDtw[ixLocal][izLocal-FAT]         = dev_lambDtw[iGlobal-FAT]*dev_c_sigmazz[iGlobal-FAT]; // Up
  shared_c_sigmazz_lambDtw[ixLocal][izLocal+BLOCK_SIZE]  = dev_lambDtw[iGlobal+BLOCK_SIZE]*dev_c_sigmazz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxz
  shared_c_sigmaxz_muxzDtw[ixLocal][izLocal-FAT]        = dev_muxzDtw[iGlobal-FAT]*dev_c_sigmaxz[iGlobal-FAT]; // Up
  shared_c_sigmaxz_muxzDtw[ixLocal][izLocal+BLOCK_SIZE] = dev_muxzDtw[iGlobal+BLOCK_SIZE]*dev_c_sigmaxz[iGlobal+BLOCK_SIZE]; // Down
  }
  __syncthreads(); // Synchronise all threads within each block -- look new sync options

  //old vx
  dev_o_vx[iGlobal] = //new vx
    dev_n_vx[iGlobal] -
    //first derivative in negative x direction of current sigmaxx scaled by dtw*(lamb+2Mu)
    (dev_xCoeff[0]*(shared_c_sigmaxx_lamb2MuDtw[ixLocal][izLocal]-shared_c_sigmaxx_lamb2MuDtw[ixLocal-1][izLocal])  +
  dev_xCoeff[1]*(shared_c_sigmaxx_lamb2MuDtw[ixLocal+1][izLocal]-shared_c_sigmaxx_lamb2MuDtw[ixLocal-2][izLocal])+
  dev_xCoeff[2]*(shared_c_sigmaxx_lamb2MuDtw[ixLocal+2][izLocal]-shared_c_sigmaxx_lamb2MuDtw[ixLocal-3][izLocal])+
  dev_xCoeff[3]*(shared_c_sigmaxx_lamb2MuDtw[ixLocal+3][izLocal]-shared_c_sigmaxx_lamb2MuDtw[ixLocal-4][izLocal])) -
    //first derivative in negative x direction of current sigmazz scaled by dtw*(lamb)
    (dev_xCoeff[0]*(shared_c_sigmazz_lambDtw[ixLocal][izLocal]-shared_c_sigmazz_lambDtw[ixLocal-1][izLocal])  +
  dev_xCoeff[1]*(shared_c_sigmazz_lambDtw[ixLocal+1][izLocal]-shared_c_sigmazz_lambDtw[ixLocal-2][izLocal])+
  dev_xCoeff[2]*(shared_c_sigmazz_lambDtw[ixLocal+2][izLocal]-shared_c_sigmazz_lambDtw[ixLocal-3][izLocal])+
  dev_xCoeff[3]*(shared_c_sigmazz_lambDtw[ixLocal+3][izLocal]-shared_c_sigmazz_lambDtw[ixLocal-4][izLocal])) -
    //first derivative in positive z direction of current sigmaxz scaled by dtw*(muxz)
    (dev_zCoeff[0]*(shared_c_sigmaxz_muxzDtw[ixLocal][izLocal+1]-shared_c_sigmaxz_muxzDtw[ixLocal][izLocal])  +
  dev_zCoeff[1]*(shared_c_sigmaxz_muxzDtw[ixLocal][izLocal+2]-shared_c_sigmaxz_muxzDtw[ixLocal][izLocal-1])+
  dev_zCoeff[2]*(shared_c_sigmaxz_muxzDtw[ixLocal][izLocal+3]-shared_c_sigmaxz_muxzDtw[ixLocal][izLocal-2])+
  dev_zCoeff[3]*(shared_c_sigmaxz_muxzDtw[ixLocal][izLocal+4]-shared_c_sigmaxz_muxzDtw[ixLocal][izLocal-3]))
    ;
  //old vz
  dev_o_vz[iGlobal] = //new vz
    dev_n_vz[iGlobal] -
    //first derivative in negative z direction of current sigmaxx scaled by dtw*(lamb)
    (dev_zCoeff[0]*(shared_c_sigmaxx_lambDtw[ixLocal][izLocal]-shared_c_sigmaxx_lambDtw[ixLocal][izLocal-1])  +
  dev_zCoeff[1]*(shared_c_sigmaxx_lambDtw[ixLocal][izLocal+1]-shared_c_sigmaxx_lambDtw[ixLocal][izLocal-2])+
  dev_zCoeff[2]*(shared_c_sigmaxx_lambDtw[ixLocal][izLocal+2]-shared_c_sigmaxx_lambDtw[ixLocal][izLocal-3])+
  dev_zCoeff[3]*(shared_c_sigmaxx_lambDtw[ixLocal][izLocal+3]-shared_c_sigmaxx_lambDtw[ixLocal][izLocal-4])) -
    //first derivative in negative z direction of current sigmazz scaled by dtw*(lamb+2Mu)
    (dev_zCoeff[0]*(shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal]-shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal-1])  +
  dev_zCoeff[1]*(shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal+1]-shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal-2])+
  dev_zCoeff[2]*(shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal+2]-shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal-3])+
  dev_zCoeff[3]*(shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal+3]-shared_c_sigmazz_lamb2MuDtw[ixLocal][izLocal-4])) -
    //first derivative in positive x direction of current sigmaxz scaled by dtw*(muxz)
    (dev_xCoeff[0]*(shared_c_sigmaxz_muxzDtw[ixLocal+1][izLocal]-shared_c_sigmaxz_muxzDtw[ixLocal][izLocal])  +
  dev_xCoeff[1]*(shared_c_sigmaxz_muxzDtw[ixLocal+2][izLocal]-shared_c_sigmaxz_muxzDtw[ixLocal-1][izLocal])+
  dev_xCoeff[2]*(shared_c_sigmaxz_muxzDtw[ixLocal+3][izLocal]-shared_c_sigmaxz_muxzDtw[ixLocal-2][izLocal])+
  dev_xCoeff[3]*(shared_c_sigmaxz_muxzDtw[ixLocal+4][izLocal]-shared_c_sigmaxz_muxzDtw[ixLocal-3][izLocal]))
    ;
  //old sigmaxx
  dev_o_sigmaxx[iGlobal] = //new sigmaxx
     dev_n_sigmaxx[iGlobal] -
     //first deriv in positive x direction of current vx scaled by dtw/rhox
     (dev_xCoeff[0]*(shared_c_vx_rhodtw[ixLocal+1][izLocal]-shared_c_vx_rhodtw[ixLocal][izLocal])  +
   dev_xCoeff[1]*(shared_c_vx_rhodtw[ixLocal+2][izLocal]-shared_c_vx_rhodtw[ixLocal-1][izLocal])+
   dev_xCoeff[2]*(shared_c_vx_rhodtw[ixLocal+3][izLocal]-shared_c_vx_rhodtw[ixLocal-2][izLocal])+
   dev_xCoeff[3]*(shared_c_vx_rhodtw[ixLocal+4][izLocal]-shared_c_vx_rhodtw[ixLocal-3][izLocal]))
     ;
  //old sigmazz
  dev_o_sigmazz[iGlobal] = //new sigmazz
     dev_n_sigmazz[iGlobal] -
     //first deriv in positive z direction of current vz scaled by dtw/rhoz
     (dev_zCoeff[0]*(shared_c_vz_rhodtw[ixLocal][izLocal+1]-shared_c_vz_rhodtw[ixLocal][izLocal])  +
   dev_zCoeff[1]*(shared_c_vz_rhodtw[ixLocal][izLocal+2]-shared_c_vz_rhodtw[ixLocal][izLocal-1])+
   dev_zCoeff[2]*(shared_c_vz_rhodtw[ixLocal][izLocal+3]-shared_c_vz_rhodtw[ixLocal][izLocal-2])+
   dev_zCoeff[3]*(shared_c_vz_rhodtw[ixLocal][izLocal+4]-shared_c_vz_rhodtw[ixLocal][izLocal-3]))
     ;
  //old sigmaxz
  dev_o_sigmaxz[iGlobal] = //new sigmaxz
     dev_n_sigmaxz[iGlobal] -
     //first deriv in negative z direction of current vx scaled by dtw/rhox
     (dev_zCoeff[0]*(shared_c_vx_rhodtw[ixLocal][izLocal]-shared_c_vx_rhodtw[ixLocal][izLocal-1])  +
   dev_zCoeff[1]*(shared_c_vx_rhodtw[ixLocal][izLocal+1]-shared_c_vx_rhodtw[ixLocal][izLocal-2])+
   dev_zCoeff[2]*(shared_c_vx_rhodtw[ixLocal][izLocal+2]-shared_c_vx_rhodtw[ixLocal][izLocal-3])+
   dev_zCoeff[3]*(shared_c_vx_rhodtw[ixLocal][izLocal+3]-shared_c_vx_rhodtw[ixLocal][izLocal-4])) -
     //first deriv in negative x direction of current vz scaled by dtw/rhoz
     (dev_xCoeff[0]*(shared_c_vz_rhodtw[ixLocal][izLocal]-shared_c_vz_rhodtw[ixLocal-1][izLocal])  +
   dev_xCoeff[1]*(shared_c_vz_rhodtw[ixLocal+1][izLocal]-shared_c_vz_rhodtw[ixLocal-2][izLocal])+
   dev_xCoeff[2]*(shared_c_vz_rhodtw[ixLocal+2][izLocal]-shared_c_vz_rhodtw[ixLocal-3][izLocal])+
   dev_xCoeff[3]*(shared_c_vz_rhodtw[ixLocal+3][izLocal]-shared_c_vz_rhodtw[ixLocal-4][izLocal]))
     ;
}


/****************************************************************************************/
/*********************************** Free Surface  ***********************************/
/****************************************************************************************/
/* kernel to compute forward time step for free surface condition. Only steps at surface.
one thread for all x inside fat.
1. set new sigmaxx at z=0 to zero
2. make current sigmazz, sigmaxz odd about z=0 by changing values above z=0
3. set new vx at z=0
4. do not update sigmaxz at or above z=0
5. set new sigmaxx at z=0
6. set current vx and vz =0 above free surface*/
__global__ void ker_step_fwd_surface_surface(double* dev_o_vx, double* dev_o_vz, double* dev_o_sigmaxx, double* dev_o_sigmazz, double* dev_o_sigmaxz,
   double* dev_c_vx, double* dev_c_vz, double* dev_c_sigmaxx, double* dev_c_sigmazz, double* dev_c_sigmaxz,
   double* dev_n_vx, double* dev_n_vz, double* dev_n_sigmaxx, double* dev_n_sigmazz, double* dev_n_sigmaxz,
   double* dev_rhoxDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw){
   //double* dev_c_all,double* dev_n_all, double* dev_elastic_param_scaled) {

  // calculate global and local x/z coordinates
  int ixGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global x-coordinate
  int iGlobal_0 = dev_nz * ixGlobal + 0; // 1D array index for the model on the global memory
  int iGlobal_1 = dev_nz * ixGlobal + 1; // 1D array index for the model on the global memory
  int iGlobal_2 = dev_nz * ixGlobal + 2; // 1D array index for the model on the global memory
  int iGlobal_3 = dev_nz * ixGlobal + 3; // 1D array index for the model on the global memory
  int iGlobal_surf = dev_nz * ixGlobal + 4; // 1D array index for the model on the global memory

  // 1. set current sigmaxx at z=0 to zero
  dev_c_sigmazz[iGlobal_surf] = 0;

  // 2. make current sigmazz, sigmaxz odd about z=0 by changing values above z=0
  dev_c_sigmazz[iGlobal_0] = -dev_c_sigmazz[iGlobal_surf+4];
  dev_c_sigmazz[iGlobal_1] = -dev_c_sigmazz[iGlobal_surf+3];
  dev_c_sigmazz[iGlobal_2] = -dev_c_sigmazz[iGlobal_surf+2];
  dev_c_sigmazz[iGlobal_3] = -dev_c_sigmazz[iGlobal_surf+1];
  dev_c_sigmaxz[iGlobal_0] = -dev_c_sigmaxz[iGlobal_surf+4];
  dev_c_sigmaxz[iGlobal_1] = -dev_c_sigmaxz[iGlobal_surf+3];
  dev_c_sigmaxz[iGlobal_2] = -dev_c_sigmaxz[iGlobal_surf+2];
  dev_c_sigmaxz[iGlobal_3] = -dev_c_sigmaxz[iGlobal_surf+1];

  // 3. set new vx at z=0
  dev_n_vx[iGlobal_surf] = //old vx
  dev_o_vx[iGlobal_surf] +
  dev_rhoxDtw[iGlobal_surf] * (
  //first derivative in negative x direction of current sigmaxx
    dev_xCoeff[0]*(dev_c_sigmaxx[iGlobal_surf+0*dev_nz]-dev_c_sigmaxx[iGlobal_surf-1*dev_nz])+
  dev_xCoeff[1]*(dev_c_sigmaxx[iGlobal_surf+1*dev_nz]-dev_c_sigmaxx[iGlobal_surf-2*dev_nz])+
  dev_xCoeff[2]*(dev_c_sigmaxx[iGlobal_surf+2*dev_nz]-dev_c_sigmaxx[iGlobal_surf-3*dev_nz])+
  dev_xCoeff[3]*(dev_c_sigmaxx[iGlobal_surf+3*dev_nz]-dev_c_sigmaxx[iGlobal_surf-4*dev_nz]));

  // 4. do not update sigmaxz at or above z=0

  // 5. set new sigmaxx at z=0
  dev_n_sigmaxx[iGlobal_surf] = //old sigmaxx
   dev_o_sigmaxx[iGlobal_surf] +
   //first deriv in positive x direction of current vx
   (dev_lamb2MuDtw[iGlobal_surf] + (dev_lambDtw[iGlobal_surf]/dev_lamb2MuDtw[iGlobal_surf])*dev_lambDtw[iGlobal_surf])* (
    dev_xCoeff[0]*(dev_c_vx[iGlobal_surf+1*dev_nz]-dev_c_vx[iGlobal_surf-0*dev_nz])  +
    dev_xCoeff[1]*(dev_c_vx[iGlobal_surf+2*dev_nz]-dev_c_vx[iGlobal_surf-1*dev_nz])+
    dev_xCoeff[2]*(dev_c_vx[iGlobal_surf+3*dev_nz]-dev_c_vx[iGlobal_surf-2*dev_nz])+
    dev_xCoeff[3]*(dev_c_vx[iGlobal_surf+4*dev_nz]-dev_c_vx[iGlobal_surf-3*dev_nz])
  );
  // 6. set current vx and vz =0 above free surface*/
  dev_c_vx[iGlobal_0] = 0;
  dev_c_vx[iGlobal_1] = 0;
  dev_c_vx[iGlobal_2] = 0;
  dev_c_vx[iGlobal_3] = 0;
  dev_c_vz[iGlobal_0] = 0;
  dev_c_vz[iGlobal_1] = 0;
  dev_c_vz[iGlobal_2] = 0;
  dev_c_vz[iGlobal_3] = 0;

}

/* kernel to compute forward time step for free surface condition. Steps everywhere but surface.*/
__global__ void ker_step_fwd_surface_body(double* dev_o_vx, double* dev_o_vz, double* dev_o_sigmaxx, double* dev_o_sigmazz, double* dev_o_sigmaxz,
   double* dev_c_vx, double* dev_c_vz, double* dev_c_sigmaxx, double* dev_c_sigmazz, double* dev_c_sigmaxz,
   double* dev_n_vx, double* dev_n_vz, double* dev_n_sigmaxx, double* dev_n_sigmazz, double* dev_n_sigmaxz,
   double* dev_rhoxDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw){
   //double* dev_c_all,double* dev_n_all, double* dev_elastic_param_scaled) {

  // Allocate shared memory for each wavefield component
  __shared__ double shared_c_vx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+5+FAT]; // vx
  __shared__ double shared_c_vz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+5+FAT]; // vz
  __shared__ double shared_c_sigmaxx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+5+FAT]; // sigmaxx
  __shared__ double shared_c_sigmazz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+5+FAT]; // sigmazz
  __shared__ double shared_c_sigmaxz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+5+FAT]; // sigmaxz

  // calculate global and local x/z coordinates
  int izGlobal = 5 + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  int izLocal = 5 + threadIdx.x; // z-coordinate on the shared grid
  int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Copy current slice from global to shared memory for each wavefield component -- center
  shared_c_vx[ixLocal][izLocal] = dev_c_vx[iGlobal]; // vx
  shared_c_vz[ixLocal][izLocal] = dev_c_vz[iGlobal]; // vz
  shared_c_sigmaxx[ixLocal][izLocal] = dev_c_sigmaxx[iGlobal]; // sigmaxx
  shared_c_sigmazz[ixLocal][izLocal] = dev_c_sigmazz[iGlobal]; // sigmaxz
  shared_c_sigmaxz[ixLocal][izLocal] = dev_c_sigmaxz[iGlobal]; // sigmazz

  // Copy current slice from global to shared memory for each wavefield component -- edges
  if (threadIdx.y < FAT) {
  // vx
  shared_c_vx[ixLocal-FAT][izLocal] = dev_c_vx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vx[ixLocal+BLOCK_SIZE][izLocal] = dev_c_vx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // vz
  shared_c_vz[ixLocal-FAT][izLocal] = dev_c_vz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_vz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_vz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxx
  shared_c_sigmaxx[ixLocal-FAT][izLocal] = dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxx[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmazz
  shared_c_sigmazz[ixLocal-FAT][izLocal] = dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmazz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  // sigmaxz
  shared_c_sigmaxz[ixLocal-FAT][izLocal] = dev_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
  shared_c_sigmaxz[ixLocal+BLOCK_SIZE][izLocal] = dev_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
  }
  if (threadIdx.x < 5) {
  // vx
  shared_c_vx[ixLocal][izLocal-5] = dev_c_vx[iGlobal-5]; // Up
  shared_c_vx[ixLocal][izLocal+BLOCK_SIZE] = dev_c_vx[iGlobal+BLOCK_SIZE]; // Down
  // vz
  shared_c_vz[ixLocal][izLocal-5] = dev_c_vz[iGlobal-5]; // Up
  shared_c_vz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_vz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxx
  shared_c_sigmaxx[ixLocal][izLocal-5] = dev_c_sigmaxx[iGlobal-5]; // Up
  shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmaxx[iGlobal+BLOCK_SIZE]; // Down
  // sigmazz
  shared_c_sigmazz[ixLocal][izLocal-5] = dev_c_sigmazz[iGlobal-5]; // Up
  shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmazz[iGlobal+BLOCK_SIZE]; // Down
  // sigmaxz
  shared_c_sigmaxz[ixLocal][izLocal-5] = dev_c_sigmaxz[iGlobal-5]; // Up
  shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE] = dev_c_sigmaxz[iGlobal+BLOCK_SIZE]; // Down
  }
  __syncthreads(); // Synchronise all threads within each block -- look new sync options

  //new vx
  dev_n_vx[iGlobal] = //old vx
    dev_o_vx[iGlobal] +
    dev_rhoxDtw[iGlobal] * (
    //first derivative in negative x direction of current sigmaxx
  dev_xCoeff[0]*(shared_c_sigmaxx[ixLocal][izLocal]-shared_c_sigmaxx[ixLocal-1][izLocal])+
   	dev_xCoeff[1]*(shared_c_sigmaxx[ixLocal+1][izLocal]-shared_c_sigmaxx[ixLocal-2][izLocal])+
   	dev_xCoeff[2]*(shared_c_sigmaxx[ixLocal+2][izLocal]-shared_c_sigmaxx[ixLocal-3][izLocal])+
   	dev_xCoeff[3]*(shared_c_sigmaxx[ixLocal+3][izLocal]-shared_c_sigmaxx[ixLocal-4][izLocal]) +
    //first derivative in positive z direction of current sigmaxz
  dev_zCoeff[0]*(shared_c_sigmaxz[ixLocal][izLocal+1]-shared_c_sigmaxz[ixLocal][izLocal])  +
   	dev_zCoeff[1]*(shared_c_sigmaxz[ixLocal][izLocal+2]-shared_c_sigmaxz[ixLocal][izLocal-1])+
   	dev_zCoeff[2]*(shared_c_sigmaxz[ixLocal][izLocal+3]-shared_c_sigmaxz[ixLocal][izLocal-2])+
   	dev_zCoeff[3]*(shared_c_sigmaxz[ixLocal][izLocal+4]-shared_c_sigmaxz[ixLocal][izLocal-3])
    );
  //new vz
  dev_n_vz[iGlobal] = //old vz
    dev_o_vz[iGlobal] +
    dev_rhozDtw[iGlobal] * (
    //first derivative in negative z direction of current sigmazz
  dev_zCoeff[0]*(shared_c_sigmazz[ixLocal][izLocal]-shared_c_sigmazz[ixLocal][izLocal-1])  +
   	dev_zCoeff[1]*(shared_c_sigmazz[ixLocal][izLocal+1]-shared_c_sigmazz[ixLocal][izLocal-2])+
   	dev_zCoeff[2]*(shared_c_sigmazz[ixLocal][izLocal+2]-shared_c_sigmazz[ixLocal][izLocal-3])+
   	dev_zCoeff[3]*(shared_c_sigmazz[ixLocal][izLocal+3]-shared_c_sigmazz[ixLocal][izLocal-4])+
    //first derivative in positive x direction of current sigmaxz
  dev_xCoeff[0]*(shared_c_sigmaxz[ixLocal+1][izLocal]-shared_c_sigmaxz[ixLocal][izLocal])  +
   	dev_xCoeff[1]*(shared_c_sigmaxz[ixLocal+2][izLocal]-shared_c_sigmaxz[ixLocal-1][izLocal])+
   	dev_xCoeff[2]*(shared_c_sigmaxz[ixLocal+3][izLocal]-shared_c_sigmaxz[ixLocal-2][izLocal])+
   	dev_xCoeff[3]*(shared_c_sigmaxz[ixLocal+4][izLocal]-shared_c_sigmaxz[ixLocal-3][izLocal])
    );
  //new sigmaxx
  dev_n_sigmaxx[iGlobal] = //old sigmaxx
     dev_o_sigmaxx[iGlobal] +
     //first deriv in positive x direction of current vx
     dev_lamb2MuDtw[iGlobal] * (
  dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
  dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
  dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
  dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal])
     ) +
     //first deriv in positive z direction of current vz
     dev_lambDtw[iGlobal] * (
  dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
  dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
  dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
  dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3])
     );
  //new sigmazz
  dev_n_sigmazz[iGlobal] = //old sigmazz
     dev_o_sigmazz[iGlobal] +
     //first deriv in positive x direction of current vx
     dev_lambDtw[iGlobal] * (
     	dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
     	dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
     	dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
     	dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal])
     ) +
     //first deriv in positive z direction of current vz
     dev_lamb2MuDtw[iGlobal] * (
     	dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
     	dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
     	dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
     	dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3])
     );
  //new sigmaxz
  dev_n_sigmaxz[iGlobal] = //old sigmaxz
     dev_o_sigmaxz[iGlobal] +
     dev_muxzDtw[iGlobal] * (
     //first deriv in negative z direction of current vx
  dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal][izLocal-1])  +
  dev_zCoeff[1]*(shared_c_vx[ixLocal][izLocal+1]-shared_c_vx[ixLocal][izLocal-2])+
  dev_zCoeff[2]*(shared_c_vx[ixLocal][izLocal+2]-shared_c_vx[ixLocal][izLocal-3])+
  dev_zCoeff[3]*(shared_c_vx[ixLocal][izLocal+3]-shared_c_vx[ixLocal][izLocal-4])+
     //first deriv in negative x direction of current vz
  dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
  dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
  dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
  dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal])
     );
}
/****************************************************************************************/
/************************************** Damping *****************************************/
/****************************************************************************************/
__global__ void dampCosineEdge(double *dev_p1_vx, double *dev_p2_vx,
     double *dev_p1_vz, double *dev_p2_vz,
     double *dev_p1_sigmaxx, double *dev_p2_sigmaxx,
     double *dev_p1_sigmazz, double *dev_p2_sigmazz,
     double *dev_p1_sigmaxz, double *dev_p2_sigmaxz) {

  int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Compute distance to the closest edge of model
  int distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
  if (distToEdge < dev_minPad){

  // Compute damping coefficient
  double damp = dev_cosDampingCoeff[distToEdge];

  // Apply damping
  dev_p1_vx[iGlobal] *= damp;
  dev_p2_vx[iGlobal] *= damp;
  dev_p1_vz[iGlobal] *= damp;
  dev_p2_vz[iGlobal] *= damp;
  dev_p1_sigmaxx[iGlobal] *= damp;
  dev_p2_sigmaxx[iGlobal] *= damp;
  dev_p1_sigmazz[iGlobal] *= damp;
  dev_p2_sigmazz[iGlobal] *= damp;
  dev_p1_sigmaxz[iGlobal] *= damp;
  dev_p2_sigmaxz[iGlobal] *= damp;
  }
}

__global__ void dampCosineEdge_surface_surface(double *dev_p1_vx, double *dev_p2_vx,
     double *dev_p1_vz, double *dev_p2_vz,
     double *dev_p1_sigmaxx, double *dev_p2_sigmaxx,
     double *dev_p1_sigmazz, double *dev_p2_sigmazz,
     double *dev_p1_sigmaxz, double *dev_p2_sigmaxz) {

  int izGlobal = 4;
  int ixGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global x-coordinate
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Compute distance to the closest edge of model
  int distToEdge = min2(ixGlobal-FAT, dev_nx-ixGlobal-1-FAT);
  if (distToEdge < dev_minPad){

  // Compute damping coefficient
  double damp = dev_cosDampingCoeff[distToEdge];

  // Apply damping
  dev_p1_vx[iGlobal] *= damp;
  dev_p2_vx[iGlobal] *= damp;
  dev_p1_vz[iGlobal] *= damp;
  dev_p2_vz[iGlobal] *= damp;
  dev_p1_sigmaxx[iGlobal] *= damp;
  dev_p2_sigmaxx[iGlobal] *= damp;
  dev_p1_sigmazz[iGlobal] *= damp;
  dev_p2_sigmazz[iGlobal] *= damp;
  dev_p1_sigmaxz[iGlobal] *= damp;
  dev_p2_sigmaxz[iGlobal] *= damp;
  }
}
__global__ void dampCosineEdge_surface_body(double *dev_p1_vx, double *dev_p2_vx,
     double *dev_p1_vz, double *dev_p2_vz,
     double *dev_p1_sigmaxx, double *dev_p2_sigmaxx,
     double *dev_p1_sigmazz, double *dev_p2_sigmazz,
     double *dev_p1_sigmaxz, double *dev_p2_sigmaxz) {

  int izGlobal = 5 + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

  // Compute distance to the closest edge of model
  int distToEdge = min3(ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
  if (distToEdge < dev_minPad && izGlobal > 3){

  // Compute damping coefficient
  double damp = dev_cosDampingCoeff[distToEdge];

  // Apply damping
  dev_p1_vx[iGlobal] *= damp;
  dev_p2_vx[iGlobal] *= damp;
  dev_p1_vz[iGlobal] *= damp;
  dev_p2_vz[iGlobal] *= damp;
  dev_p1_sigmaxx[iGlobal] *= damp;
  dev_p2_sigmaxx[iGlobal] *= damp;
  dev_p1_sigmazz[iGlobal] *= damp;
  dev_p2_sigmazz[iGlobal] *= damp;
  dev_p1_sigmaxz[iGlobal] *= damp;
  dev_p2_sigmaxz[iGlobal] *= damp;
  }
}

/****************************************************************************************/
/************************************** Saving Wavefields *******************************/
/****************************************************************************************/

__global__ void interpWavefield(double* dev_wavefieldDts_all,
  double* dev_p0_vx, double* dev_p0_vz, double* dev_p0_sigmaxx, double* dev_p0_sigmazz, double* dev_p0_sigmaxz,
  int its,int it2){
  unsigned long long int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  unsigned long long int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  unsigned long long int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
  unsigned long long int t_slice = 5 * dev_nz * dev_nx;
  unsigned long long int pos = (t_slice * its) + iGlobal;
  //Vx its
  unsigned long long int iGlobalWavefield_vx_its = pos;
  //Vz its
  unsigned long long int iGlobalWavefield_vz_its = pos + (dev_nz * dev_nx);
  //SIGMAxx its
  unsigned long long int iGlobalWavefield_sigmaxx_its = pos + 2 * (dev_nz * dev_nx);
  //SIGMAzz its
  unsigned long long int iGlobalWavefield_sigmazz_its = pos + 3 * (dev_nz * dev_nx);
  //SIGMAxz its
  unsigned long long int iGlobalWavefield_sigmaxz_its = pos + 4 * (dev_nz * dev_nx);

  //interpolating Vx
  dev_wavefieldDts_all[iGlobalWavefield_vx_its] += dev_p0_vx[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_all[iGlobalWavefield_vx_its + t_slice] += dev_p0_vx[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating Vz
  dev_wavefieldDts_all[iGlobalWavefield_vz_its] += dev_p0_vz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_all[iGlobalWavefield_vz_its + t_slice] += dev_p0_vz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAxx
  dev_wavefieldDts_all[iGlobalWavefield_sigmaxx_its] += dev_p0_sigmaxx[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_all[iGlobalWavefield_sigmaxx_its + t_slice] += dev_p0_sigmaxx[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAzz
  dev_wavefieldDts_all[iGlobalWavefield_sigmazz_its] += dev_p0_sigmazz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_all[iGlobalWavefield_sigmazz_its + t_slice] += dev_p0_sigmazz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAxz
  dev_wavefieldDts_all[iGlobalWavefield_sigmaxz_its] += dev_p0_sigmaxz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_all[iGlobalWavefield_sigmaxz_its + t_slice] += dev_p0_sigmaxz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1
}

__global__ void interpWavefieldStreams(double* dev_wavefieldDts_left, double* dev_wavefieldDts_right,
  double* dev_p0_vx, double* dev_p0_vz, double* dev_p0_sigmaxx, double* dev_p0_sigmazz, double* dev_p0_sigmaxz,
  int its,int it2){
  unsigned long long int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  unsigned long long int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  unsigned long long int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
  //Vx its
  unsigned long long int iGlobalWavefield_vx_its = iGlobal;
  //Vz its
  unsigned long long int iGlobalWavefield_vz_its = iGlobal + (dev_nz * dev_nx);
  //SIGMAxx its
  unsigned long long int iGlobalWavefield_sigmaxx_its = iGlobal + 2 * (dev_nz * dev_nx);
  //SIGMAzz its
  unsigned long long int iGlobalWavefield_sigmazz_its = iGlobal + 3 * (dev_nz * dev_nx);
  //SIGMAxz its
  unsigned long long int iGlobalWavefield_sigmaxz_its = iGlobal + 4 * (dev_nz * dev_nx);

  //interpolating Vx
  dev_wavefieldDts_left[iGlobalWavefield_vx_its] += dev_p0_vx[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_right[iGlobalWavefield_vx_its] += dev_p0_vx[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating Vz
  dev_wavefieldDts_left[iGlobalWavefield_vz_its] += dev_p0_vz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_right[iGlobalWavefield_vz_its] += dev_p0_vz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAxx
  dev_wavefieldDts_left[iGlobalWavefield_sigmaxx_its] += dev_p0_sigmaxx[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_right[iGlobalWavefield_sigmaxx_its] += dev_p0_sigmaxx[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAzz
  dev_wavefieldDts_left[iGlobalWavefield_sigmazz_its] += dev_p0_sigmazz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_right[iGlobalWavefield_sigmazz_its] += dev_p0_sigmazz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

  //interpolating SIGMAxz
  dev_wavefieldDts_left[iGlobalWavefield_sigmaxz_its] += dev_p0_sigmaxz[iGlobal] * dev_interpFilter[it2]; // its
  dev_wavefieldDts_right[iGlobalWavefield_sigmaxz_its] += dev_p0_sigmaxz[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1
}


/****************************************************************************************/
/*************************** Elastic Scattering/Imaging *********************************/
/****************************************************************************************/
__global__ void imagingElaFwdGpu(double* dev_wavefieldVx, double* dev_wavefieldVz,
  double* dev_vx, double* dev_vz, double* dev_sigmaxx, double* dev_sigmazz, double* dev_sigmaxz, double* dev_drhox, double* dev_drhoz, double* dev_dlame, double* dev_dmu, double* dev_dmuxz, int its){
  //Index definition
  unsigned long long int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  unsigned long long int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  unsigned long long int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
  unsigned long long int iGlobal_cur = iGlobal + its * (dev_nz * dev_nx); // 1D array index for the model on the global memory (its-1)
  unsigned long long int iGlobal_old = iGlobal_cur - (dev_nz * dev_nx); // 1D array index for the model on the global memory (its)
  unsigned long long int iGlobal_new = iGlobal_cur + (dev_nz * dev_nx); // 1D array index for the model on the global memory (its+1)
  int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
  int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid

  // Allocate shared memory for each wavefield component
  __shared__ double shared_c_vx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vx
  __shared__ double shared_c_vz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vz

  // Copy current slice from global to shared memory for each wavefield component -- center
  shared_c_vx[ixLocal][izLocal] = dev_wavefieldVx[iGlobal_cur]; // vx
  shared_c_vz[ixLocal][izLocal] = dev_wavefieldVz[iGlobal_cur]; // vz

  // Copy current slice from global to shared memory for each wavefield component -- edges
  if (threadIdx.y < FAT) {
  // vx
    shared_c_vx[ixLocal-FAT][izLocal] = dev_wavefieldVx[iGlobal_cur-dev_nz*FAT]; // Left side
    shared_c_vx[ixLocal+BLOCK_SIZE][izLocal] = dev_wavefieldVx[iGlobal_cur+dev_nz*BLOCK_SIZE] ; // Right side
  // vz
    shared_c_vz[ixLocal-FAT][izLocal] = dev_wavefieldVz[iGlobal_cur-dev_nz*FAT]; // Left side
    shared_c_vz[ixLocal+BLOCK_SIZE][izLocal] = dev_wavefieldVz[iGlobal_cur+dev_nz*BLOCK_SIZE] ; // Right side
  }
  if (threadIdx.x < 5) {
  // vx
    shared_c_vx[ixLocal][izLocal-5] = dev_wavefieldVx[iGlobal_cur-5]; // Up
    shared_c_vx[ixLocal][izLocal+BLOCK_SIZE] = dev_wavefieldVx[iGlobal_cur+BLOCK_SIZE]; // Down
  // vz
    shared_c_vz[ixLocal][izLocal-5] = dev_wavefieldVz[iGlobal_cur-5]; // Up
    shared_c_vz[ixLocal][izLocal+BLOCK_SIZE] = dev_wavefieldVz[iGlobal_cur+BLOCK_SIZE]; // Down
  }
  __syncthreads(); // Synchronise all threads within each block -- look new sync options

  //Computing dVx/dx (forward staggering)
  double dvx_dx = dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
  		   		  		dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
  		   		  		dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
  		   		  		dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal]);
  //Computing dVz/dz (forward staggering)
  double dvz_dz = dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
    				      dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
     		  			  dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
     		  			  dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3]);

  //Note we assume zero wavefield for its < 0 and its > ntw
  //Scattering Vx and Vz components (- drho * dvx/dt and - drho * dvz/dt)
  if(its == 0){
    dev_vx[iGlobal] = dev_drhox[iGlobal] * (- dev_wavefieldVx[iGlobal_new])*dev_dts_inv;
    dev_vz[iGlobal] = dev_drhoz[iGlobal] * (- dev_wavefieldVz[iGlobal_new])*dev_dts_inv;
  } else if(its == dev_nts){
    dev_vx[iGlobal] = dev_drhox[iGlobal] * (dev_wavefieldVx[iGlobal_old])*dev_dts_inv;
    dev_vz[iGlobal] = dev_drhoz[iGlobal] * (dev_wavefieldVz[iGlobal_old])*dev_dts_inv;
  } else {
    dev_vx[iGlobal] = dev_drhox[iGlobal] * (dev_wavefieldVx[iGlobal_old] - dev_wavefieldVx[iGlobal_new])*dev_dts_inv;
    dev_vz[iGlobal] = dev_drhoz[iGlobal] * (dev_wavefieldVz[iGlobal_old] - dev_wavefieldVz[iGlobal_new])*dev_dts_inv;
  }
  //Scattering Sigmaxx component
  dev_sigmaxx[iGlobal] = dev_dlame[iGlobal] * (dvx_dx + dvz_dz) + 2.0 * dev_dmu[iGlobal] * dvx_dx;
  //Scattering Sigmazz component
  dev_sigmazz[iGlobal] = dev_dlame[iGlobal] * (dvx_dx + dvz_dz) + 2.0 * dev_dmu[iGlobal] * dvz_dz;
  //Scattering Sigmaxz component             //first deriv in negative z direction of current vx
  dev_sigmaxz[iGlobal] = dev_dmuxz[iGlobal]*(dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal][izLocal-1])  +
                                             dev_zCoeff[1]*(shared_c_vx[ixLocal][izLocal+1]-shared_c_vx[ixLocal][izLocal-2])+
                                             dev_zCoeff[2]*(shared_c_vx[ixLocal][izLocal+2]-shared_c_vx[ixLocal][izLocal-3])+
                                             dev_zCoeff[3]*(shared_c_vx[ixLocal][izLocal+3]-shared_c_vx[ixLocal][izLocal-4])+
                                             //first deriv in negative x direction of current vz
                                             dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
                                             dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
                                             dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
                                             dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal]));


}

__global__ void imagingElaAdjGpu(double* dev_wavefieldVx, double* dev_wavefieldVz,
  double* dev_vx, double* dev_vz, double* dev_sigmaxx, double* dev_sigmazz, double* dev_sigmaxz, double* dev_drhox, double* dev_drhoz, double* dev_dlame, double* dev_dmu, double* dev_dmuxz, int its){
  //Index definition
  unsigned long long int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
  unsigned long long int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
  unsigned long long int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
  unsigned long long int iGlobal_cur = iGlobal + its * (dev_nz * dev_nx); // 1D array index for the model on the global memory (its-1)
  unsigned long long int iGlobal_old = iGlobal_cur - (dev_nz * dev_nx); // 1D array index for the model on the global memory (its)
  unsigned long long int iGlobal_new = iGlobal_cur + (dev_nz * dev_nx); // 1D array index for the model on the global memory (its+1)
  int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
  int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid

  // Allocate shared memory for each wavefield component
  __shared__ double shared_c_vx[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vx
  __shared__ double shared_c_vz[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // vz

  // Copy current slice from global to shared memory for each wavefield component -- center
  shared_c_vx[ixLocal][izLocal] = dev_wavefieldVx[iGlobal_cur]; // vx
  shared_c_vz[ixLocal][izLocal] = dev_wavefieldVz[iGlobal_cur]; // vz

  // Copy current slice from global to shared memory for each wavefield component -- edges
  if (threadIdx.y < FAT) {
  // vx
    shared_c_vx[ixLocal-FAT][izLocal] = dev_wavefieldVx[iGlobal_cur-dev_nz*FAT]; // Left side
    shared_c_vx[ixLocal+BLOCK_SIZE][izLocal] = dev_wavefieldVx[iGlobal_cur+dev_nz*BLOCK_SIZE] ; // Right side
  // vz
    shared_c_vz[ixLocal-FAT][izLocal] = dev_wavefieldVz[iGlobal_cur-dev_nz*FAT]; // Left side
    shared_c_vz[ixLocal+BLOCK_SIZE][izLocal] = dev_wavefieldVz[iGlobal_cur+dev_nz*BLOCK_SIZE] ; // Right side
  }
  if (threadIdx.x < 5) {
  // vx
    shared_c_vx[ixLocal][izLocal-5] = dev_wavefieldVx[iGlobal_cur-5]; // Up
    shared_c_vx[ixLocal][izLocal+BLOCK_SIZE] = dev_wavefieldVx[iGlobal_cur+BLOCK_SIZE]; // Down
  // vz
    shared_c_vz[ixLocal][izLocal-5] = dev_wavefieldVz[iGlobal_cur-5]; // Up
    shared_c_vz[ixLocal][izLocal+BLOCK_SIZE] = dev_wavefieldVz[iGlobal_cur+BLOCK_SIZE]; // Down
  }
  __syncthreads(); // Synchronise all threads within each block -- look new sync options

  //Computing dVx/dx (forward staggering)
  double dvx_dx = dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
  		   		  		dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
  		   		  		dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
  		   		  		dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal]);
  //Computing dVz/dz (forward staggering)
  double dvz_dz = dev_zCoeff[0]*(shared_c_vz[ixLocal][izLocal+1]-shared_c_vz[ixLocal][izLocal])  +
    				      dev_zCoeff[1]*(shared_c_vz[ixLocal][izLocal+2]-shared_c_vz[ixLocal][izLocal-1])+
     		  			  dev_zCoeff[2]*(shared_c_vz[ixLocal][izLocal+3]-shared_c_vz[ixLocal][izLocal-2])+
     		  			  dev_zCoeff[3]*(shared_c_vz[ixLocal][izLocal+4]-shared_c_vz[ixLocal][izLocal-3]);

  //Imaging drhox and drhoz components
  if(its == 0){
    dev_drhox[iGlobal] += dev_vx[iGlobal] * (- dev_wavefieldVx[iGlobal_new])*dev_dts_inv;
    dev_drhoz[iGlobal] += dev_vz[iGlobal] * (- dev_wavefieldVz[iGlobal_new])*dev_dts_inv;
  } else if(its == dev_nts){
    dev_drhox[iGlobal] += dev_vx[iGlobal] * (dev_wavefieldVx[iGlobal_old])*dev_dts_inv;
    dev_drhoz[iGlobal] += dev_vz[iGlobal] * (dev_wavefieldVz[iGlobal_old])*dev_dts_inv;
  } else {
    dev_drhox[iGlobal] += dev_vx[iGlobal] * (dev_wavefieldVx[iGlobal_old] - dev_wavefieldVx[iGlobal_new])*dev_dts_inv;
    dev_drhoz[iGlobal] += dev_vz[iGlobal] * (dev_wavefieldVz[iGlobal_old] - dev_wavefieldVz[iGlobal_new])*dev_dts_inv;
  }
  //Imaging dlame component
  dev_dlame[iGlobal] += (dev_sigmaxx[iGlobal] + dev_sigmazz[iGlobal]) * (dvx_dx + dvz_dz);
  //Imaging dmu component
  dev_dmu[iGlobal] += 2.0 * (dvx_dx * dev_sigmaxx[iGlobal] + dvz_dz * dev_sigmazz[iGlobal]);
  //Scattering Sigmaxz component             //first deriv in negative z direction of current vx
  dev_dmuxz[iGlobal] += dev_sigmaxz[iGlobal]*(dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal][izLocal-1])  +
                                             dev_zCoeff[1]*(shared_c_vx[ixLocal][izLocal+1]-shared_c_vx[ixLocal][izLocal-2])+
                                             dev_zCoeff[2]*(shared_c_vx[ixLocal][izLocal+2]-shared_c_vx[ixLocal][izLocal-3])+
                                             dev_zCoeff[3]*(shared_c_vx[ixLocal][izLocal+3]-shared_c_vx[ixLocal][izLocal-4])+
                                             //first deriv in negative x direction of current vz
                                             dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
                                             dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
                                             dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
                                             dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal]));
}
