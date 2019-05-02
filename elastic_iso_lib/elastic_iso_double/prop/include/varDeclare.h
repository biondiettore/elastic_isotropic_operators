#ifndef VAR_DECLARE_H
#define VAR_DECLARE_H 1

#include <math.h>
#define BLOCK_SIZE 16
#define BLOCK_SIZE_DATA 128
#define BLOCK_SIZE_EXT 8
#define FAT 4
#define COEFF_SIZE 4 // Laplacian coefficient array for 10th order
#define PI_CUDA M_PI // Import the number "Pi" from the math library
#define PAD_MAX 200 // Maximum number of points for padding (on one side)
#define SUB_MAX 50 // Maximum subsampling value

#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */

#if __CUDACC__
/************************************* DEVICE DECLARATION *******************************/
// Device function
__device__ int min3(int v1,int v2,int v3){return min2(v1,min2(v2,v3));}
__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

// Constant memory variables
__constant__ double dev_zCoeff[COEFF_SIZE]; // 8th-order Laplacian coefficients on Device
__constant__ double dev_xCoeff[COEFF_SIZE];

__constant__ int dev_nInterpFilter; // Time interpolation filter length
__constant__ int dev_hInterpFilter; // Time interpolation filter half-length
__constant__ double dev_interpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
__constant__ int dev_nz; // nz on Device
__constant__ int dev_nx; // nx on Device
__constant__ int dev_nep; // number of elastic parameters on Device
__constant__ int dev_sub; // Subsampling in time
 __constant__ int dev_nExt; // Length of extension axis
 __constant__ int dev_hExt; // Half-length of extension axis

 __constant__ int dev_nSourcesRegCenterGrid; // Nb of source grid points on center grid
 __constant__ int dev_nSourcesRegXGrid; // Nb of source grid points on x shifted grid
 __constant__ int dev_nSourcesRegZGrid; // Nb of source grid points on z shifted grid
 __constant__ int dev_nSourcesRegXZGrid; // Nb of source grid points on xz shifted grid

__constant__ int dev_nReceiversRegCenterGrid; // Nb of receiver grid points on center grid
__constant__ int dev_nReceiversRegXGrid; // Nb of receiver grid points on z shifted grid
__constant__ int dev_nReceiversRegZGrid; // Nb of receiver grid points on x shifted grid
__constant__ int dev_nReceiversRegXZGrid; // Nb of receiver grid points on xz shifted grid

__constant__ double dev_alphaCos; // Decay coefficient
__constant__ int dev_minPad; // Minimum padding length
__constant__ double dev_cosDampingCoeff[PAD_MAX]; // Padding array
__constant__ double dev_cSide;
__constant__ double dev_cCenter;

// Global memory variables
int **dev_sourcesPositionRegCenterGrid, **dev_sourcesPositionRegXGrid, **dev_sourcesPositionRegZGrid, **dev_sourcesPositionRegXZGrid; // Array containing the positions of the sources on the regular grid
int **dev_receiversPositionRegCenterGrid,**dev_receiversPositionRegXGrid,
**dev_receiversPositionRegZGrid,**dev_receiversPositionRegXZGrid; // Array containing the positions of the receivers on the regular grid
double **dev_p0_vx, **dev_p0_vz, **dev_p0_sigmaxx, **dev_p0_sigmazz, **dev_p0_sigmaxz; // Temporary slices for stepping
double **dev_p1_vx, **dev_p1_vz, **dev_p1_sigmaxx, **dev_p1_sigmazz, **dev_p1_sigmaxz; // Temporary slices for stepping
double **dev_temp1; // Temporary slices for stepping
// double **dev_ss0, **dev_ss1, **dev_ss2, **dev_ssTemp2;
// double **dev_ssLeft, **dev_ssRight, **dev_ssTemp1; // Temporary slices for secondary source
// double **dev_scatLeft, **dev_scatRight, **dev_scatTemp1; // Temporary slices for scattered wavefield (used in tomo)
double **dev_modelRegDtw_vx, **dev_modelRegDtw_vz, **dev_modelRegDtw_sigmaxx, **dev_modelRegDtw_sigmazz, **dev_modelRegDtw_sigmaxz; // Model for nonlinear propagation (wavelet)
double **dev_dataRegDts_vx, **dev_dataRegDts_vz, **dev_dataRegDts_sigmaxx, **dev_dataRegDts_sigmazz, **dev_dataRegDts_sigmaxz; // Data on device at coarse time-sampling (converted to regular grid)
// double **dev_interpFilterTime; // Time interpolation filter (second order) to interpolate wavefields and data as we propagate
double *dev_wavefieldDts_all;
// double **dev_wavefieldDts_vx,**dev_wavefieldDts_vz,**dev_wavefieldDts_sigmaxx,**dev_wavefieldDts_sigmazz,**dev_wavefieldDts_sigmaxz
// , **dev_BornSrcWavefield, *dev_BornSecWavefield;
double **dev_tomoSrcWavefieldDt2, **dev_tomoScatWavefield1, **dev_tomoScatWavefield2, **dev_tomoRecWavefield;
double **dev_sourcesSignals; // Sources for Born modeling
//double **dev_vel2Dtw2; // Precomputed scaling v^2 * dtw^2
double **dev_rhoxDtw; // Precomputed scaling dtw / rho_x
double **dev_rhozDtw; // Precomputed scaling dtw / rho_z
double **dev_lamb2MuDtw; // Precomputed scaling (lambda + 2*mu) * dtw
double **dev_lambDtw; // Precomputed scaling lambda * dtw
double **dev_muxzDtw; // Precomputed scaling mu_xz * dtw
double **dev_reflectivityScale; // scale = -2.0 / (vel*vel*vel)
double **dev_modelBorn, **dev_modelBornExt; // Reflectivity model for Born / Born extended
double **dev_modelTomo;  // Model for tomo
double **dev_extReflectivity; // Extended reflectivity for tomo
cudaStream_t *stream1, *stream2; // Streams


/************************************* HOST DECLARATION *********************************/
long long host_nz; // Includes padding + FAT
long long host_nx;
double host_dz;
double host_dx;
int host_nts;
double host_dts;
int host_ntw;
int host_sub;
int host_nExt; // Length of extended axis
int host_hExt; // Half-length of extended axis
double host_cSide, host_cCenter; // Coefficients for the second-order time derivative
int host_leg1, host_leg2; // Flags to indicate which legs to compute in tomo and wemva

#endif

#endif
