#ifndef VAR_DECLARE_H
#define VAR_DECLARE_H 1

#include <math.h>
#define BLOCK_SIZE 16
#define BLOCK_SIZE_DATA 128
#define BLOCK_SIZE_EXT 8
#define FAT 4
#define COEFF_SIZE 4 // Derivative coefficient array for 8th order
#define PI_CUDA M_PI // Import the number "Pi" from the math library
#define PAD_MAX 500 // Maximum number of points for padding (on one side)
#define SUB_MAX 100 // Maximum subsampling value

#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */

#if __CUDACC__
	/************************************* DEVICE DECLARATION *******************************/
	// Device function
	__device__ int min3(int v1,int v2,int v3){return min2(v1,min2(v2,v3));}
	__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

	// Constant memory variables
	__constant__ float dev_zCoeff[COEFF_SIZE]; // 8th-order Laplacian coefficients on Device
	__constant__ float dev_xCoeff[COEFF_SIZE];

	__constant__ int dev_nInterpFilter; // Time interpolation filter length
	__constant__ int dev_hInterpFilter; // Time interpolation filter half-length
	__constant__ float dev_interpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

	__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
	__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
	__constant__ int dev_nz; // nz on Device
	__constant__ int dev_nx; // nx on Device
	__constant__ int dev_nep; // number of elastic parameters on Device
	__constant__ int dev_sub; // Subsampling in time
	__constant__ float dev_dts_inv; // 1/dts for computing time derivative on device
	__constant__ float dev_dtw; // dtw

	__constant__ int dev_nSourcesRegCenterGrid; // Nb of source grid points on center grid
	__constant__ int dev_nSourcesRegXGrid; // Nb of source grid points on x shifted grid
	__constant__ int dev_nSourcesRegZGrid; // Nb of source grid points on z shifted grid
	__constant__ int dev_nSourcesRegXZGrid; // Nb of source grid points on xz shifted grid

	__constant__ int dev_nReceiversRegCenterGrid; // Nb of receiver grid points on center grid
	__constant__ int dev_nReceiversRegXGrid; // Nb of receiver grid points on z shifted grid
	__constant__ int dev_nReceiversRegZGrid; // Nb of receiver grid points on x shifted grid
	__constant__ int dev_nReceiversRegXZGrid; // Nb of receiver grid points on xz shifted grid

	__constant__ float dev_alphaCos; // Decay coefficient
	__constant__ int dev_minPad; // Minimum padding length
	__constant__ float dev_cosDampingCoeff[PAD_MAX]; // Padding array
	__constant__ float dev_cSide;
	__constant__ float dev_cCenter;

	// Global memory variables
	int **dev_sourcesPositionRegCenterGrid, **dev_sourcesPositionRegXGrid, **dev_sourcesPositionRegZGrid, **dev_sourcesPositionRegXZGrid; // Array containing the positions of the sources on the regular grid
	int **dev_receiversPositionRegCenterGrid,**dev_receiversPositionRegXGrid,
	**dev_receiversPositionRegZGrid,**dev_receiversPositionRegXZGrid; // Array containing the positions of the receivers on the regular grid
	float **dev_p0_vx, **dev_p0_vz, **dev_p0_sigmaxx, **dev_p0_sigmazz, **dev_p0_sigmaxz; // Temporary slices for stepping
	float **dev_p1_vx, **dev_p1_vz, **dev_p1_sigmaxx, **dev_p1_sigmazz, **dev_p1_sigmaxz; // Temporary slices for stepping
	float **dev_temp1; // Temporary slices for stepping
	// float **dev_ss0, **dev_ss1, **dev_ss2, **dev_ssTemp2;
	// float **dev_ssLeft, **dev_ssRight, **dev_ssTemp1; // Temporary slices for secondary source
	float **dev_modelRegDtw_vx, **dev_modelRegDtw_vz, **dev_modelRegDtw_sigmaxx, **dev_modelRegDtw_sigmazz, **dev_modelRegDtw_sigmaxz; // Model for nonlinear propagation (wavelet)
	float **dev_dataRegDts_vx, **dev_dataRegDts_vz, **dev_dataRegDts_sigmaxx, **dev_dataRegDts_sigmazz, **dev_dataRegDts_sigmaxz; // Data on device at coarse time-sampling (converted to regular grid)
	// float **dev_interpFilterTime; // Time interpolation filter (second order) to interpolate wavefields and data as we propagate
	float *dev_wavefieldDts_all;
	float **dev_wavefieldDts_left, **dev_wavefieldDts_right, **dev_pStream, **dev_pStream_Vx, **dev_pStream_Vz; //Left, right, and temp time slices for saving the wavefield using streams
	float **pin_wavefieldSlice, **pin_wavefieldSlice_Vx, **pin_wavefieldSlice_Vz; //Pinnned memory to allow ansync memory copy
	// float **dev_wavefieldDts_vx,**dev_wavefieldDts_vz,**dev_wavefieldDts_sigmaxx,**dev_wavefieldDts_sigmazz,**dev_wavefieldDts_sigmaxz
	// , **dev_BornSrcWavefield, *dev_BornSecWavefield;
	float **dev_sourcesSignals; // Sources for Born modeling

	float **dev_rhoxDtw; // Precomputed scaling dtw / rho_x
	float **dev_rhozDtw; // Precomputed scaling dtw / rho_z
	float **dev_lamb2MuDtw; // Precomputed scaling (lambda + 2*mu) * dtw
	float **dev_lambDtw; // Precomputed scaling lambda * dtw
	float **dev_muxzDtw; // Precomputed scaling mu_xz * dtw

	//Variables for Born operator
	float **dev_ssVxLeft, **dev_ssVxRight, **dev_ssVzLeft, **dev_ssVzRight, **dev_ssSigmaxxLeft, **dev_ssSigmaxxRight, **dev_ssSigmazzLeft, **dev_ssSigmazzRight, **dev_ssSigmaxzLeft, **dev_ssSigmaxzRight, **dev_ssTemp1; // Temporary slices for secondary source
	float **dev_drhox, **dev_drhoz, **dev_dlame, **dev_dmu, **dev_dmuxz; //model perturbations
	float **dev_wavefieldVx, **dev_wavefieldVz; //Vx and Vz wavefields
	float **dev_wavefieldVx_left, **dev_wavefieldVx_right, **dev_wavefieldVz_left, **dev_wavefieldVz_right, **dev_wavefieldVx_cur, **dev_wavefieldVz_cur; //Vx and Vz wavefields slices for Streams
	float **dev_sourceRegDtw_vx, **dev_sourceRegDtw_vz, **dev_sourceRegDtw_sigmaxx, **dev_sourceRegDtw_sigmazz, **dev_sourceRegDtw_sigmaxz; // Source terms

	//Streams declaration
	cudaStream_t *compStream, *transferStream;


	/************************************* HOST DECLARATION *********************************/
	long long host_nz; // Includes padding + FAT
	long long host_nx;
	float host_dz;
	float host_dx;
	int host_nts;
	float host_dts;
	int host_ntw;
	int host_sub;
	float host_cSide, host_cCenter; // Coefficients for the second-order time derivativexw

	// wavefield pointers for stream-based operators
	float **host_wavefieldVx, **host_wavefieldVz;

#endif

#endif
