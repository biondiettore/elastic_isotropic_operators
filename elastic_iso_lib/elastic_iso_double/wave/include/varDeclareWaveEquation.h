#ifndef VAR_DECLARE_WAVE_EQUATION_H
#define VAR_DECLARE_WAVE_EQUATION_H 1

#include <math.h>
#define BLOCK_SIZE 8
#define FAT 4
#define COEFF_SIZE 4


/*************************** Constant memory variable *************************/

__constant__ double dev_zCoeff[COEFF_SIZE]; // 8th-order Laplacian coefficients on Device
__constant__ double dev_xCoeff[COEFF_SIZE];

__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
__constant__ int dev_nz; // nz on Device
__constant__ int dev_nx; // nx on Device
__constant__ int dev_nw; // nw on Device
__constant__ double dev_dx; // dx on Device
__constant__ double dev_dz; // dz on Device
__constant__ double dev_dts; // dz on Device

/****************************** Device Pointers *******************************/
double **dev_p0, **dev_p1;
double **dev_rhox; // Precomputed scaling dtw / rho_x
double **dev_rhoz; // Precomputed scaling dtw / rho_z
double **dev_lamb2Mu; // Precomputed scaling (lambda + 2*mu) * dtw
double **dev_lamb; // Precomputed scaling lambda * dtw
double **dev_muxz; // Precomputed scaling mu_xz * dtw

/************************************* HOST DECLARATION *********************************/
long long host_nz; // Includes padding + FAT
long long host_nx;
const long long host_nw = 5; // number of wavefields
double host_dz;
double host_dx;
int host_nts;
double host_dts;

#endif
