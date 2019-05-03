#include "varDeclareWaveEquation.h"
#include <stdio.h>

/****************************************************************************************/
/*********************************** Forward kernel ***********************************/
/****************************************************************************************/
/* kernel to compute forward wabe equation */
__global__ void ker_we_fwd(float* dev_p0, float* dev_p1,
                             float* dev_rhox, float* dev_rhoz, float* dev_lamb2Mu, float* dev_lamb, float* dev_muxz,
                            int absoluteFirstTimeSampleForBlock, int absoluteLastTimeSampleForBlock){
                             //float* dev_c_all,float* dev_n_all, float* dev_elastic_param_scaled) {

    // calculate global and local x/z/t coordinates
    int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int itInTimeBlock = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int itGlobal = absoluteFirstTimeSampleForBlock + itInTimeBlock;
    int iGlobal = dev_nz*ixGlobal + izGlobal;

    int iGlobal_vx_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vx_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vx_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;

    int iGlobal_vz_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vz_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vz_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmaxx_prev= dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 2 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxx_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 2 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxx_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 2 + dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmazz_prev= dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmazz_cur= dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmazz_next= dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmaxz_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxz_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxz_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;


    //t=0 boundary condition.
    if(itGlobal==0){
      dev_p0[iGlobal_vx_cur] += dev_rhox[iGlobal]*dev_p1[iGlobal_vx_next]
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxx_cur]-dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1]-dev_p1[iGlobal_sigmaxz_cur])+
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2]-dev_p1[iGlobal_sigmaxz_cur-1])+
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3]-dev_p1[iGlobal_sigmaxz_cur-2])+
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4]-dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_next]
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmazz_cur]-dev_p1[iGlobal_sigmazz_cur-1])+
                                   dev_zCoeff[1]*(dev_p1[iGlobal_sigmazz_cur+1]-dev_p1[iGlobal_sigmazz_cur-2])+
                                   dev_zCoeff[2]*(dev_p1[iGlobal_sigmazz_cur+2]-dev_p1[iGlobal_sigmazz_cur-3])+
                                   dev_zCoeff[3]*(dev_p1[iGlobal_sigmazz_cur+3]-dev_p1[iGlobal_sigmazz_cur-4]))
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxz_cur])+
                                   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                                   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                                   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += dev_p1[iGlobal_sigmaxx_next]/(2*dev_dts)
            - dev_lamb2Mu[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
               - dev_lamb[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmazz_cur] += dev_p1[iGlobal_sigmazz_next]/(2*dev_dts)
               - dev_lamb[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
            - dev_lamb2Mu[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += dev_p1[iGlobal_sigmaxz_next]/(2*dev_dts)
               - dev_muxz[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
               - dev_muxz[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                                  	dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                                  	dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                                  	dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }
    else if(itInTimeBlock==0){
      //do nothing in this case.
    }
    //t=nt-1 boundary condition
    else if(itGlobal==dev_nts-1){
      dev_p0[iGlobal_vx_cur] += dev_rhox[iGlobal]*-dev_p1[iGlobal_vx_prev]
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxx_cur]-dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1]-dev_p1[iGlobal_sigmaxz_cur])+
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2]-dev_p1[iGlobal_sigmaxz_cur-1])+
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3]-dev_p1[iGlobal_sigmaxz_cur-2])+
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4]-dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += -dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_prev]
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmazz_cur]-dev_p1[iGlobal_sigmazz_cur-1])+
                                   dev_zCoeff[1]*(dev_p1[iGlobal_sigmazz_cur+1]-dev_p1[iGlobal_sigmazz_cur-2])+
                                   dev_zCoeff[2]*(dev_p1[iGlobal_sigmazz_cur+2]-dev_p1[iGlobal_sigmazz_cur-3])+
                                   dev_zCoeff[3]*(dev_p1[iGlobal_sigmazz_cur+3]-dev_p1[iGlobal_sigmazz_cur-4]))
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxz_cur])+
                                   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                                   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                                   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += -dev_p1[iGlobal_sigmaxx_prev]/(2*dev_dts)
            - dev_lamb2Mu[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
               - dev_lamb[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmazz_cur] += -dev_p1[iGlobal_sigmazz_prev]/(2*dev_dts)
               - dev_lamb[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
            - dev_lamb2Mu[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += -dev_p1[iGlobal_sigmaxz_prev]/(2*dev_dts)
               - dev_muxz[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
               - dev_muxz[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                                  	dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                                  	dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                                  	dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }
    else if(itGlobal<absoluteLastTimeSampleForBlock){
      dev_p0[iGlobal_vx_cur] += dev_rhox[iGlobal]*(dev_p1[iGlobal_vx_next] - dev_p1[iGlobal_vx_prev])
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxx_cur]-dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1]-dev_p1[iGlobal_sigmaxz_cur])+
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2]-dev_p1[iGlobal_sigmaxz_cur-1])+
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3]-dev_p1[iGlobal_sigmaxz_cur-2])+
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4]-dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += dev_rhoz[iGlobal]*(dev_p1[iGlobal_vz_next] - dev_p1[iGlobal_vz_prev])
                                - (dev_zCoeff[0]*(dev_p1[iGlobal_sigmazz_cur]-dev_p1[iGlobal_sigmazz_cur-1])+
                                   dev_zCoeff[1]*(dev_p1[iGlobal_sigmazz_cur+1]-dev_p1[iGlobal_sigmazz_cur-2])+
                                   dev_zCoeff[2]*(dev_p1[iGlobal_sigmazz_cur+2]-dev_p1[iGlobal_sigmazz_cur-3])+
                                   dev_zCoeff[3]*(dev_p1[iGlobal_sigmazz_cur+3]-dev_p1[iGlobal_sigmazz_cur-4]))
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_p1[iGlobal_sigmaxz_cur])+
                                   dev_xCoeff[1]*(dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                                   dev_xCoeff[2]*(dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                                   dev_xCoeff[3]*(dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += (dev_p1[iGlobal_sigmaxx_next] - dev_p1[iGlobal_sigmaxx_prev])/(2*dev_dts)
            - dev_lamb2Mu[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
               - dev_lamb[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmazz_cur] += (dev_p1[iGlobal_sigmazz_next] - dev_p1[iGlobal_sigmazz_prev])/(2*dev_dts)
               - dev_lamb[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                                    dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                                    dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                                    dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]))
            - dev_lamb2Mu[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += (dev_p1[iGlobal_sigmaxz_next] - dev_p1[iGlobal_sigmaxz_prev])/(2*dev_dts)
               - dev_muxz[iGlobal]*(dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                                    dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                                    dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                                    dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
               - dev_muxz[iGlobal]*(dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                                  	dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                                  	dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                                  	dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }

}

/* kernel to compute adjoint time step */
__global__ void ker_we_adj(float* dev_p0, float* dev_p1,
                             float* dev_rhox, float* dev_rhoz, float* dev_lamb2Mu, float* dev_lamb, float* dev_muxz,
                             int absoluteFirstTimeSampleForBlock, int absoluteLastTimeSampleForBlock){
                             //float* dev_c_all,float* dev_n_all, float* dev_elastic_param_scaled) {

    // calculate global and local x/z/t coordinates
    int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int itInTimeBlock = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int itGlobal = absoluteFirstTimeSampleForBlock + itInTimeBlock;
    int iGlobal = dev_nz*ixGlobal + izGlobal;

    int iGlobal_vx_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vx_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vx_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 0 +  dev_nz*ixGlobal + izGlobal;

    int iGlobal_vz_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vz_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_vz_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 1 +  dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmaxx_prev= dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 2 +  dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxx_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 2 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxx_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 2 + dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmazz_prev= dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmazz_cur= dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmazz_next= dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 3 + dev_nz*ixGlobal + izGlobal;

    int iGlobal_sigmaxz_prev = dev_nz*dev_nx*5*(itInTimeBlock-1) + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxz_cur = dev_nz*dev_nx*5*itInTimeBlock + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;
    int iGlobal_sigmaxz_next = dev_nz*dev_nx*5*(itInTimeBlock+1) + dev_nz*dev_nx* 4 + dev_nz*ixGlobal + izGlobal;

    //t=0 boundary condition
    if(itGlobal==0){
      dev_p0[iGlobal_vx_cur] += dev_rhox[iGlobal]*-dev_p1[iGlobal_vx_next]
                            + (dev_xCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb2Mu[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb2Mu[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_lamb2Mu[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb2Mu[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_lamb2Mu[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb2Mu[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_lamb2Mu[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                            + (dev_xCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmazz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmazz_cur+1*dev_nz]-dev_lamb[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmazz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmazz_cur+2*dev_nz]-dev_lamb[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmazz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmazz_cur+3*dev_nz]-dev_lamb[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmazz_cur-4*dev_nz]))
                            + (dev_zCoeff[0]*(dev_muxz[iGlobal+1]*dev_p1[iGlobal_sigmaxz_cur+1]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_zCoeff[1]*(dev_muxz[iGlobal+2]*dev_p1[iGlobal_sigmaxz_cur+2]-dev_muxz[iGlobal-1]*dev_p1[iGlobal_sigmaxz_cur-1])+
                               dev_zCoeff[2]*(dev_muxz[iGlobal+3]*dev_p1[iGlobal_sigmaxz_cur+3]-dev_muxz[iGlobal-2]*dev_p1[iGlobal_sigmaxz_cur-2])+
                               dev_zCoeff[3]*(dev_muxz[iGlobal+4]*dev_p1[iGlobal_sigmaxz_cur+4]-dev_muxz[iGlobal-3]*dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += -dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_next]
                            + (dev_zCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb[iGlobal-1]*dev_p1[iGlobal_sigmaxx_cur-1])+
                               dev_zCoeff[1]*(dev_lamb[iGlobal+1]*dev_p1[iGlobal_sigmaxx_cur+1]-dev_lamb[iGlobal-2]*dev_p1[iGlobal_sigmaxx_cur-2])+
                               dev_zCoeff[2]*(dev_lamb[iGlobal+2]*dev_p1[iGlobal_sigmaxx_cur+2]-dev_lamb[iGlobal-3]*dev_p1[iGlobal_sigmaxx_cur-3])+
                               dev_zCoeff[3]*(dev_lamb[iGlobal+3]*dev_p1[iGlobal_sigmaxx_cur+3]-dev_lamb[iGlobal-4]*dev_p1[iGlobal_sigmaxx_cur-4]))
                            + (dev_zCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb2Mu[iGlobal-1]*dev_p1[iGlobal_sigmazz_cur-1])+
                               dev_zCoeff[1]*(dev_lamb2Mu[iGlobal+1]*dev_p1[iGlobal_sigmazz_cur+1]-dev_lamb2Mu[iGlobal-2]*dev_p1[iGlobal_sigmazz_cur-2])+
                               dev_zCoeff[2]*(dev_lamb2Mu[iGlobal+2]*dev_p1[iGlobal_sigmazz_cur+2]-dev_lamb2Mu[iGlobal-3]*dev_p1[iGlobal_sigmazz_cur-3])+
                               dev_zCoeff[3]*(dev_lamb2Mu[iGlobal+3]*dev_p1[iGlobal_sigmazz_cur+3]-dev_lamb2Mu[iGlobal-4]*dev_p1[iGlobal_sigmazz_cur-4]))
                            + (dev_xCoeff[0]*(dev_muxz[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_xCoeff[1]*(dev_muxz[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_muxz[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_muxz[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_muxz[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_muxz[iGlobal+4*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_muxz[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += -dev_p1[iGlobal_sigmaxx_next]/(2*dev_dts)
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmazz_cur] += -dev_p1[iGlobal_sigmazz_next]/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += -dev_p1[iGlobal_sigmaxz_next]/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }
    else if(itInTimeBlock==0){
      //do nothing in this case.
    }
    //t=nt-1 boundary condition
    else if(itGlobal==dev_nts-1){
      dev_p0[iGlobal_vx_cur] += dev_rhox[iGlobal]*dev_p1[iGlobal_vx_prev]
                            + (dev_xCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb2Mu[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb2Mu[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_lamb2Mu[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb2Mu[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_lamb2Mu[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb2Mu[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_lamb2Mu[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                            + (dev_xCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmazz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmazz_cur+1*dev_nz]-dev_lamb[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmazz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmazz_cur+2*dev_nz]-dev_lamb[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmazz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmazz_cur+3*dev_nz]-dev_lamb[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmazz_cur-4*dev_nz]))
                            + (dev_zCoeff[0]*(dev_muxz[iGlobal+1]*dev_p1[iGlobal_sigmaxz_cur+1]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_zCoeff[1]*(dev_muxz[iGlobal+2]*dev_p1[iGlobal_sigmaxz_cur+2]-dev_muxz[iGlobal-1]*dev_p1[iGlobal_sigmaxz_cur-1])+
                               dev_zCoeff[2]*(dev_muxz[iGlobal+3]*dev_p1[iGlobal_sigmaxz_cur+3]-dev_muxz[iGlobal-2]*dev_p1[iGlobal_sigmaxz_cur-2])+
                               dev_zCoeff[3]*(dev_muxz[iGlobal+4]*dev_p1[iGlobal_sigmaxz_cur+4]-dev_muxz[iGlobal-3]*dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_prev]
                            + (dev_zCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb[iGlobal-1]*dev_p1[iGlobal_sigmaxx_cur-1])+
                               dev_zCoeff[1]*(dev_lamb[iGlobal+1]*dev_p1[iGlobal_sigmaxx_cur+1]-dev_lamb[iGlobal-2]*dev_p1[iGlobal_sigmaxx_cur-2])+
                               dev_zCoeff[2]*(dev_lamb[iGlobal+2]*dev_p1[iGlobal_sigmaxx_cur+2]-dev_lamb[iGlobal-3]*dev_p1[iGlobal_sigmaxx_cur-3])+
                               dev_zCoeff[3]*(dev_lamb[iGlobal+3]*dev_p1[iGlobal_sigmaxx_cur+3]-dev_lamb[iGlobal-4]*dev_p1[iGlobal_sigmaxx_cur-4]))
                            + (dev_zCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb2Mu[iGlobal-1]*dev_p1[iGlobal_sigmazz_cur-1])+
                               dev_zCoeff[1]*(dev_lamb2Mu[iGlobal+1]*dev_p1[iGlobal_sigmazz_cur+1]-dev_lamb2Mu[iGlobal-2]*dev_p1[iGlobal_sigmazz_cur-2])+
                               dev_zCoeff[2]*(dev_lamb2Mu[iGlobal+2]*dev_p1[iGlobal_sigmazz_cur+2]-dev_lamb2Mu[iGlobal-3]*dev_p1[iGlobal_sigmazz_cur-3])+
                               dev_zCoeff[3]*(dev_lamb2Mu[iGlobal+3]*dev_p1[iGlobal_sigmazz_cur+3]-dev_lamb2Mu[iGlobal-4]*dev_p1[iGlobal_sigmazz_cur-4]))
                            + (dev_xCoeff[0]*(dev_muxz[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_xCoeff[1]*(dev_muxz[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_muxz[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_muxz[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_muxz[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_muxz[iGlobal+4*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_muxz[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += dev_p1[iGlobal_sigmaxx_prev]/(2*dev_dts)
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmazz_cur] += dev_p1[iGlobal_sigmazz_prev]/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += dev_p1[iGlobal_sigmaxz_prev]/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }
    else if(itGlobal<absoluteLastTimeSampleForBlock){
      dev_p0[iGlobal_vx_cur] += (dev_rhox[iGlobal]*dev_p1[iGlobal_vx_prev] - dev_rhox[iGlobal]*dev_p1[iGlobal_vx_next])
                            + (dev_xCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb2Mu[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb2Mu[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+1*dev_nz]-dev_lamb2Mu[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb2Mu[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+2*dev_nz]-dev_lamb2Mu[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb2Mu[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxx_cur+3*dev_nz]-dev_lamb2Mu[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmaxx_cur-4*dev_nz]))
                            + (dev_xCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmazz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_lamb[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmazz_cur+1*dev_nz]-dev_lamb[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmazz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_lamb[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmazz_cur+2*dev_nz]-dev_lamb[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmazz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_lamb[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmazz_cur+3*dev_nz]-dev_lamb[iGlobal-4*dev_nz]*dev_p1[iGlobal_sigmazz_cur-4*dev_nz]))
                            + (dev_zCoeff[0]*(dev_muxz[iGlobal+1]*dev_p1[iGlobal_sigmaxz_cur+1]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_zCoeff[1]*(dev_muxz[iGlobal+2]*dev_p1[iGlobal_sigmaxz_cur+2]-dev_muxz[iGlobal-1]*dev_p1[iGlobal_sigmaxz_cur-1])+
                               dev_zCoeff[2]*(dev_muxz[iGlobal+3]*dev_p1[iGlobal_sigmaxz_cur+3]-dev_muxz[iGlobal-2]*dev_p1[iGlobal_sigmaxz_cur-2])+
                               dev_zCoeff[3]*(dev_muxz[iGlobal+4]*dev_p1[iGlobal_sigmaxz_cur+4]-dev_muxz[iGlobal-3]*dev_p1[iGlobal_sigmaxz_cur-3]));
      dev_p0[iGlobal_vz_cur] += (dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_prev] - dev_rhoz[iGlobal]*dev_p1[iGlobal_vz_next])
                            + (dev_zCoeff[0]*(dev_lamb[iGlobal]*dev_p1[iGlobal_sigmaxx_cur]-dev_lamb[iGlobal-1]*dev_p1[iGlobal_sigmaxx_cur-1])+
                               dev_zCoeff[1]*(dev_lamb[iGlobal+1]*dev_p1[iGlobal_sigmaxx_cur+1]-dev_lamb[iGlobal-2]*dev_p1[iGlobal_sigmaxx_cur-2])+
                               dev_zCoeff[2]*(dev_lamb[iGlobal+2]*dev_p1[iGlobal_sigmaxx_cur+2]-dev_lamb[iGlobal-3]*dev_p1[iGlobal_sigmaxx_cur-3])+
                               dev_zCoeff[3]*(dev_lamb[iGlobal+3]*dev_p1[iGlobal_sigmaxx_cur+3]-dev_lamb[iGlobal-4]*dev_p1[iGlobal_sigmaxx_cur-4]))
                            + (dev_zCoeff[0]*(dev_lamb2Mu[iGlobal]*dev_p1[iGlobal_sigmazz_cur]-dev_lamb2Mu[iGlobal-1]*dev_p1[iGlobal_sigmazz_cur-1])+
                               dev_zCoeff[1]*(dev_lamb2Mu[iGlobal+1]*dev_p1[iGlobal_sigmazz_cur+1]-dev_lamb2Mu[iGlobal-2]*dev_p1[iGlobal_sigmazz_cur-2])+
                               dev_zCoeff[2]*(dev_lamb2Mu[iGlobal+2]*dev_p1[iGlobal_sigmazz_cur+2]-dev_lamb2Mu[iGlobal-3]*dev_p1[iGlobal_sigmazz_cur-3])+
                               dev_zCoeff[3]*(dev_lamb2Mu[iGlobal+3]*dev_p1[iGlobal_sigmazz_cur+3]-dev_lamb2Mu[iGlobal-4]*dev_p1[iGlobal_sigmazz_cur-4]))
                            + (dev_xCoeff[0]*(dev_muxz[iGlobal+1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+1*dev_nz]-dev_muxz[iGlobal]*dev_p1[iGlobal_sigmaxz_cur])+
                               dev_xCoeff[1]*(dev_muxz[iGlobal+2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+2*dev_nz]-dev_muxz[iGlobal-1*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_muxz[iGlobal+3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+3*dev_nz]-dev_muxz[iGlobal-2*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_muxz[iGlobal+4*dev_nz]*dev_p1[iGlobal_sigmaxz_cur+4*dev_nz]-dev_muxz[iGlobal-3*dev_nz]*dev_p1[iGlobal_sigmaxz_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmaxx_cur] += (dev_p1[iGlobal_sigmaxx_prev] - dev_p1[iGlobal_sigmaxx_next])/(2*dev_dts)
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vx_cur+1*dev_nz]-dev_p1[iGlobal_vx_cur])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vx_cur+2*dev_nz]-dev_p1[iGlobal_vx_cur-1*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vx_cur+3*dev_nz]-dev_p1[iGlobal_vx_cur-2*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vx_cur+4*dev_nz]-dev_p1[iGlobal_vx_cur-3*dev_nz]));
      dev_p0[iGlobal_sigmazz_cur] += (dev_p1[iGlobal_sigmazz_prev] - dev_p1[iGlobal_sigmazz_next])/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vz_cur+1]-dev_p1[iGlobal_vz_cur])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vz_cur+2]-dev_p1[iGlobal_vz_cur-1])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vz_cur+3]-dev_p1[iGlobal_vz_cur-2])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vz_cur+4]-dev_p1[iGlobal_vz_cur-3]));
      dev_p0[iGlobal_sigmaxz_cur] += (dev_p1[iGlobal_sigmaxz_prev] - dev_p1[iGlobal_sigmaxz_next])/(2*dev_dts)
                            + (dev_zCoeff[0]*(dev_p1[iGlobal_vx_cur]-dev_p1[iGlobal_vx_cur-1])+
                               dev_zCoeff[1]*(dev_p1[iGlobal_vx_cur+1]-dev_p1[iGlobal_vx_cur-2])+
                               dev_zCoeff[2]*(dev_p1[iGlobal_vx_cur+2]-dev_p1[iGlobal_vx_cur-3])+
                               dev_zCoeff[3]*(dev_p1[iGlobal_vx_cur+3]-dev_p1[iGlobal_vx_cur-4]))
                            + (dev_xCoeff[0]*(dev_p1[iGlobal_vz_cur]-dev_p1[iGlobal_vz_cur-1*dev_nz])+
                               dev_xCoeff[1]*(dev_p1[iGlobal_vz_cur+1*dev_nz]-dev_p1[iGlobal_vz_cur-2*dev_nz])+
                               dev_xCoeff[2]*(dev_p1[iGlobal_vz_cur+2*dev_nz]-dev_p1[iGlobal_vz_cur-3*dev_nz])+
                               dev_xCoeff[3]*(dev_p1[iGlobal_vz_cur+3*dev_nz]-dev_p1[iGlobal_vz_cur-4*dev_nz]));
    }

}
