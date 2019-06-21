/*
Original Author: Guillaume Barnier
Additional Authors (adding elastic prop): Stuart Farris and Ettore Biondi
*/

#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu.h"
#include "fdParamElastic.h"
//#include "nonlinearPropElasticShotsGpu.h"
#include <vector>

using namespace SEP;

int main(int argc, char **argv) {

	/************************************** Main IO *************************************/
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	/* General parameters */
	int adj = par->getInt("adj");
	int saveWavefield = par->getInt("saveWavefield");
	int dotProd = par->getInt("dotProd", 0);
	int nShot = par->getInt("nShot");
	axis shotAxis = axis(nShot, 1.0, 1.0);

	if (adj == 0 && dotProd == 0 ){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "----------------------- Running nonlinear forward -----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	if (adj == 1 && dotProd == 0 ){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "----------------------- Running nonlinear adjoint -----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}
	if (dotProd == 1){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------------ Running dot product test -----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	/* Model and data declaration */
	std::shared_ptr<float3DReg> model1float, data1float;
	std::shared_ptr<float3DReg> model1Float, data1Float;
	std::shared_ptr<float3DReg> wavefield1Float;
	std::shared_ptr<float3DReg> wavefield1float;
	std::shared_ptr <genericRegFile> model1File, data1File, wavefield1File, dampFile;

	/* Read time parameters */
	int nts = par->getInt("nts");
	float dts = par->getFloat("dts", 0.0);
	int sub = par->getInt("sub");
	axis timeAxisCoarse = axis(nts, 0.0, dts);
	int ntw = (nts - 1) * sub + 1;
	float dtw = dts / float(sub);
	axis timeAxisFine = axis(ntw, 0.0, dtw);

	/* Read padding parameters */
	int zPadMinus = par->getInt("zPadMinus");
	int zPadPlus = par->getInt("zPadPlus");
	int xPadMinus = par->getInt("xPadMinus");
	int xPadPlus = par->getInt("xPadPlus");
	int fat = par->getInt("fat");

	/************************************** Velocity model ******************************/
	/* Read velocity (includes the padding + FAT) */
	std::shared_ptr<SEP::genericRegFile> velFile = io->getRegFile("vel",usageIn);
	std::shared_ptr<SEP::hypercube> velHyper = velFile->getHyper();
	std::shared_ptr<SEP::float2DReg> velFloat(new SEP::float2DReg(velHyper));
	std::shared_ptr<SEP::float2DReg> velfloat(new SEP::float2DReg(velHyper));
	velFile->readFloatStream(velFloat);
	int nz = velFloat->getHyper()->getAxis(1).n;
	int nx = velFloat->getHyper()->getAxis(2).n;
	for (int ix = 0; ix < nx; ix++) {
		for (int iz = 0; iz < nz; iz++) {
			(*velfloat->_mat)[ix][iz] = (*velFloat->_mat)[ix][iz];
		}
	}

// 	/********************************* Create sources vector ****************************/
// 	// Create source device vector
// 	int nzSource = 1;
// 	int ozSource = par->getInt("zSource") - 1 + zPadMinus + fat;
// 	int dzSource = 1;
// 	int nxSource = 1;
// 	int oxSource = par->getInt("xSource") - 1 + xPadMinus + fat;
// 	int dxSource = 1;
// 	int spacingShots = par->getInt("spacingShots", spacingShots);
// 	axis sourceAxis(nxSource, oxSource, dxSource);
// 	std::vector<std::shared_ptr<deviceGpu>> sourcesVector;
// 	for (int iShot; iShot<nShot; iShot++){
// 		std::shared_ptr<deviceGpu> sourceDevice(new deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velfloat, nts));
// 		sourcesVector.push_back(sourceDevice);
// 		oxSource = oxSource + spacingShots;
// 	}

// 	/********************************* Create receivers vector **************************/
// 	int nzReceiver = 1;
// 	int ozReceiver = par->getInt("depthReceiver") - 1 + zPadMinus + fat;
// 	int dzReceiver = 1;
// 	int nxReceiver = par->getInt("nReceiver");
// 	int oxReceiver = par->getInt("oReceiver") - 1 + xPadMinus + fat;
// 	int dxReceiver = par->getInt("dReceiver");
// 	axis receiverAxis(nxReceiver, oxReceiver, dxReceiver);
// 	std::vector<std::shared_ptr<deviceGpu>> receiversVector;
// 	int nRecGeom = 1; // Constant receivers' geometry
// 	for (int iRec; iRec<nRecGeom; iRec++){
// 		std::shared_ptr<deviceGpu> recDevice(new deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velfloat, nts));
// 		receiversVector.push_back(recDevice);
// 	}

// 	/*********************************** Allocation *************************************/
// 	/* Forward propagation */
// 	if (adj == 0) {

// 		/* Allocate and read model */
// 		// Provide one wavelet for all shots
// 		model1File = io->getRegFile(std::string("model1"),usageIn);
// 		std::shared_ptr <hypercube> model1Hyper = model1File->getHyper();
// 		if (model1Hyper->getNdim() == 1){
// 			axis a(1);
// 			model1Hyper->addAxis(a);
// 			model1Hyper->addAxis(a);
// 		}
// 		model1Float = std::make_shared<float3DReg>(model1Hyper);
// 		model1float = std::make_shared<float3DReg>(model1Hyper);
// 		model1File->readFloatStream(model1Float);

// 		for (int i = 0; i < model1Hyper->getAxis(2).n; i++) {
// 			#pragma omp parallel for num_threads(24)
// 			for (int it = 0; it < model1Hyper->getAxis(1).n; it++) {
// 				(*model1float->_mat)[0][i][it] = (*model1Float->_mat)[0][i][it];
// 			}
// 		}

// 		/* Data float allocation */
// 		std::shared_ptr<hypercube> data1Hyper(new hypercube(model1Hyper->getAxis(1), receiverAxis, shotAxis));
// 		data1float = std::make_shared<float3DReg>(data1Hyper);
// 		data1Float = std::make_shared<float3DReg>(data1Hyper);

// 		/* Files shits */
// 		data1File = io->getRegFile(std::string("data1"), usageOut);
// 		data1File->setHyper(data1Hyper);
// 		data1File->writeDescription();
// 	}

// 	if (adj == 1) {

// 		/* Allocate and read data */
// 		data1File = io->getRegFile(std::string("data1"),usageIn);

// 		std::shared_ptr <hypercube> data1Hyper = data1File->getHyper();
// 		if (data1Hyper->getNdim() == 2) {
// 			axis a(1);
// 			data1Hyper->addAxis(a);
// 		}

// 		data1float = std::make_shared<float3DReg>(data1Hyper);
// 		data1Float = std::make_shared<float3DReg>(data1Hyper);
// 		data1File->readFloatStream(data1Float);

// 		for (int iShot = 0; iShot < data1Hyper->getAxis(3).n; iShot++) {
// 			for (int iReceiver = 0; iReceiver < data1Hyper->getAxis(2).n; iReceiver++) {
// 				for (int its = 0; its < data1Hyper->getAxis(1).n; its++) {
// 					(*data1float->_mat)[iShot][iReceiver][its] = (*data1Float->_mat)[iShot][iReceiver][its];
// 				}
// 			}
// 		}

// 		/* Allocate model */
// 		axis a(1);
// 		std::shared_ptr <hypercube> model1Hyper(new hypercube(timeAxisCoarse, a, a));
// 		model1Float = std::make_shared<float3DReg>(model1Hyper);
// 		model1float = std::make_shared<float3DReg>(model1Hyper);

// 		for (int i = 0; i < model1Hyper->getAxis(2).n; i++) {
// 			for (int it = 0; it < model1Hyper->getAxis(1).n; it++) {
// 				(*model1float->_mat)[0][i][it] = (*model1Float->_mat)[0][i][it];
// 			}
// 		}

// 		/* Files shits */
// 		model1File = io->getRegFile(std::string("model1"),usageOut);
// 		model1File->setHyper(model1Hyper);
// 		model1File->writeDescription();
// 	}

// 	if (saveWavefield == 1){
// 		// The wavefield(s) allocation is done inside the nonlinearPropShotsGpu object -> no need to allocate outside
// 		std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));
// 		wavefield1File = io->getRegFile(std::string("wavefield1"), usageOut);
// 		wavefield1File->setHyper(wavefield1Hyper);
// 		wavefield1File->writeDescription();
// 	}

// 	/************************************************************************************/
// 	/******************************** SIMULATIONS ***************************************/
// 	/************************************************************************************/

// 	/* Create nonlinear propagation object */
// 	std::shared_ptr<nonlinearPropShotsGpu> object1(new nonlinearPropShotsGpu(velfloat, par, sourcesVector, receiversVector));

// 	/********************************** FORWARD *****************************************/
// 	if (adj == 0 && dotProd == 0) {

// 		/* Apply forward */
// 		if (saveWavefield == 1){
// 			object1->forwardWavefield(false, model1float, data1float);
// 		} else {
// 			object1->forward(false, model1float, data1float);
// 		}

// 		/* Copy data */
// 		#pragma omp parallel for
// 		for (int iShot=0; iShot<nShot; iShot++){
// 			for (int iReceiver = 0; iReceiver < data1float->getHyper()->getAxis(2).n; iReceiver++) {
// 				for (int it = 0; it < data1float->getHyper()->getAxis(1).n; it++) {
// 					(*data1Float->_mat)[iShot][iReceiver][it] = (*data1float->_mat)[iShot][iReceiver][it];
// 				}
// 			}
// 		}

// 		/* Output data */
// 		data1File->writeFloatStream(data1Float);

// 		/* Wavefield */
// 		if (saveWavefield == 1){
// 			std::cout << "Writing wavefield..." << std::endl;
// 			wavefield1float = object1->getWavefield();
// 			wavefield1Float = std::make_shared<float3DReg>(wavefield1float->getHyper());

// 			#pragma omp parallel for
// 			for (int its = 0; its < nts; its++){
// 				for (int ix = 0; ix < nx; ix++){
// 					for (int iz = 0; iz < nz; iz++){
// 						(*wavefield1Float->_mat)[its][ix][iz] = (*wavefield1float->_mat)[its][ix][iz];
// 					}
// 				}
// 			}
// 			wavefield1File->writeFloatStream(wavefield1Float);
// 			std::cout << "Done!" << std::endl;
// 		}
// 	}

// 	/********************************** ADJOINT *****************************************/
// 	if (adj == 1 && dotProd == 0){

// 		/* Apply adjoint */
// 		if (saveWavefield == 1){
// 			object1->adjointWavefield(false, model1float, data1float);
// 		} else {
// 			object1->adjoint(false, model1float, data1float);
// 		}

// 		/* Copy model */
// 		for (int iShot=0; iShot<model1float->getHyper()->getAxis(3).n; iShot++){
// 			for (int iSource=0; iSource<model1float->getHyper()->getAxis(2).n; iSource++){
// 				for (int its=0; its<model1float->getHyper()->getAxis(1).n; its++){
// 					(*model1Float->_mat)[iShot][iSource][its] = (*model1float->_mat)[iShot][iSource][its];
// 				}
// 			}
// 		}
// 		model1File->writeFloatStream(model1Float);

// 		/* Wavefield */
// 		if (saveWavefield == 1){
// 			std::cout << "Writing wavefield..." << std::endl;
// 			wavefield1float = object1->getWavefield();
// 			wavefield1Float = std::make_shared<float3DReg>(wavefield1float->getHyper());
// 			#pragma omp parallel for
// 			for (int its = 0; its < nts; its++){
// 				for (int ix = 0; ix < nx; ix++){
// 					for (int iz = 0; iz < nz; iz++){
// 						(*wavefield1Float->_mat)[its][ix][iz] = (*wavefield1float->_mat)[its][ix][iz];
// 					}
// 				}
// 			}
// 			wavefield1File->writeFloatStream(wavefield1Float);
// 			std::cout << "Done!" << std::endl;
// 		}
// 	}

// 	/****************************** DOT PRODUCT TEST ************************************/
// 	if (dotProd == 1){
// 		object1->setDomainRange(model1float, data1float);
// 		bool dotprod;
// 		dotprod = object1->dotTest(true);
// 	}

// 	std::cout << " " << std::endl;
// 	std::cout << "-------------------------------------------------------------------" << std::endl;
// 	std::cout << "------------------------------ ALL DONE ---------------------------" << std::endl;
// 	std::cout << "-------------------------------------------------------------------" << std::endl;
// 	std::cout << " " << std::endl;

	return 0;

}
