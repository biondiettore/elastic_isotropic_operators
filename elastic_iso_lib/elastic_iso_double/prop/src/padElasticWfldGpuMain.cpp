#include <iostream>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"

using namespace SEP;

int main(int argc, char **argv) {

	// IO bullshit
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	// Model
	std::shared_ptr <genericRegFile> modelFile = io->getRegFile("model",usageIn);
 	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	if (modelHyper->getNdim() == 2){
		axis extAxis(1, 0.0, 1.0);
		modelHyper->addAxis(extAxis);
	}
 	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

	// Model parameters
	long long nz = model->getHyper()->getAxis(1).n;
	long long nx = model->getHyper()->getAxis(2).n;
	long long nPar = model->getHyper()->getAxis(3).n;
	float oz = model->getHyper()->getAxis(1).o;
	float ox = model->getHyper()->getAxis(2).o;
	float dz = model->getHyper()->getAxis(1).d;
	float dx = model->getHyper()->getAxis(2).d;

	// Parfile
	int zPad = par->getInt("zPad",0);
	int xPad = par->getInt("xPad",0);
	int fat = par->getInt("fat", 4);
	int nts = par->getInt("nts",-1);

	//data nx and nz
	long long new_nx = nx + 2*fat + 2*xPad;
	long long new_nz = nx + 2*fat + 2*zPad;
	float new_ox = ox - (fat + xPad) * dx;
	float new_oz = oz - (fat + zPad) * dz;

	// Data
	axis new_zAxis = axis(new_nz, new_oz, dz);
	axis new_xAxis = axis(new_nx, new_ox, dx);
	axis extAxis = 	axis(model->getHyper()->getAxis(3));
	axis tAxis = axis(model->getHyper()->getAxis(4));
 	std::shared_ptr<SEP::hypercube> dataHyper(new hypercube(new_zAxis, new_xAxis, extAxis,tAxis));
 	std::shared_ptr<SEP::float4DReg> data(new SEP::float4DReg(dataHyper));
	std::shared_ptr <genericRegFile> dataFile = io->getRegFile("data",usageOut);
	dataFile->setHyper(dataHyper);
	dataFile->writeDescription();
	data->scale(0.0);

	/****************************************************************************/
	//First model parameter is considered to be density and will be padded with values within the fat layer
	// Copy central part
	for (long long ix=0; ix<nx; ix++){
		for (long long iz=0; iz<nz; iz++){
			if(surfaceCondition==0){
				(*data->_mat)[0][ix+fat+xPad][iz+fat+zPad] = (*model->_mat)[0][ix][iz];
			}
			else if(surfaceCondition==1){
				(*data->_mat)[0][ix+fat+xPad][iz+zPad] = (*model->_mat)[0][ix][iz];
			}
		}
	}

	for (long long ix=0; ix<nx; ix++){
		// Top central part
		if(surfaceCondition==0){
			for (long long iz=0; iz<zPad+fat; iz++){
				(*data->_mat)[0][ix+fat+xPad][iz] = (*model->_mat)[0][ix][0];
			}
		}
		else if(surfaceCondition==1){
			for (long long iz=0; iz<zPad; iz++){
				(*data->_mat)[0][ix+fat+xPad][iz] = (*model->_mat)[0][ix][0];
			}
		}

		// Bottom central part
		if(surfaceCondition==0){
			for (long long iz=0; iz<zPadPlus+fat; iz++){
				(*data->_mat)[0][ix+fat+xPad][iz+fat+zPad+nz] = (*model->_mat)[0][ix][nz-1];
			}
		}
		else if(surfaceCondition==1){
			for (long long iz=0; iz<zPadPlus+fat; iz++){
				(*data->_mat)[0][ix+fat+xPad][iz+zPad+nz] = (*model->_mat)[0][ix][nz-1];
			}
		}

	}

	// Left part
	for (long long ix=0; ix<xPad+fat; ix++){
		for (long long iz=0; iz<nzNewTotal; iz++) {
			(*data->_mat)[0][ix][iz] = (*data->_mat)[0][xPad+fat][iz];
		}
	}

	// Right part
	for (long long ix=0; ix<xPadPlus+fat; ix++){
		for (long long iz=0; iz<nzNewTotal; iz++){
			(*data->_mat)[0][ix+fat+nx+xPad][iz] = (*data->_mat)[0][fat+xPad+nx-1][iz];
		}
	}

	/****************************************************************************/
	//Lame and mu padding
	for (int iPar=1; iPar<nPar; iPar++) {

		// Copy central part
		for (long long ix=0; ix<nx; ix++){
			for (long long iz=0; iz<nz; iz++){
				if(surfaceCondition==0){
					(*data->_mat)[iPar][ix+fat+xPad][iz+fat+zPad] = (*model->_mat)[iPar][ix][iz];
				}
				else if(surfaceCondition==1){
					(*data->_mat)[iPar][ix+fat+xPad][iz+zPad] = (*model->_mat)[iPar][ix][iz];
				}
			}
		}

		for (long long ix=0; ix<nx; ix++){
			// Top central part
			for (long long iz=0; iz<zPad; iz++){
				if(surfaceCondition==0){
					(*data->_mat)[iPar][ix+fat+xPad][iz+fat] = (*model->_mat)[iPar][ix][0];
				}
				else if(surfaceCondition==1){
					(*data->_mat)[iPar][ix+fat+xPad][iz] = (*model->_mat)[iPar][ix][0];
				}
			}
			// Bottom central part
			for (long long iz=0; iz<zPadPlus; iz++){
				if(surfaceCondition==0){
					(*data->_mat)[iPar][ix+fat+xPad][iz+fat+zPad+nz] = (*model->_mat)[iPar][ix][nz-1];
				}
				else if(surfaceCondition==1){
					(*data->_mat)[iPar][ix+fat+xPad][iz+zPad+nz] = (*model->_mat)[iPar][ix][nz-1];
				}

			}
		}

		// Left part
		for (long long ix=0; ix<xPad; ix++){
			if(surfaceCondition==0){
				for (long long iz=0; iz<nzNew; iz++) {
					(*data->_mat)[iPar][ix+fat][iz+fat] = (*data->_mat)[iPar][xPad+fat][iz+fat];
				}
			}
			else if(surfaceCondition==1){
				for (long long iz=0; iz<nzNewTotal; iz++) {
					(*data->_mat)[iPar][ix+fat][iz] = (*data->_mat)[iPar][xPad+fat][iz];
				}
			}

		}

		// Right part
		for (long long ix=0; ix<xPadPlus; ix++){
			if(surfaceCondition==0){
				for (long long iz=0; iz<nzNew; iz++){
					(*data->_mat)[iPar][ix+fat+nx+xPad][iz+fat] = (*data->_mat)[iPar][fat+xPad+nx-1][iz+fat];
				}
			}
			else if(surfaceCondition==1){
				for (long long iz=0; iz<nzNewTotal; iz++){
					(*data->_mat)[iPar][ix+fat+nx+xPad][iz] = (*data->_mat)[iPar][fat+xPad+nx-1][iz];
				}
			}

		}
	}

	/****************************************************************************/
	// Write model
	dataFile->writeFloatStream(data);

	// Display info
	std::cout << " " << std::endl;
	std::cout << "------------------------ Model padding program --------------------" << std::endl;
	std::cout << "Chosen surface condition parameter: ";
	if(surfaceCondition==0) std::cout << "(0) no free surface condition" << '\n';
	else if(surfaceCondition==1 ) std::cout << "(1) free surface condition from Robertsson (1998) chosen." << '\n';

	std::cout << "Original nz = " << nz << " [samples]" << std::endl;
	std::cout << "Original nx = " << nx << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "zPadMinus = " << zPad << " [samples]" << std::endl;
	std::cout << "zPadPlus = " << zPadPlus << " [samples]" << std::endl;
	std::cout << "xPadMinus = " << xPad << " [samples]" << std::endl;
	std::cout << "xPadPlus = " << xPadPlus << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "blockSize = " << blockSize << " [samples]" << std::endl;
	std::cout << "FAT = " << fat << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "New nz = " << nzNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "New nx = " << nxNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "-------------------------------------------------------------------" << std::endl;
	std::cout << " " << std::endl;
	return 0;

}
