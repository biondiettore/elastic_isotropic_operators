#!/usr/bin/env python3
"""
GPU-based elastic isotropic velocity-stress wave-equation non-linear modeling operator (single-precision version)

USAGE EXAMPLE:
	nonlinearPythonElasticFloatMain.py elasticParam=elasticModel.H model=elastic_wavelet.H par=parNonlinear.p data=elastic_data.H

INPUT PARAMETERS:
	elasticParam = [no default] - string; Header file defining elastic subsurface parameters (z [m], x [m], component [see mod_par]).
	                                      These parameters must be correctly padded using the padElasticFileGpuMain program.

	model = [no default] - string; Header file defining elastic source term (t,component=[fx -> volumetric force along x axis [N/m^3], fz -> volumetric force along z axis [N/m^3],
	                               Sxx -> injection rate within normal xx stress [Pa/s], Szz -> injection rate within normal zz stress [Pa/s],
	                               Sxz -> injection rate within shear xz stress [Pa/s]]).

	data [no default] - string; Header file in which recorded elastic data will be written (t [s],receiver,
	                            component=[vx [m/s], vz [m/s], sigmaxx [Pa], sigmazz [Pa], sigmaxz [Pa]])

	info [0] - int; boolean; Verbosity of the program. If true, certain program information will be displayed

	mod_par [0] - int; Choice of parameterization for the elasticParam header file. [0 = Density [Kg/m^3],Lame [Pa],Shear modulus [Pa];
	                   1 = Vp [m/s],Vs [m/s], Density [Kg/m^3]]
					   2 = Vp [km/s],Vs [km/s], Density [g/cm^3]]

	nts [no default] - int; Number of time samples within the elastic source/data

	dts [no default] - float; Coarse sampling of the elastic source/data

	sub [no default] - int; Ratio between coarse and propagation samplings to meet stability conditions for stable modeling (should be greater than 1)

	nz [no default] - int; Number of samples in the z direction with padding

	dz [no default] - float; Sampling in the z direction

	nx [no default] - int; Number of samples in the x direction with padding

	dx [no default] - float; Sampling in the x direction

	zPadMinus,zPadPlus,xPadMinus,xPadPlus [no default] - int; Number of padding samples on the top, bottom, left, and right portions of the model.
	                                                          These numbers are printed by the padElasticFileGpuMain program in which only
	                                                          zPad and xPad parameters need to be provided

	fMax [no default] - float; Maximum frequency content within the provided elastic source term. Necessary to check stability and dispersion conditions

	nExp [no default] - int; Number of shots to be modeled

	zSource [no default] - int; Depth of the source term in sample value without padding (note: first sample is identified by 1)

	xSource [no default] - int; X position of the first shot in sample value without padding (note: first sample is identified by 1)

	spacingShots [no default] - int; Shot sampling in samples

	nReceiver [no default] - int; Number of receivers to be used for recording

	depthReceiver [no default] - int; Depth of the receivers in sample value without padding (note: first sample is identified by 1)

	oReceiver [no default] - int; X position of the first receiver in sample value without padding (note: first sample is identified by 1)

	dReceiver [no default] - int; Receiver sampling in samples

	nGpu [no default] - int; Number of GPU cards to be used during the modeling. Note the program only parallelizes each shot and does not perform domain decomposition. Hence, it is useless to use 2 GPU cards if only one shot needs to be modeled

	iGpu [no default] - int; List of GPU cards to be used during the modeling. Same comments as in the nGpu parameter. Note this argument overwrites nGpu

	saveWavefield [0] - boolean; Flag to save the wavefield for a given shot defined by the flag wavefieldShotNumber

	wfldFile [no default] - string; Header-file name where the wavefield is written if saveWavefield is 1

	wavefieldShotNumber [0] - int; Shot index of the wavefield to be saved

	useStreams [0] - boolean; Flag to use CUDA streams when the wavefield needs to be saved (useful for large wavefield and does not fit in GPU memory)

	blockSize [16] - int; GPU-grid-related block size (i.e., number of threads per block in a GPU grid)

"""


import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_float_prop
import numpy as np
import time

if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	# Initialize operator
	modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid = Elastic_iso_float_prop.nonlinearOpInitFloat(sys.argv)

	# Construct nonlinear operator object
	nonlinearElasticOp=Elastic_iso_float_prop.nonlinearPropElasticShotsGpu(modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)


	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		nonlinearElasticOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------------ Running Python nonlinear forward ---------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")
		#modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
		modelTemp=genericIO.defaultIO.getVector(modelFile)
		modelFMat=modelFloat.getNdArray()
		modelTMat=modelTemp.getNdArray()
		modelFMat[0,:,0,:]=modelTMat

		#check if we want to save wavefield
		if (parObject.getInt("saveWavefield",0) == 1):
			wfldFile=parObject.getString("wfldFile","noWfldFile")
			if (wfldFile == "noWfldFile"):
				raise IOError("**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****")
			#run Nonlinear forward with wavefield saving
			nonlinearElasticOp.forwardWavefield(False,modelFloat,dataFloat)
			#save wavefield to disk
			wavefieldFloat = nonlinearElasticOp.getWavefield()
			# genericIO.defaultIO.writeVector(wfldFile,wavefieldFloat)
			wavefieldFloat.writeVec(wfldFile)
		else:
			#run Nonlinear forward without wavefield saving
			nonlinearElasticOp.forward(False,modelFloat,dataFloat)
		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:
		raise NotImplementedError("ERROR! Adjoint operator not implemented yet!")
