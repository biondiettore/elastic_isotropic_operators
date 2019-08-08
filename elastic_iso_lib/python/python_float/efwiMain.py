#!/usr/bin/env python3.5
"""
GPU-based elastic isotropic velocity-stress wave-equation full-waveform inversion script

USAGE EXAMPLE:


INPUT PARAMETERS:
"""
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Elastic_iso_float_prop
import elasticParamConvertModule as ElaConv
import interpBSplineModule
import dataTaperModule
import spatialDerivModule
import maskGradientModule

# Solver library
import pyOperator as pyOp
import pyNLCGsolver as NLCG
import pyLBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStopperBase as Stopper
import pyStepperParabolic as Stepper
import inversionUtils
from sys_util import logger

############################ Bounds vectors ####################################
# Create bound vectors for FWI
def createBoundVectors(parObject,model):

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	fat=parObject.getInt("fat")
	spline=parObject.getInt("spline",0)
	if (spline==1): fat=0

	# Min bound
	minBoundVectorFile=parObject.getString("minBoundVector","noMinBoundVectorFile")
	if (minBoundVectorFile=="noMinBoundVectorFile"):
		minBound1=parObject.getFloat("minBound_par1",-np.inf)
		minBound2=parObject.getFloat("minBound_par2",-np.inf)
		minBound3=parObject.getFloat("minBound_par3",-np.inf)
		minBoundVector=model.clone()
		minBoundVector.set(0.0)
		minBoundVectorNd=minBoundVector.getNdArray()
		minBoundVectorNd[0,fat:nx-fat,fat:nz-fat]=minBound1
		minBoundVectorNd[1,fat:nx-fat,fat:nz-fat]=minBound2
		minBoundVectorNd[2,fat:nx-fat,fat:nz-fat]=minBound3
	else:
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound1=parObject.getFloat("maxBound_par1",np.inf)
		maxBound2=parObject.getFloat("maxBound_par2",np.inf)
		maxBound3=parObject.getFloat("maxBound_par3",np.inf)
		maxBoundVector=model.clone()
		maxBoundVector.set(0.0)
		maxBoundVectorNd=maxBoundVector.getNdArray()
		maxBoundVectorNd[0,fat:nx-fat,fat:nz-fat]=maxBound1
		maxBoundVectorNd[1,fat:nx-fat,fat:nz-fat]=maxBound2
		maxBoundVectorNd[2,fat:nx-fat,fat:nz-fat]=maxBound3

	else:
		maxBoundVector=genericIO.defaultIO.getVector(maxBoundVectorFile)


	return minBoundVector,maxBoundVector


# Elastic FWI workflow script
if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

    # IO object
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	# regType=parObject.getString("reg","None")
	# reg=0
	# if (regType != "None"): reg=1
	# epsilonEval=parObject.getInt("epsilonEval",0)

	# Nonlinear solver
	solverType=parObject.getString("solver")
	stepper=parObject.getString("stepper","default")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("----------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Elastic FWI logfile -------------------------")
	if(pyinfo): print("----------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Elastic FWI logfile -------------------------")


	############################# Initialization ###############################

	# FWI nonlinear operator
	elasticParamFloat,dataFloat,sourcesFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid = Elastic_iso_float_prop.nonlinearFwiOpInitFloat(sys.argv)

	# Born
	# Initialize operator
	_,_,_,_,sourcesSignalsVector,_,_,_,_,_,_,_,_ = Elastic_iso_float_prop.BornOpInitFloat(sys.argv)

	############################# Read files ###################################
	# Seismic source
	sourceFile=parObject.getString("sources","noSourcesFile")
	if (modelFile == "noSourcesFile"):
		raise IOError("**** ERROR: User did not provide sources file ****\n")
	sourcesTemp=genericIO.defaultIO.getVector(sourceFile)
	sourcesFMat=sourcesFloat.getNdArray()
	sourcesTMat=sourcesTemp.getNdArray()
	sourcesFMat[0,:,0,:]=sourcesTMat
	del sourcesTemp

	# Data
	dataFile=parObject.getString("data","noDataFile")
	if (dataFile == "noDataFile"):
		raise IOError("**** ERROR: User did not provide data file ****\n")
	data=genericIO.defaultIO.getVector(dataFile)

	############################# Instanciation ################################
	# Nonlinear
	nonlinearElasticOp=Elastic_iso_float_prop.nonlinearFwiPropElasticShotsGpu(elasticParamFloat,dataFloat,sourcesFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

	# Construct nonlinear operator object
	BornElasticOp=Elastic_iso_float_prop.BornElasticShotsGpu(elasticParamFloat,dataFloat,elasticParamFloat,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

	# Conventional FWI non-linear operator
	fwiOp=pyOp.NonLinearOperator(nonlinearElasticOp,BornElasticOp,BornElasticOp.setBackground)

	#Elastic parameter conversion if any
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv.ElasticConv(elasticParamFloat,mod_par)
		#Jacobian
		convOpJac = ElaConv.ElasticConvJab(elasticParamFloat,elasticParamFloat,mod_par)
		#Creating non-linear operator
		convOpNl=pyOp.NonLinearOperator(convOp,convOpJac,convOpJac.setBackground)
		#Chanining non-linear operators if not using Lame,Mu,Density parameterization
		#f(g(m)) where f is the non-linear modeling operator and g is the non-linear change of variables
		fwiOp=pyOp.CombNonlinearOp(convOpNl,fwiOp)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=createBoundVectors(parObject,modelInit)
