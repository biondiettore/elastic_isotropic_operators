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
from dataCompModule import ElasticDatComp
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
		if(minBound1 == minBound2 == minBound3 == -np.inf):
			minBoundVector = None
		else:
			minBoundVector=model.clone()
			minBoundVector.set(0.0)
			minBoundVectorNd=minBoundVector.getNdArray()
			minBoundVectorNd[0,:,:]=minBound1
			minBoundVectorNd[1,:,:]=minBound2
			minBoundVectorNd[2,:,:]=minBound3
	else:
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound1=parObject.getFloat("maxBound_par1",np.inf)
		maxBound2=parObject.getFloat("maxBound_par2",np.inf)
		maxBound3=parObject.getFloat("maxBound_par3",np.inf)
		if(maxBound1 == maxBound2 == maxBound3 == np.inf):
			maxBoundVector = None
		else:
			maxBoundVector=model.clone()
			maxBoundVector.set(0.0)
			maxBoundVectorNd=maxBoundVector.getNdArray()
			maxBoundVectorNd[0,:,:]=maxBound1
			maxBoundVectorNd[1,:,:]=maxBound2
			maxBoundVectorNd[2,:,:]=maxBound3

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
	modelInit,dataFloat,sourcesFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid = Elastic_iso_float_prop.nonlinearFwiOpInitFloat(sys.argv)

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

	########################### Data components ################################
	comp = parObject.getString("comp")
	if(comp != "vx,vz,sxx,szz,sxz"):
		sampOp = ElasticDatComp(comp,dataFloat)
		sampOpNl = pyOp.NonLinearOperator(sampOp,sampOp)
	else:
		if(not dataFloat.checkSame(data)):
			raise ValueError("ERROR! The input data have different size of the expected inversion data! Check your arguments and paramater file")
		dataFloat = data
		sampOpNl = None

	############################# Instanciation ################################
	# Nonlinear
	nonlinearElasticOp=Elastic_iso_float_prop.nonlinearFwiPropElasticShotsGpu(modelInit,dataFloat,sourcesFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

	# Construct nonlinear operator object
	BornElasticOp=Elastic_iso_float_prop.BornElasticShotsGpu(modelInit,dataFloat,modelInit,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

	# Conventional FWI non-linear operator
	fwiInvOp=pyOp.NonLinearOperator(nonlinearElasticOp,BornElasticOp,BornElasticOp.setBackground)

	#Elastic parameter conversion if any
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv.ElasticConv(modelInit,mod_par)
		#Jacobian
		convOpJac = ElaConv.ElasticConvJab(modelInit,modelInit,mod_par)
		#Creating non-linear operator
		convOpNl=pyOp.NonLinearOperator(convOp,convOpJac,convOpJac.setBackground)
		#Chanining non-linear operators if not using Lame,Mu,Density parameterization
		#f(g(m)) where f is the non-linear modeling operator and g is the non-linear change of variables
		fwiInvOp=pyOp.CombNonlinearOp(convOpNl,fwiInvOp)

	#Sampling of elastic data if necessary
	if (sampOpNl):
		#modeling operator = Sf(m)
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,sampOpNl)

	############################# Gradient mask ################################
	maskGradientFile=parObject.getString("maskGradient","NoMask")
	if (maskGradientFile=="NoMask"):
		maskGradient=None
	else:
		if(pyinfo): print("--- User provided a mask for the gradients ---")
		inv_log.addToLog("--- User provided a mask for the gradients ---")
		maskGradient=genericIO.defaultIO.getVector(maskGradientFile)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=createBoundVectors(parObject,modelInit)

	########################### Inverse Problem ################################
	fwiProb=Prblm.ProblemL2NonLinear(modelInit,data,fwiInvOp,grad_mask=maskGradient,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG.NLCGsolver(stop,logger=inv_log)
	# LBFGS
	elif (solverType=="lbfgs"):
		nlSolver=LBFGS.LBFGSsolver(stop,logger=inv_log)
	# Steepest descent
	elif (solverType=="sd"):
		nlSolver=NLCG.NLCGsolver(stop,beta_type="SD",logger=inv_log)
	else:
		raise ValueError("ERROR! Provided unknonw solver type: %s"%(solverType))

	############################# Stepper ######################################
	if (stepper == "parabolic"):
		nlSolver.stepper.eval_parab=True
	elif (stepper == "linear"):
		nlSolver.stepper.eval_parab=False
	elif (stepper == "parabolicNew"):
		nlSolver.stepper = Stepper.ParabolicStepConst()
	else:
		raise ValueError("ERROR! Provided unknonw stepper type: %s"%(stepper))

	####################### Manual initial step length #########################
	initStep=parObject.getFloat("initStep",-1.0)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	nlSolver.run(fwiProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
