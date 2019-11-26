#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Elastic_iso_double_we

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilDouble
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)
	epsilonEval=parObject.getInt("epsilonEval",0)
	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ wavefield reconstruction --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ wavefield reconstruction --------------")

	############################# Initialization ###############################
	# Data extraction
	if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
	dataSamplingOp = wriUtilDouble.data_extraction_op_init(sys.argv)

	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelDouble,dataDouble,elasticParamDouble,parObject = Elastic_iso_double_we.waveEquationOpInitDouble(sys.argv)
	waveEquationElasticOp=Elastic_iso_double_we.waveEquationElasticGpu(modelDouble,dataDouble,elasticParamDouble,parObject)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init --------------------------------")
	forcingTermOp,prior = wriUtilDouble.forcing_term_op_init(sys.argv)

	# scale prior
	prior.scale(1/(elasticParamDouble.getHyper().getAxis(1).d*elasticParamDouble.getHyper().getAxis(2).d))

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelDouble.clone()
		modelInit.scale(0.0)

	# Data
	dataFile=parObject.getString("data")
	dataFloat=genericIO.defaultIO.getVector(dataFile)
	dataDouble=SepVector.getSepVector(dataSamplingOp.getRange().getHyper(),storage="dataDouble")
	dataSMat=dataFloat.getNdArray()
	dataDMat=dataDouble.getNdArray()
	dataDMat[:]=dataSMat

	############################# Regularization ###############################
	epsilon=0
	invProb=Prblm.ProblemL2LinearReg(modelInit,dataDouble,dataSamplingOp,epsilon,reg_op=waveEquationElasticOp,prior_model=prior)

	# Evaluate Epsilon
	if (epsilonEval==1):
		if(pyinfo): print("--- Epsilon evaluation ---")
		inv_log.addToLog("--- Epsilon evaluation ---")
		epsilonOut=invProb.estimate_epsilon()
		if(pyinfo): print("--- Epsilon value: ",epsilonOut," ---")
		inv_log.addToLog("--- Epsilon value: %s ---"%(epsilonOut))
		invProb.epsilon=epsilonOut*parObject.getFloat("epsScale",1.0)


	############################## Solver ######################################
	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

	outputDouble = invProb.model
	outputFloat=SepVector.getSepVector(outputDouble.getHyper(),storage="dataFloat")
	outputFloatNp=outputFloat.getNdArray()
	outputDoubleNp=outputDouble.getNdArray()
	outputFloatNp[:]=outputDoubleNp
	genericIO.defaultIO.writeVector("./inv1/final_model_100.H",outputFloat)
