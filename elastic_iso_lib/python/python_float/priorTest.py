#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Elastic_iso_float_we

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init --------------------------------")
	forcingTermOp,prior = wriUtilFloat.forcing_term_op_init(sys.argv)

	if(pyinfo): print("--------------------------- writing CSPK_adj --------------------------------")
	pOutFile=parObject.getString("pOut","p.H")
	genericIO.defaultIO.writeVector(pOutFile,prior)

	if(parObject.getInt("adj",0)):
		pInFile=parObject.getString("pIn","nofile")
		# pIn = forcingTermOp.getRange().clone()
		pIn = genericIO.defaultIO.getVector(pInFile)
		fOut = forcingTermOp.getDomain().clone()
		forcingTermOp.adjoint(0,fOut,pIn)
		if(pyinfo): print("--------------------------- writing KP_adjS_adjC_adj --------------------------------")
		fOutFile=parObject.getString("fOut","f.H")
		genericIO.defaultIO.writeVector(fOutFile,fOut)


	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
