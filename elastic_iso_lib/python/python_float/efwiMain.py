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


# Elastic FWI workflow script
if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)
