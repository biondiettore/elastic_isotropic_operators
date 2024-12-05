#!/usr/bin/env python3
"""
GPU-based elastic isotropic velocity-stress wave-equation full-waveform inversion script

USAGE EXAMPLE:
	efwiMain.py elasticParam=InitialelasticModel.H par=parEFWI.p data=elastic_data.H sources=wavelet.H nIter=1000 folder=EFWIdir solver=lbfgs saveObj=1 saveRes=0 saveGrad=0 saveModel=1 comp=p info=0

INPUT PARAMETERS:
	elasticParam = [no default] - string; Header file defining initial elastic subsurface parameters (z [m], x [m], component [see mod_par]).
	                                      These parameters must be correctly padded using the padElasticFileGpuMain program.

	sources = [no default] - string; Header file defining elastic source term (t,component=[fx -> volumetric force along x axis [N/m^3], fz -> volumetric force along z axis [N/m^3],
	                               Sxx -> injection rate within normal xx stress [Pa/s], Szz -> injection rate within normal zz stress [Pa/s],
	                               Sxz -> injection rate within shear xz stress [Pa/s]]).

	data [no default] - string; Header file of the elastic data to be inverted. Use dataCompMain.py to create this file.

	comp [no default] - string; Comma-separated list of the output/sampled components consistent with the data header file (e.g., 'vx,vz,sxx,szz,sxz' or 'p,vx,vz'; the order matters!).
	                            Currently, supported: vx,vz,sxx,szz,sxz,p (i.e., p = 0.5*(sxx+szz))

	info [0] - int; boolean; Verbosity of the program. If true, certain program information will be displayed (Set this always to 0)

	mod_par [0] - int; Choice of parameterization for the elasticParam header file. [0 = Density [Kg/m^3],Lame [Pa],Shear modulus [Pa];
	                   1 = Vp [m/s],Vs [m/s], Density [Kg/m^3];
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

	                           ################ SOLVER-RELATED VARIABLES ######################

	nIter [no default] - int; Number of solver iterations to apply.

	folder [no default] - string; Folder name in which all results and log file are written.

	prefix [no default] - string; Prefix of the result files.

	bufferSize [3] - int; Number of iteration results to keep in memory before flushing to files.

	iterSampling [10] - int; Iteration sampling. By default the script saves result every 10 iterations except for the objective function, which is save every iteration.

	saveObj [1] - int; boolean; Flag to save the objective function value into folder/prefix_obj.H

	saveRes [1] - int; boolean; Flag to save the residual vector into folder/prefix_res.H

	saveGrad [1] - int; boolean; Flag to save the gradient vector into folder/prefix_grad.H

	saveModel [1] - int; boolean; Flag to save the model vector into folder/prefix_model.H
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
# import dataTaperModule
import spatialDerivModule
import maskGradientModule

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
import inversionUtils
from sys_util import logger

#Dask-related modules
import pyDaskOperator as DaskOp
import pyDaskVector


############################ Bounds vectors ####################################
# Create bound vectors for FWI
def createBoundVectors(parObject, model, inv_log):

  # Get model dimensions
  nz = parObject.getInt("nz")
  nx = parObject.getInt("nx")

  # Min bound
  minBoundVectorFile = parObject.getString("minBoundVector",
                                           "noMinBoundVectorFile")
  if (minBoundVectorFile == "noMinBoundVectorFile"):
    minBound1 = parObject.getFloat("minBound_par1", -np.inf)
    minBound2 = parObject.getFloat("minBound_par2", -np.inf)
    minBound3 = parObject.getFloat("minBound_par3", -np.inf)
    if (minBound1 == minBound2 == minBound3 == -np.inf):
      minBoundVector = None
    else:
      if (pyinfo):
        print("--- User provided minimum bounds ---")
      inv_log.addToLog("--- User provided minimum bounds ---")
      minBoundVector = model.clone()
      minBoundVector.set(0.0)
      minBoundVectorNd = minBoundVector.getNdArray()
      minBoundVectorNd[0, :, :] = minBound1
      minBoundVectorNd[1, :, :] = minBound2
      minBoundVectorNd[2, :, :] = minBound3
  else:
    if (pyinfo):
      print("--- User provided a minimum-bound vector ---")
    inv_log.addToLog("--- User provided a minimum-bound vector ---")
    minBoundVector = genericIO.defaultIO.getVector(minBoundVectorFile)

  # Max bound
  maxBoundVectorFile = parObject.getString("maxBoundVector",
                                           "noMaxBoundVectorFile")
  if (maxBoundVectorFile == "noMaxBoundVectorFile"):
    maxBound1 = parObject.getFloat("maxBound_par1", np.inf)
    maxBound2 = parObject.getFloat("maxBound_par2", np.inf)
    maxBound3 = parObject.getFloat("maxBound_par3", np.inf)
    if (maxBound1 == maxBound2 == maxBound3 == np.inf):
      maxBoundVector = None
    else:
      if (pyinfo):
        print("--- User provided maximum bounds ---")
      inv_log.addToLog("--- User provided maximum bounds ---")
      maxBoundVector = model.clone()
      maxBoundVector.set(0.0)
      maxBoundVectorNd = maxBoundVector.getNdArray()
      maxBoundVectorNd[0, :, :] = maxBound1
      maxBoundVectorNd[1, :, :] = maxBound2
      maxBoundVectorNd[2, :, :] = maxBound3

  else:
    if (pyinfo):
      print("--- User provided a maximum-bound vector ---")
    inv_log.addToLog("--- User provided a maximum-bound vector ---")
    maxBoundVector = genericIO.defaultIO.getVector(maxBoundVectorFile)

  return minBoundVector, maxBoundVector


# Elastic FWI workflow script
if __name__ == '__main__':
  #Printing documentation if no arguments were provided
  if (len(sys.argv) == 1):
    print(__doc__)
    quit(0)

# IO object
  parObject = genericIO.io(params=sys.argv)

  # Checking if Dask was requested
  client, nWrks = Elastic_iso_float_prop.create_client(parObject)

  pyinfo = parObject.getInt("pyinfo", 1)
  spline = parObject.getInt("spline", 0)
  dataTaper = parObject.getInt("dataTaper", 0)
  # regType=parObject.getString("reg","None")
  # reg=0
  # if (regType != "None"): reg=1
  # epsilonEval=parObject.getInt("epsilonEval",0)

  # Nonlinear solver
  solverType = parObject.getString("solver")
  stepper = parObject.getString("stepper", "default")

  # Initialize parameters for inversion
  stop, logFile, saveObj, saveRes, saveGrad, saveModel, prefix, bufferSize, iterSampling, restartFolder, flushMemory, info = inversionUtils.inversionInit(
      sys.argv)

  # Logger
  inv_log = logger(logFile)

  if (pyinfo):
    print(
        "----------------------------------------------------------------------"
    )
  if (pyinfo):
    print(
        "------------------------ Elastic FWI logfile -------------------------"
    )
  if (pyinfo):
    print(
        "----------------------------------------------------------------------\n"
    )
  inv_log.addToLog(
      "------------------------ Elastic FWI logfile -------------------------")

  ############################# Initialization ###############################

  # FWI nonlinear operator
  modelInit, modelInitConv, dataFloat, sourcesFloat, parObject1, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, modelInitLocal = Elastic_iso_float_prop.nonlinearFwiOpInitFloat(
      sys.argv, client)

  # Born operator
  _, _, _, _, sourcesSignalsVector, _, _, _, _, _, _, _, _, _ = Elastic_iso_float_prop.BornOpInitFloat(
      sys.argv, client)

  ############################# Read files ###################################
  # Seismic source
  # Read within nonlinearFwiOpInitFloat

  # Data
  dataFile = parObject.getString("data", "noDataFile")
  if (dataFile == "noDataFile"):
    raise IOError("**** ERROR: User did not provide data file (data) ****\n")
  data = genericIO.defaultIO.getVector(dataFile, ndims=4)

  ############################# Gradient mask ################################
  maskGradientFile = parObject.getString("maskGradient", "NoMask")
  if (maskGradientFile == "NoMask"):
    maskGradientOp = None
  else:
    if (pyinfo):
      print("--- User provided a mask for the gradients ---")
    inv_log.addToLog("--- User provided a mask for the gradients ---")
    maskGradient = genericIO.defaultIO.getVector(maskGradientFile)
    maskGradientOp = pyOp.DiagonalOp(maskGradient)

  ############################# Instantiation ################################
  if client:
    #Spreading operator and concatenating with non-linear and born operators
    Sprd = DaskOp.DaskSpreadOp(client, modelInitLocal, [1] * nWrks)
    # Nonlinear
    nlOp_args = [(modelInitConv.vecDask[iwrk], dataFloat.vecDask[iwrk],
                  sourcesFloat.vecDask[iwrk], parObject1[iwrk],
                  sourcesVectorCenterGrid[iwrk], sourcesVectorXGrid[iwrk],
                  sourcesVectorZGrid[iwrk], sourcesVectorXZGrid[iwrk],
                  recVectorCenterGrid[iwrk], recVectorXGrid[iwrk],
                  recVectorZGrid[iwrk], recVectorXZGrid[iwrk])
                 for iwrk in range(nWrks)]
    nonlinearElasticOp = DaskOp.DaskOperator(
        client, Elastic_iso_float_prop.nonlinearFwiPropElasticShotsGpu,
        nlOp_args, [1] * nWrks)
    #Concatenating spreading and non-linear
    nonlinearElasticOp = pyOp.ChainOperator(Sprd, nonlinearElasticOp)

    # Born
    BornOp_args = [
        (modelInit.vecDask[iwrk], dataFloat.vecDask[iwrk],
         modelInitConv.vecDask[iwrk], parObject1[iwrk],
         sourcesSignalsVector[iwrk], sourcesVectorCenterGrid[iwrk],
         sourcesVectorXGrid[iwrk], sourcesVectorZGrid[iwrk],
         sourcesVectorXZGrid[iwrk], recVectorCenterGrid[iwrk],
         recVectorXGrid[iwrk], recVectorZGrid[iwrk], recVectorXZGrid[iwrk])
        for iwrk in range(nWrks)
    ]
    BornElasticOpDask = DaskOp.DaskOperator(
        client,
        Elastic_iso_float_prop.BornElasticShotsGpu,
        BornOp_args, [1] * nWrks,
        setbackground_func_name="setBackground",
        spread_op=Sprd)
    # Concatenating spreading and Born
    BornElasticOp = pyOp.ChainOperator(Sprd, BornElasticOpDask)
  else:
    # Nonlinear
    nonlinearElasticOp = Elastic_iso_float_prop.nonlinearFwiPropElasticShotsGpu(
        modelInitConv, dataFloat, sourcesFloat, parObject,
        sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
        sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid,
        recVectorZGrid, recVectorXZGrid)

    # Construct nonlinear operator object
    BornElasticOp = Elastic_iso_float_prop.BornElasticShotsGpu(
        modelInit, dataFloat, modelInitConv, parObject, sourcesSignalsVector,
        sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
        sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid,
        recVectorZGrid, recVectorXZGrid)

  #Born operator pointer for inversion
  BornElasticOpInv = BornElasticOp

  # Conventional FWI non-linear operator
  if client:
    fwiInvOp = pyOp.NonLinearOperator(nonlinearElasticOp, BornElasticOpInv,
                                      BornElasticOpDask.set_background)
  else:
    fwiInvOp = pyOp.NonLinearOperator(nonlinearElasticOp, BornElasticOpInv,
                                      BornElasticOp.setBackground)

  # Dask model vector not necessary any more
  modelInit = modelInitLocal
  #Elastic parameter conversion if any
  mod_par = parObject.getInt("mod_par", 0)
  if (mod_par != 0):
    convOp = ElaConv.ElasticConv(modelInit, mod_par)
    #Jacobian
    convOpJac = ElaConv.ElasticConvJab(modelInit, modelInit, mod_par)
    #Creating non-linear operator
    convOpNl = pyOp.NonLinearOperator(convOp, convOpJac,
                                      convOpJac.setBackground)
    #Chaining non-linear operators if not using Lame,Mu,Density parameterization
    #f(g(m)) where f is the non-linear modeling operator and g is the non-linear change of variables
    fwiInvOp = pyOp.CombNonlinearOp(convOpNl, fwiInvOp)


  if maskGradientOp is not None:
    fwiInvOp.lin_op = pyOp.ChainOperator(maskGradientOp, fwiInvOp.lin_op)

  ########################### Data components ################################
  comp = parObject.getString("comp")
  if (comp != "vx,vz,sxx,szz,sxz"):
    if client:
      sampOp_args = [(comp, BornElasticOp.getRange().vecDask[iwrk])
                     for iwrk in range(nWrks)]
      sampOp = DaskOp.DaskOperator(client, ElasticDatComp, sampOp_args,
                                   [1] * nWrks)
      #Dask interface
      if (client):
        #Chunking the data and spreading them across workers if dask was requested
        data = Elastic_iso_float_prop.chunkData(data, sampOp.getRange())
    else:
      sampOp = ElasticDatComp(comp, BornElasticOp.getRange())
    sampOpNl = pyOp.NonLinearOperator(sampOp, sampOp)
    #modeling operator = Sf(m)
    fwiInvOp = pyOp.CombNonlinearOp(fwiInvOp, sampOpNl)
  else:
    #Dask interface
    if (client):
      #Chunking the data and spreading them across workers if dask was requested
      data = Elastic_iso_float_prop.chunkData(data, BornElasticOp.getRange())
    if (not dataFloat.checkSame(data)):
      raise ValueError(
          "ERROR! The input data have different size of the expected inversion data! Check your arguments and paramater file"
      )

  ##################### Data muting with mask ################################
  dataMaskFile = parObject.getString("dataMaskFile", "noDataMask")
  if (dataMaskFile != "noDataMask"):
    if (pyinfo):
      print("--- User provided a mask for the data ---")
    inv_log.addToLog("--- User provided a mask for the data ---")
    dataMask = genericIO.defaultIO.getVector(dataMaskFile, ndims=4)
    if client:
      dataMaskD = Elastic_iso_float_prop.chunkData(dataMask,
                                                   fwiInvOp.getRange())
      dataMask_args = [dataMaskD.vecDask[iwrk] for iwrk in range(nWrks)]
      dataMaskOp = DaskOp.DaskOperator(client, pyOp.DiagonalOp, dataMask_args,
                                       [1] * nWrks)
    else:
      dataMaskOp = pyOp.DiagonalOp(dataMask)
    # Creating non-linear operator and concatenating with fwiInvOp
    dataMaskNl = pyOp.NonLinearOperator(dataMaskOp, dataMaskOp)
    fwiInvOp = pyOp.CombNonlinearOp(fwiInvOp, dataMaskNl)
    data_tmp = data.clone()
    dataMaskOp.forward(False, data, data_tmp)
    data = data_tmp

  ############################# Spline operator ##############################
  if (spline == 1):
    if (pyinfo):
      print("--- Using spline interpolation ---")
    inv_log.addToLog("--- Using spline interpolation ---")
    # Coarse-grid model
    modelCoarseInitFile = parObject.getString("modelCoarseInit")
    modelCoarseInit = genericIO.defaultIO.getVector(modelCoarseInitFile)
    modelFineInit = modelInit
    modelInit = modelCoarseInit
    # Parameter parsing
    _, _, zOrder, xOrder, zSplineMesh, xSplineMesh, zDataAxis, xDataAxis, nzParam, nxParam, scaling, zTolerance, xTolerance, _ = interpBSplineModule.bSplineIter2dInit(
        sys.argv, fat=0)
    splineOp = interpBSplineModule.bSplineIter2d(modelCoarseInit,
                                                 modelFineInit,
                                                 zOrder,
                                                 xOrder,
                                                 zSplineMesh,
                                                 xSplineMesh,
                                                 zDataAxis,
                                                 xDataAxis,
                                                 nzParam,
                                                 nxParam,
                                                 scaling,
                                                 zTolerance,
                                                 xTolerance,
                                                 fat=0)
    splineNlOp = pyOp.NonLinearOperator(
        splineOp, splineOp)  # Create spline nonlinear operator
    fwiInvOp = pyOp.CombNonlinearOp(splineNlOp, fwiInvOp)

  ############################### Bounds #####################################
  minBoundVector, maxBoundVector = createBoundVectors(parObject, modelInit,
                                                      inv_log)

  ########################### Inverse Problem ################################
  fwiProb = Prblm.ProblemL2NonLinear(modelInit,
                                     data,
                                     fwiInvOp,
                                     minBound=minBoundVector,
                                     maxBound=maxBoundVector)

  ############################# Solver #######################################
  # Nonlinear conjugate gradient
  if (solverType == "nlcg"):
    nlSolver = NLCG(stop, logger=inv_log)
  # LBFGS
  elif (solverType == "lbfgs"):
    illumination_file = parObject.getString("illumination", "noIllum")
    H0_Op = None
    if illumination_file != "noIllum":
      print("--- Using illumination as initial Hessian inverse ---")
      illumination = genericIO.defaultIO.getVector(illumination_file)
      H0_Op = pyOp.DiagonalOp(illumination)
    nlSolver = LBFGS(stop, H0=H0_Op, logger=inv_log)
  # Steepest descent
  elif (solverType == "sd"):
    nlSolver = NLCG(stop, beta_type="SD", logger=inv_log)
  else:
    raise ValueError("ERROR! Provided unknonw solver type: %s" % (solverType))

  ############################# Stepper ######################################
  if (stepper == "parabolic"):
    nlSolver.stepper.eval_parab = True
  elif (stepper == "linear"):
    nlSolver.stepper.eval_parab = False
  elif (stepper == "parabolicNew"):
    nlSolver.stepper = Stepper.ParabolicStepConst()
  elif (stepper == "default"):
    pass
  else:
    raise ValueError("ERROR! Provided unknonw stepper type: %s" % (stepper))

  ####################### Manual initial step length #########################
  initStep = parObject.getFloat("initStep", -1.0)
  if (initStep > 0):
    nlSolver.stepper.alpha = initStep

  nlSolver.setDefaults(save_obj=saveObj,
                       save_res=saveRes,
                       save_grad=saveGrad,
                       save_model=saveModel,
                       prefix=prefix,
                       iter_buffer_size=bufferSize,
                       iter_sampling=iterSampling,
                       flush_memory=flushMemory)

  # Run solver
  nlSolver.run(fwiProb, verbose=info)

  if (pyinfo):
    print("-------------------------------------------------------------------")
  if (pyinfo):
    print("--------------------------- All done ------------------------------")
  if (pyinfo):
    print(
        "-------------------------------------------------------------------\n")
