#!/usr/bin/env python3
"""
GPU-based elastic isotropic velocity-stress wave-equation source estimation script (single-precision version)

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
import pyOperator as pyOp
import Elastic_iso_float_prop
import numpy as np
import time
from dataCompModule import ElasticDatComp
from collections import Counter

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import pyStepper as Stepper
import inversionUtils
from sys_util import logger

#Dask-related modules
import pyDaskOperator as DaskOp


# Source component selection operator
class ElasticSouComp(pyOp.Operator):
  """
	   Operator for sampling elastic source components
	"""

  def __init__(self, components, range):
    """
		   Constructor for resampling elastic source components
		   components = [no default] - string; Comma-separated list of the output/sampled components (e.g., 'fx,fz,Mxx,Mzz,Mxz' or 'fx,fz'; the order matters!). Currently, supported: fx,fz,Mxx,Mzz,Mxz
		   range     = [no default] - vector class; Original elastic source
		"""
    #Getting axes from domain vector
    timeAxis = range.getHyper().getAxis(1)
    SimAxis = range.getHyper().getAxis(2)  # Spatial extent of the source
    shotAxis = range.getHyper().getAxis(4)
    #Getting number of output components
    self.comp_list = components.split(",")
    if any(elem > 1 for elem in Counter(self.comp_list).values()):
      raise ValueError(
          "ERROR! A component was provided multiple times! Check your input arguments"
      )
    #Creating component axis for output vector
    compAxis = Hypercube.axis(n=len(self.comp_list),
                              label="Component %s" % (components))
    domain = SepVector.getSepVector(
        Hypercube.hypercube(axes=[timeAxis, SimAxis, compAxis, shotAxis]))
    #Setting domain and range
    self.setDomainRange(domain, range)
    #Making the list lower case
    self.comp_list = [elem.lower() for elem in self.comp_list]
    if not any([
        "fx" in self.comp_list, "fz" in self.comp_list, "mxx" in self.comp_list,
        "mzz" in self.comp_list, "mxz" in self.comp_list
    ]):
      raise ValueError("ERROR! Provided unknown source components: %s" %
                       (components))

    return

  def forward(self, add, model, data):
    """
		   Forward operator: sampled source -> elastic source
		"""
    self.checkDomainRange(model, data)
    if (not add):
      data.zero()
    modelNd = model.getNdArray()
    dataNd = data.getNdArray()
    #Checking if fx was requested to be sampled
    if ("fx" in self.comp_list):
      idx = self.comp_list.index("fx")
      dataNd[:, 0, :, :] += modelNd[:, idx, :, :]
    #Checking if fz was requested to be sampled
    if ("fz" in self.comp_list):
      idx = self.comp_list.index("fz")
      dataNd[:, 1, :, :] += modelNd[:, idx, :, :]
    #Checking if Mxx (normal stress) was requested to be sampled
    if ("mxx" in self.comp_list):
      idx = self.comp_list.index("mxx")
      dataNd[:, 2, :, :] += modelNd[:, idx, :, :]
    #Checking if Mzz (normal stress) was requested to be sampled
    if ("mzz" in self.comp_list):
      idx = self.comp_list.index("mzz")
      dataNd[:, 3, :, :] += modelNd[:, idx, :, :]
    #Checking if Mzz (normal stress) was requested to be sampled
    if ("mxz" in self.comp_list):
      idx = self.comp_list.index("mxz")
      dataNd[:, 4, :, :] += modelNd[:, idx, :, :]
    return

  def adjoint(self, add, model, data):
    """
		   Adjoint operator: elastic source -> sampled source
		"""
    self.checkDomainRange(model, data)
    if (not add):
      model.zero()
    modelNd = model.getNdArray()
    dataNd = data.getNdArray()
    #Checking if fx was requested to be sampled
    if ("fx" in self.comp_list):
      idx = self.comp_list.index("fx")
      modelNd[:, idx, :, :] += dataNd[:, 0, :, :]
    #Checking if fz was requested to be sampled
    if ("fz" in self.comp_list):
      idx = self.comp_list.index("fz")
      modelNd[:, idx, :, :] += dataNd[:, 1, :, :]
    #Checking if Mxx (normal stress) was requested to be sampled
    if ("mxx" in self.comp_list):
      idx = self.comp_list.index("mxx")
      modelNd[:, idx, :, :] += dataNd[:, 2, :, :]
    #Checking if Mzz (normal stress) was requested to be sampled
    if ("mzz" in self.comp_list):
      idx = self.comp_list.index("mzz")
      modelNd[:, idx, :, :] += dataNd[:, 3, :, :]
    #Checking if Mzz (normal stress) was requested to be sampled
    if ("mxz" in self.comp_list):
      idx = self.comp_list.index("mxz")
      modelNd[:, idx, :, :] += dataNd[:, 4, :, :]
    return


if __name__ == '__main__':
  #Printing documentation if no arguments were provided
  if (len(sys.argv) == 1):
    print(__doc__)
    quit(0)

  #Getting parameter object
  parObject = genericIO.io(params=sys.argv)

  # Checking if Dask was requested
  client, nWrks = Elastic_iso_float_prop.create_client(parObject)

  # Initialize operator
  modelFloat, dataFloat, elasticParamFloat, parObject1, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, modelFloatLocal = Elastic_iso_float_prop.nonlinearOpInitFloat(
      sys.argv, client)

  # Initialize parameters for inversion
  stop, logFile, saveObj, saveRes, saveGrad, saveModel, prefix, bufferSize, iterSampling, restartFolder, flushMemory, info = inversionUtils.inversionInit(
      sys.argv)
  inv_log = logger(logFile)
  solver = parObject.getString("solver", "LCG")  #[LCG,LSQR]

  print("-------------------------------------------------------------------")
  print("------------------ Running Python Source Estimation ---------------")
  print("-------------------------------------------------------------------\n")

  ########################## Operator/Vectors ################################

  if (client):
    #Instantiating Dask Operator
    nlOp_args = [(modelFloat.vecDask[iwrk], dataFloat.vecDask[iwrk],
                  elasticParamFloat[iwrk], parObject1[iwrk],
                  sourcesVectorCenterGrid[iwrk], sourcesVectorXGrid[iwrk],
                  sourcesVectorZGrid[iwrk], sourcesVectorXZGrid[iwrk],
                  recVectorCenterGrid[iwrk], recVectorXGrid[iwrk],
                  recVectorZGrid[iwrk], recVectorXZGrid[iwrk])
                 for iwrk in range(nWrks)]
    nonlinearElasticOp = DaskOp.DaskOperator(
        client, Elastic_iso_float_prop.nonlinearPropElasticShotsGpu, nlOp_args,
        [1] * nWrks)
    #Adding spreading operator and concatenating with non-linear operator (using modelFloatLocal)
    Sprd = DaskOp.DaskSpreadOp(client, modelFloatLocal, [1] * nWrks)
    invOp = pyOp.ChainOperator(Sprd, nonlinearElasticOp)
  else:
    # Construct nonlinear operator object
    nonlinearElasticOp = Elastic_iso_float_prop.nonlinearPropElasticShotsGpu(
        modelFloat, dataFloat, elasticParamFloat, parObject,
        sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
        sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid,
        recVectorZGrid, recVectorXZGrid)
    invOp = nonlinearElasticOp

  ############################# Read data ####################################

  dataFile = parObject.getString("data", "noDataFile")
  if (dataFile == "noDataFile"):
    raise IOError("**** ERROR: User did not provide data file name ****\n")

  #Reading model
  data = genericIO.defaultIO.getVector(dataFile, ndims=4)

  ########################### Data components ################################
  comp = parObject.getString("comp")
  if (comp != "vx,vz,sxx,szz,sxz"):
    if client:
      sampOp_args = [(comp, nonlinearElasticOp.getRange().vecDask[iwrk])
                     for iwrk in range(nWrks)]
      sampOp = DaskOp.DaskOperator(client, ElasticDatComp, sampOp_args,
                                   [1] * nWrks)
      #Dask interface
      if (client):
        #Chunking the data and spreading them across workers if dask was requested
        data = Elastic_iso_float_prop.chunkData(data, sampOp.getRange())
    else:
      sampOp = ElasticDatComp(comp, nonlinearElasticOp.getRange())
      #Necessary to Fix strange checkSame error in pyVector
      dataTmp = sampOp.getRange().clone()
      dataTmp.getNdArray()[:] = data.getNdArray()
      data = dataTmp
    #modeling operator = S*F
    invOp = pyOp.ChainOperator(invOp, sampOp)
  else:
    #Dask interface
    if (client):
      #Chunking the data and spreading them across workers if dask was requested
      data = Elastic_iso_float_prop.chunkData(data,
                                              nonlinearElasticOp.getRange())
    if (not dataFloat.checkSame(data)):
      raise ValueError(
          "ERROR! The input data have different size of the expected inversion data! Check your arguments and paramater file"
      )

  ########################### Data Masking Op ################################

  dataMaskFile = parObject.getString("dataMask", "noDataMask")

  if dataMaskFile != "noDataMask":
    print("----- Provided data mask -----")
    dataMask = genericIO.defaultIO.getVector(dataMaskFile, ndims=4)
    #Necessary to Fix strange checkSame error in pyVector
    dataTmp = data.clone()
    dataTmp.getNdArray()[:] = dataMask.getNdArray()
    dataMask = dataTmp
    if client:
      dataMask = Elastic_iso_float_prop.chunkData(dataMask, data)
    if not dataMask.checkSame(data):
      raise ValueError(
          "ERROR! Data mask file inconsistent with observed data vector")
    dataMask = pyOp.DiagonalOp(dataMask)
    #modeling operator = M*F
    invOp = pyOp.ChainOperator(invOp, dataMask)

  ########################### Source components ##############################

  compSou = parObject.getString("compSou")
  if (compSou != "fx,fz,Mxx,Mzz,Mxz"):
    sampSouOp = ElasticSouComp(compSou, invOp.getDomain())
    invOp = pyOp.ChainOperator(sampSouOp, invOp)
  else:
    if (not modelFloatLocal.checkSame(modelInit)):
      raise ValueError(
          "ERROR! The input data have different size of the expected inversion data! Check your arguments and paramater file"
      )

  # Read initial model
  modelInitFile = parObject.getString("modelInit", "None")
  if (modelInitFile == "None"):
    modelInit = invOp.getDomain().clone()
    modelInit.scale(0.0)
  else:
    modelInit = genericIO.defaultIO.getVector(modelInitFile, ndims=4)

  ############################## Problem ######################################

  invProb = Prblm.ProblemL2Linear(modelInit, data, invOp)

  ############################## Solver ######################################
  # Solver
  if solver == "LCG":
    Linsolver = LCG(stop, logger=inv_log)
  elif solver == "LSQR":
    Linsolver = LSQR(stop, logger=inv_log)
  else:
    raise ValueError("Unknown solver: %s" % (solver))
  Linsolver.setDefaults(save_obj=saveObj,
                        save_res=saveRes,
                        save_grad=saveGrad,
                        save_model=saveModel,
                        prefix=prefix,
                        iter_buffer_size=bufferSize,
                        iter_sampling=iterSampling,
                        flush_memory=flushMemory)

  # Run solver
  Linsolver.run(invProb, verbose=True)

  print("-------------------------------------------------------------------")
  print("--------------------------- All done ------------------------------")
  print("-------------------------------------------------------------------\n")
