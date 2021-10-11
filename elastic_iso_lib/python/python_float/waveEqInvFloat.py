#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Elastic_iso_float_we
#import CosTaperWfldFloat
# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
from sys_util import logger


class maskComp(Op.Operator):
  """Wrapper encapsulating PYBIND11 module"""

  def __init__(self, domain, compKeep):
    #Checking if getCpp is present
    self.setDomainRange(domain, domain)
    self.compKeep = compKeep
    return

  def forward(self, add, model, data):
    self.checkDomainRange(model, data)
    if (not add):
      data.zero()
    data.getNdArray()[:, compKeep, :, :] = data.getNdArray(
    )[:, compKeep, :, :] + model.getNdArray()[:, compKeep, :, :]

    return

  def adjoint(self, add, model, data):
    self.checkDomainRange(model, data)
    self.forward(add, model, data)
    return


# Template for linearized waveform inversion workflow
if __name__ == '__main__':

  # Bullshit stuff
  io = genericIO.pyGenericIO.ioModes(sys.argv)
  ioDef = io.getDefaultIO()
  parObject = ioDef.getParamObj()
  pyinfo = parObject.getInt("pyinfo", 1)
  epsilonEval = parObject.getInt("epsilonEval", 0)
  # Initialize parameters for inversion
  stop, logFile, saveObj, saveRes, saveGrad, saveModel, prefix, bufferSize, iterSampling, restartFolder, flushMemory, info = inversionUtils.inversionInit(
      sys.argv)
  # Logger
  inv_log = logger(logFile)

  if (pyinfo):
    print("-------------------------------------------------------------------")
  if (pyinfo):
    print("------------------ elastic wavefield reconstruction --------------")
  if (pyinfo):
    print(
        "-------------------------------------------------------------------\n")
  inv_log.addToLog(
      "------------------ elastic wavefield reconstruction --------------")

  ############################# Initialization ###############################
  # Wave equation op init
  if (pyinfo):
    print(
        "--------------------------- Wave equation op init --------------------------------"
    )
  modelFloat, dataFloat, elasticParamFloat, parObject = Elastic_iso_float_we.waveEquationOpInitFloat(
      sys.argv)
  waveEquationElasticOp = Elastic_iso_float_we.waveEquationElasticGpu(
      modelFloat, dataFloat, elasticParamFloat, parObject)

  priorFile = parObject.getString("prior", "none")
  if (priorFile == "none"):
    # forcing term op
    if (pyinfo):
      print(
          "--------------------------- forcing term op init --------------------------------"
      )
    forcingTermOp, prior = wriUtilFloat.forcing_term_op_init(sys.argv)

    # scale prior
    prior.scale(1 / (elasticParamFloat.getHyper().getAxis(1).d *
                     elasticParamFloat.getHyper().getAxis(2).d))
  else:
    if (pyinfo):
      print(
          "--------------------------- reading in provided prior --------------------------------"
      )
    prior = genericIO.defaultIO.getVector(priorFile)

  ################################ DP Test ###################################
  if (parObject.getInt("dp", 0) == 1):
    print("\nModel op dp test:")
    waveEquationElasticOp.dotTest(1)

  ############################# Read files ###################################
  # Read initial model
  modelInitFile = parObject.getString("modelInit", "None")
  if (modelInitFile == "None"):
    modelInit = modelFloat.clone()
    modelInit.scale(0.0)
  else:
    modelInit = genericIO.defaultIO.getVector(modelInitFile)

  print("*** domain and range checks *** ")
  print("* Amp - f * ")
  print("Am domain: ", waveEquationElasticOp.getDomain().getNdArray().shape)
  print("p shape: ", modelInit.getNdArray().shape)
  print("Am range: ", waveEquationElasticOp.getRange().getNdArray().shape)
  print("f shape: ", prior.getNdArray().shape)

  ############################# Regularization ###############################
  epsilon = parObject.getFloat("epsScale", 1.0) * parObject.getFloat("eps", 1.0)
  invProb = Prblm.ProblemL2Linear(modelInit, prior, waveEquationElasticOp)

  ############################## Solver ######################################
  # Solver
  LCGsolver = LCG.LCGsolver(stop, logger=inv_log)
  LCGsolver.setDefaults(save_obj=saveObj,
                        save_res=saveRes,
                        save_grad=saveGrad,
                        save_model=saveModel,
                        prefix=prefix,
                        iter_buffer_size=bufferSize,
                        iter_sampling=iterSampling,
                        flush_memory=flushMemory)

  # Run solver
  if (pyinfo):
    print(
        "--------------------------- Running --------------------------------")
  LCGsolver.run(invProb, verbose=True)

  if (pyinfo):
    print("-------------------------------------------------------------------")
  if (pyinfo):
    print("--------------------------- All done ------------------------------")
  if (pyinfo):
    print(
        "-------------------------------------------------------------------\n")
