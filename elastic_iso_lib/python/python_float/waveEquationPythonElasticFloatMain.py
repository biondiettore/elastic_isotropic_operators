#!/usr/bin/env python3
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_float_we
import numpy as np
import time
import StaggerFloat

if __name__ == '__main__':
  # Initialize operator
  modelFloat, dataFloat, elasticParamFloat, parObject = Elastic_iso_float_we.waveEquationOpInitFloat(
      sys.argv)

  # Construct nonlinear operator object
  waveEquationElasticOp = Elastic_iso_float_we.waveEquationElasticGpu(
      modelFloat, dataFloat, elasticParamFloat, parObject.param)

  # Forward
  if (parObject.getInt("adj", 0) == 0):

    print("-------------------------------------------------------------------")
    print(
        "------------------ Running Python wave equation forward ---------------"
    )
    print(
        "-------------------------------------------------------------------\n")

    # Check that model was provided
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      print("**** ERROR: User did not provide model file ****\n")
      quit()
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      print("**** ERROR: User did not provide data file name ****\n")
      quit()
    #modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
    modelFloat = genericIO.defaultIO.getVector(modelFile)

    domain_hyper = waveEquationElasticOp.domain.getHyper()
    model_hyper = modelFloat.getHyper()
    range_hyper = waveEquationElasticOp.range.getHyper()
    data_hyper = dataFloat.getHyper()

    #run dot product
    if (parObject.getInt("dp", 0) == 1):
      waveEquationElasticOp.dotTest(verb=True)

    #run Nonlinear forward without wavefield saving
    waveEquationElasticOp.forward(False, modelFloat, dataFloat)

    #if flag is set, stagger wfld back to normal grid
    if (parObject.getInt("staggerBack", 0) == 1):
      print("Applying stagger adjoint")
      dataFloatShifted = SepVector.getSepVector(dataFloat.getHyper(),
                                                storage="dataFloat")
      wavefieldStaggerOp = StaggerFloat.stagger_wfld(dataFloatShifted,
                                                     dataFloat)
      wavefieldStaggerOp.adjoint(False, dataFloatShifted, dataFloat)
      dataFloat = dataFloatShifted

    #write data to disk

    genericIO.defaultIO.writeVector(dataFile, dataFloat)

  # Adjoint
  else:
    # Check that data was provided
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      print("**** ERROR: User did not provide data file ****\n")
      quit()
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      print("**** ERROR: User did not provide model file name ****\n")
      quit()
    #modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
    dataFloat = genericIO.defaultIO.getVector(dataFile)

    domain_hyper = waveEquationElasticOp.domain.getHyper()
    model_hyper = modelFloat.getHyper()
    range_hyper = waveEquationElasticOp.range.getHyper()
    data_hyper = dataFloat.getHyper()

    #run dot product
    if (parObject.getInt("dp", 0) == 1):
      waveEquationElasticOp.dotTest(verb=True)

    #run Nonlinear forward without wavefield saving
    waveEquationElasticOp.adjoint(False, modelFloat, dataFloat)

    #if flag is set, stagger wfld back to normal grid
    # if(parObject.getInt("staggerBack",0)==1):
    #     print("Applying stagger adjoint")
    #     dataFloatShifted=SepVector.getSepVector(dataFloat.getHyper(),storage="dataFloat")
    #     wavefieldStaggerOp=StaggerFloat.stagger_wfld(dataFloatShifted,dataFloat)
    #     wavefieldStaggerOp.adjoint(False,dataFloatShifted,dataFloat)
    #     dataFloat=dataFloatShifted

    #write data to disk

    genericIO.defaultIO.writeVector(modelFile, modelFloat)

  print("-------------------------------------------------------------------")
  print("--------------------------- All done ------------------------------")
  print("-------------------------------------------------------------------\n")
