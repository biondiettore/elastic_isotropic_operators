#!/usr/bin/env python3
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_double_prop
import numpy as np
import time

if __name__ == '__main__':
  # Initialize operator
  modelDouble, dataDouble, elasticParamDouble, parObject, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid = Elastic_iso_double_prop.nonlinearOpInitDouble(
      sys.argv)

  # Construct nonlinear operator object
  nonlinearElasticOp = Elastic_iso_double_prop.nonlinearPropElasticShotsGpu(
      modelDouble, dataDouble, elasticParamDouble, parObject.param,
      sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
      sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid,
      recVectorXZGrid)

  #Testing dot-product test of the operator
  if (parObject.getInt("dpTest", 0) == 1):
    nonlinearElasticOp.dotTest(True)
    quit()

  # Forward
  if (parObject.getInt("adj", 0) == 0):

    print("-------------------------------------------------------------------")
    print("------------------ Running Python nonlinear forward ---------------")
    print(
        "-------------------------------------------------------------------\n")

    # Check that model was provided
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      raise IOError("**** ERROR: User did not provide model file ****\n")
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      raise IOError("**** ERROR: User did not provide data file name ****\n")

    sourceGeomFile = parObject.getString("sourceGeomFile", "None")
    if sourceGeomFile != "None":
      modelTemp = genericIO.defaultIO.getVector(modelFile)
      modelDouble.getNdArray()[0, :, :, :] = modelTemp.getNdArray()
    else:
      modelFloat = genericIO.defaultIO.getVector(modelFile)
      modelDMat = modelDouble.getNdArray()
      modelSMat = modelFloat.getNdArray()
      modelDMat[0, :, 0, :] = modelSMat

    #check if we want to save wavefield
    if (parObject.getInt("saveWavefield", 0) == 1):
      wfldFile = parObject.getString("wfldFile", "noWfldFile")
      if (wfldFile == "noWfldFile"):
        raise IOError(
            "**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****"
        )
      #run Nonlinear forward with wavefield saving
      nonlinearElasticOp.forwardWavefield(False, modelDouble, dataDouble)
      #save wavefield to disk
      wavefieldDouble = nonlinearElasticOp.getWavefield()
      wavefieldFloat = SepVector.getSepVector(wavefieldDouble.getHyper(),
                                              storage="dataFloat")
      wavefieldFloatNp = wavefieldFloat.getNdArray()
      wavefieldDoubleNp = wavefieldDouble.getNdArray()
      wavefieldFloatNp[:] = wavefieldDoubleNp
      genericIO.defaultIO.writeVector(wfldFile, wavefieldFloat)
    else:
      #run Nonlinear forward without wavefield saving
      nonlinearElasticOp.forward(False, modelDouble, dataDouble)
    #write data to disk
    dataFloat = SepVector.getSepVector(dataDouble.getHyper(),
                                       storage="dataFloat")
    dataFloatNp = dataFloat.getNdArray()
    dataDoubleNp = dataDouble.getNdArray()
    dataFloatNp[:] = dataDoubleNp
    genericIO.defaultIO.writeVector(dataFile, dataFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print(
        "-------------------------------------------------------------------\n")

  # Adjoint
  else:

    print("-------------------------------------------------------------------")
    print("------------------ Running Python nonlinear adjoint ---------------")
    print(
        "-------------------------------------------------------------------\n")

    # Check that model was provided
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      raise IOError("**** ERROR: User did not provide model file ****\n")
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      raise IOError("**** ERROR: User did not provide data file name ****\n")

    #Reading model
    dataFloat = genericIO.defaultIO.getVector(dataFile, ndims=4)
    dataDMat = dataDouble.getNdArray()
    dataSMat = dataFloat.getNdArray()
    dataDMat[:] = dataSMat

    #check if we want to save wavefield
    if (parObject.getInt("saveWavefield", 0) == 1):
      wfldFile = parObject.getString("wfldFile", "noWfldFile")
      if (wfldFile == "noWfldFile"):
        raise IOError(
            "**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****"
        )
      #run Nonlinear adjoint with wavefield saving
      nonlinearElasticOp.adjointWavefield(False, modelDouble, dataDouble)
      #save wavefield to disk
      wavefieldDouble = nonlinearElasticOp.getWavefield()
      wavefieldFloat = SepVector.getSepVector(wavefieldDouble.getHyper(),
                                              storage="dataFloat")
      wavefieldFloatNp = wavefieldFloat.getNdArray()
      wavefieldDoubleNp = wavefieldDouble.getNdArray()
      wavefieldFloatNp[:] = wavefieldDoubleNp
      genericIO.defaultIO.writeVector(wfldFile, wavefieldFloat)
    else:
      #run Nonlinear forward without wavefield saving
      nonlinearElasticOp.adjoint(False, modelDouble, dataDouble)
    #write data to disk
    modelFloat = SepVector.getSepVector(modelDouble.getHyper(),
                                        storage="dataFloat")
    modelFloatNp = modelFloat.getNdArray()
    modelDoubleNp = modelDouble.getNdArray()
    modelFloatNp[:] = modelDoubleNp
    modelFloat.writeVec(modelFile)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print(
        "-------------------------------------------------------------------\n")
