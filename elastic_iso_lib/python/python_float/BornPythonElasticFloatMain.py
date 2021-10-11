#!/usr/bin/env python3
import sys
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Elastic_iso_float_prop
import numpy as np
import time

#Dask-related modules
import pyDaskOperator as DaskOp

if __name__ == '__main__':

  #Getting parameter object
  parObject = genericIO.io(params=sys.argv)

  # Checking if Dask was requested
  client, nWrks = Elastic_iso_float_prop.create_client(parObject)

  # Initialize operator
  modelFloat, dataFloat, elasticParamFloat, parObject1, sourcesSignalsVector, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, modelFloatLocal = Elastic_iso_float_prop.BornOpInitFloat(
      sys.argv, client)

  if (client):
    #Instantiating Dask Operator
    BornOp_args = [
        (modelFloat.vecDask[iwrk], dataFloat.vecDask[iwrk],
         elasticParamFloat[iwrk], parObject1[iwrk], sourcesSignalsVector[iwrk],
         sourcesVectorCenterGrid[iwrk], sourcesVectorXGrid[iwrk],
         sourcesVectorZGrid[iwrk], sourcesVectorXZGrid[iwrk],
         recVectorCenterGrid[iwrk], recVectorXGrid[iwrk], recVectorZGrid[iwrk],
         recVectorXZGrid[iwrk]) for iwrk in range(nWrks)
    ]
    BornElasticOp = DaskOp.DaskOperator(
        client, Elastic_iso_float_prop.BornElasticShotsGpu, BornOp_args,
        [1] * nWrks)
    #Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
    Sprd = DaskOp.DaskSpreadOp(client, modelFloatLocal, [1] * nWrks)
    BornElasticOp = pyOp.ChainOperator(Sprd, BornElasticOp)
  else:
    # Construct nonlinear operator object
    BornElasticOp = Elastic_iso_float_prop.BornElasticShotsGpu(
        modelFloat, dataFloat, elasticParamFloat, parObject1,
        sourcesSignalsVector, sourcesVectorCenterGrid, sourcesVectorXGrid,
        sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid,
        recVectorXGrid, recVectorZGrid, recVectorXZGrid)

  #Testing dot-product test of the operator
  if (parObject.getInt("dpTest", 0) == 1):
    BornElasticOp.dotTest(True)
    quit()

  # Forward
  if (parObject.getInt("adj", 0) == 0):

    print(
        "----------------------------------------------------------------------"
    )
    print(
        "------------------ Running Python Born Elastic forward ---------------"
    )
    print(
        "----------------------------------------------------------------------\n"
    )

    # Check that model was provided
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      raise IOError("**** ERROR: User did not provide model file ****\n")
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      raise IOError("**** ERROR: User did not provide data file name ****\n")

    #Reading model
    modelFloat = genericIO.defaultIO.getVector(modelFile)

    # Apply forward
    BornElasticOp.forward(False, modelFloat, dataFloat)

    # Write data
    dataFloat.writeVec(dataFile)

  # Adjoint
  else:
    print(
        "----------------------------------------------------------------------"
    )
    print(
        "------------------ Running Python Born Elastic adjoint ---------------"
    )
    print(
        "----------------------------------------------------------------------\n"
    )

    # Check that model was provided
    modelFile = parObject.getString("model", "noModelFile")
    if (modelFile == "noModelFile"):
      raise IOError("**** ERROR: User did not provide model file ****\n")
    dataFile = parObject.getString("data", "noDataFile")
    if (dataFile == "noDataFile"):
      raise IOError("**** ERROR: User did not provide data file name ****\n")

    #Reading model
    dataFloat = genericIO.defaultIO.getVector(dataFile, ndims=4)
    if (client):
      #Chunking the data and spreading them across workers if dask was requested
      dataFloat = Elastic_iso_float_prop.chunkData(dataFloat,
                                                   BornElasticOp.getRange())

    # Apply forward
    BornElasticOp.adjoint(False, modelFloatLocal, dataFloat)

    # Write data
    modelFloatLocal.writeVec(modelFile)

  print("-------------------------------------------------------------------")
  print("--------------------------- All done ------------------------------")
  print("-------------------------------------------------------------------\n")
