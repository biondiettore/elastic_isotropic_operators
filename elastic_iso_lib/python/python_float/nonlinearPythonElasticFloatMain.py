#!/usr/bin/env python3.5
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_float_nl
import numpy as np
import time

if __name__ == '__main__':
    # Initialize operator
    modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid = Elastic_iso_float_nl.nonlinearOpInitFloat(sys.argv)

    # Construct nonlinear operator object
    nonlinearElasticOp=Elastic_iso_float_nl.nonlinearPropElasticShotsGpu(modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

    # Forward
    if (parObject.getInt("adj",0) == 0):

        print("-------------------------------------------------------------------")
        print("------------------ Running Python nonlinear forward ---------------")
        print("-------------------------------------------------------------------\n")

        # Check that model was provided
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file ****\n")
            quit()
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file name ****\n")
            quit()
        #modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
        modelTemp=genericIO.defaultIO.getVector(modelFile)
        modelFMat=modelFloat.getNdArray()
        modelTMat=modelTemp.getNdArray()
        modelFMat[0,:,0,:]=modelTMat

        domain_hyper=nonlinearElasticOp.domain.getHyper()
        model_hyper=modelFloat.getHyper()
        range_hyper=nonlinearElasticOp.range.getHyper()
        data_hyper=dataFloat.getHyper()

        #check if we want to save wavefield
        if (parObject.getInt("saveWavefield",0) == 1):
            wfldFile=parObject.getString("wfldFile","noWfldFile")
            if (wfldFile == "noWfldFile"):
                print("**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****\n")
                quit()
            #run Nonlinear forward with wavefield saving
            nonlinearElasticOp.forwardWavefield(False,modelFloat,dataFloat)
            #save wavefield to disk
            wavefieldFloat = nonlinearElasticOp.getWavefield()
            genericIO.defaultIO.writeVector(wfldFile,wavefieldFloat)
        else:
            #run Nonlinear forward without wavefield saving
            nonlinearElasticOp.forward(False,modelFloat,dataFloat)
        #write data to disk
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

    # Adjoint
    else:
        raise NotImplementedError("ERROR! Adjoint operator not implemented yet!")
