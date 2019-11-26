#!/usr/bin/env python3
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_double_we
import numpy as np
import time
import StaggerDouble

if __name__ == '__main__':
    # Initialize operator
    modelDouble,dataDouble,elasticParamDouble,parObject = Elastic_iso_double_we.waveEquationOpInitDouble(sys.argv)

    # Construct nonlinear operator object
    waveEquationElasticOp=Elastic_iso_double_we.waveEquationElasticGpu(modelDouble,dataDouble,elasticParamDouble,parObject.param)

    # Forward
    if (parObject.getInt("adj",0) == 0):

        print("-------------------------------------------------------------------")
        print("------------------ Running Python wave equation forward ---------------")
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
        modelFloat=genericIO.defaultIO.getVector(modelFile)
        modelDMat=modelDouble.getNdArray()
        modelSMat=modelFloat.getNdArray()
        modelDMat[:]=modelSMat

        domain_hyper=waveEquationElasticOp.domain.getHyper()
        model_hyper=modelDouble.getHyper()
        range_hyper=waveEquationElasticOp.range.getHyper()
        data_hyper=dataDouble.getHyper()

        #run dot product
        if (parObject.getInt("dp",0)==1):
            waveEquationElasticOp.dotTest(verb=True)

        #run Nonlinear forward without wavefield saving
        waveEquationElasticOp.forward(False,modelDouble,dataDouble)

        #if flag is set, stagger wfld back to normal grid
        if(parObject.getInt("staggerBack",0)==1):
            print("Applying stagger adjoint")
            dataDoubleShifted=SepVector.getSepVector(dataDouble.getHyper(),storage="dataDouble")
            wavefieldStaggerOp=Stagger.stagger_wfld(dataDoubleShifted,dataDouble)
            wavefieldStaggerOp.adjoint(False,dataDoubleShifted,dataDouble)
            dataDouble=dataDoubleShifted

        #if flag is set, interp sources

        #write data to disk
        dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
        dataFloatNp=dataFloat.getNdArray()
        dataDoubleNp=dataDouble.getNdArray()
        dataFloatNp[:]=dataDoubleNp
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

    # Adjoint
    else:
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
        dataFloat=genericIO.defaultIO.getVector(dataFile)
        dataDMat=dataDouble.getNdArray()
        dataSMat=dataFloat.getNdArray()
        dataDMat[:]=dataSMat

        domain_hyper=waveEquationElasticOp.domain.getHyper()
        model_hyper=modelDouble.getHyper()
        range_hyper=waveEquationElasticOp.range.getHyper()
        data_hyper=dataDouble.getHyper()

        #run dot product
        if (parObject.getInt("dp",0)==1):
            waveEquationElasticOp.dotTest(verb=True)

        #run Nonlinear forward without wavefield saving
        waveEquationElasticOp.adjoint(False,modelDouble,dataDouble)

        # #if flag is set, stagger wfld back to normal grid
        # if(parObject.getInt("staggerBack",0)==1):
        #     print("Applying stagger adjoint")
        #     dataDoubleShifted=SepVector.getSepVector(dataDouble.getHyper(),storage="dataDouble")
        #     wavefieldStaggerOp=Stagger.stagger_wfld(dataDoubleShifted,dataDouble)
        #     wavefieldStaggerOp.adjoint(False,dataDoubleShifted,dataDouble)
        #     dataDouble=dataDoubleShifted

        #if flag is set, interp sources

        #write data to disk
        modelFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
        modelFloatNp=modelFloat.getNdArray()
        modelDoubleNp=modelDouble.getNdArray()
        modelFloatNp[:]=modelDoubleNp
        genericIO.defaultIO.writeVector(modelFile,modelFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")
