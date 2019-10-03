#!/usr/bin/env python3.5
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_double_prop
import numpy as np
import time

if __name__ == '__main__':
	# Initialize operator
	modelDouble,dataDouble,elasticParamDouble,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid = Elastic_iso_double_prop.BornOpInitDouble(sys.argv)

	# Construct nonlinear operator object
	BornElasticOp=Elastic_iso_double_prop.BornElasticShotsGpu(modelDouble,dataDouble,elasticParamDouble,parObject.param,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		BornElasticOp.dotTest(True)
		quit()

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("----------------------------------------------------------------------")
		print("------------------ Running Python Born Elastic forward ---------------")
		print("----------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		#Reading model
		modelFloat=genericIO.defaultIO.getVector(modelFile)
		modelDMat=modelDouble.getNdArray()
		modelSMat=modelFloat.getNdArray()
		modelDMat[:]=modelSMat

		# Apply forward
		BornElasticOp.forward(False,modelDouble,dataDouble)

		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	# Adjoint
	else:
		print("----------------------------------------------------------------------")
		print("------------------ Running Python Born Elastic adjoint ---------------")
		print("----------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		#Reading model
		dataFloat=genericIO.defaultIO.getVector(dataFile)
		dataDMat=dataDouble.getNdArray()
		dataSMat=dataFloat.getNdArray()
		dataDMat[:]=dataSMat

		# Apply forward
		BornElasticOp.adjoint(False,modelDouble,dataDouble)

		# Write data
		modelFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp[:]=modelDoubleNp
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")
