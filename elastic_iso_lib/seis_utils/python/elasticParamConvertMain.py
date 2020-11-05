#!/usr/bin/env python3
import genericIO
import SepVector
import elasticParamConvertModule as ElaConv
import numpy as np
import sys
import os.path


if __name__ == '__main__':
	"""
	   Convert between different elastic parameters.
	   inputFile  = [no default] - string. Path to input file.
	   outputFile = [no default] - string. Path to output file.
	   conv_type    = [no default] - int; Conversion kind
		1 = VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
		2 = VpVsRho to RhoLameMu (km/s|km/s|g/cm3 -> kg/m3|Pa|Pa)
	   -1 = RhoLameMu to VpVsRho (m/s|m/s|kg/m3 <- kg/m3|Pa|Pa)
	   verbose = [0] int. verbose mode or not.
	"""
	########################## PARSE COMMAND LINE ##############################
	# IO object
	parObject=genericIO.io(params=sys.argv)

	#check if verbose
	verbose = parObject.getInt("verbose",0)
	#check conversion kind
	conv_type = parObject.getInt("conv_type",0)
	#read in file if it exists
	inputFile=parObject.getString("inputFile","noFile")
	if (not os.path.isfile(inputFile)) or (inputFile == "noFile"):
		raise IOError("ERROR! no inputFile specified or given file does not exist!")
	model = genericIO.defaultIO.getVector(inputFile,ndims=3)

	#obtain output file
	outputFile=parObject.getString("outputFile","noFile")
	if outputFile == "noFile":
		raise IOError("ERROR! no outputFile specified!")

	if(conv_type == 1):
		if(verbose): print("CONVERSION: VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)")
	elif(conv_type == -1):
		if(verbose): print("CONVERSION: RhoLameMu to VpVsRho (m/s|m/s|kg/m3 <- kg/m3|Pa|Pa)")
	elif(conv_type == 2):
		if(verbose): print("CONVERSION: RhoLameMu to VpVsRho (km/s|km/s|g/cm3 <- kg/m3|Pa|Pa)")
	else:
		raise ValueError("ERROR! Unsupported conv_type (current value: %s)!"%(self.conv_type))

	#Instantiating conversion operator
	convOp = ElaConv.ElasticConv(model,conv_type)

	#Applying conversion operator
	data = model.clone()
	convOp.forward(False,model,data)
	############################## Write output ################################
	genericIO.defaultIO.writeVector(outputFile, data)
