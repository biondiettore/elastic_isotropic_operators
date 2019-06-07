#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# operators
import Elastic_iso_double_we
import SpaceInterpMulti
import Stagger
import PadTruncateSource
import pyOperator as Op

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpMulti.space_interp_multi_init_source(sys.argv)
	print(zCoord)
	print(xCoord)

	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpMultiOp = SpaceInterpMulti.space_interp_multi(zCoord,xCoord,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)


	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	wfldAxis=Hypercube.axis(n=5,o=0.0,d=1)
	regSourceAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,wfldAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,wfldAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),wfldAxis,tAxis])

	model = SepVector.getSepVector(irregSourceHyper,storage="dataDouble")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataDouble")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataDouble")
	sourceGridPositions = spaceInterpMultiOp.getRegPosUniqueVector()

	padTruncateSourceOp = PadTruncateSource.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions)

	#stagger op
	staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	data = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	wavefieldStaggerOp=Stagger.stagger_wfld(staggerDummyModel,data)

	#chain operators
	spaceInterpMultiOp.setDomainRange(padTruncateDummyModel,model)
	spaceInterpMultiOp = Op.Transpose(spaceInterpMultiOp)
	PK_adj = Op.ChainOperator(spaceInterpMultiOp,padTruncateSourceOp)
	SPK_adj = Op.ChainOperator(PK_adj,wavefieldStaggerOp)

	#forward
	if(parObject.getInt("adj")==0):
		#read in test source
		waveletFile=parObject.getString("wavelet")
		waveletFloat=genericIO.defaultIO.getVector(waveletFile)
		waveletSMat=waveletFloat.getNdArray()
		waveletSMatT=np.transpose(waveletFloat.getNdArray())
		waveletDMat=model.getNdArray()
		#loop over irreg grid sources and set each to wavelet
		for iShot in range(irregSourceAxis.n):
			waveletDMat[:,:,iShot] = waveletSMatT

		# #apply forward
		SPK_adj.forward(0,model,data)

		#write data to disk
		dataFloat=SepVector.getSepVector(data.getHyper(),storage="dataFloat")
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=data.getNdArray()
		dataFloatNp[:]=dataDoubleNp/(centerHyper.getAxis(1).d*centerHyper.getAxis(2).d)
		#dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(parObject.getString("dataFile","./test1.H"),dataFloat)
	#adjoint
	else:
		#read in test wfld
		wfldFile=parObject.getString("wfld")
		wfldFloat=genericIO.defaultIO.getVector(wfldFile)
		wfldSMat=wfldFloat.getNdArray()
		wfldDMat=data.getNdArray()
		wfldDMat[:]=wfldSMat

		# #apply forward
		SPK_adj.adjoint(0,model,data)

		#write output to disk
		modelFloat=SepVector.getSepVector(model.getHyper(),storage="dataFloat")
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp=model.getNdArray()
		print(centerHyper.getAxis(1).d)
		print(centerHyper.getAxis(2).d)
		modelFloatNp[:]=modelDoubleNp*(centerHyper.getAxis(1).d*centerHyper.getAxis(2).d)
		#modelFloatNp[:]=modelDoubleNp
		genericIO.defaultIO.writeVector(parObject.getString("modelFile","./test1.H"),modelFloat)
