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
import SpaceInterpMultiDouble
import StaggerDouble
import PadTruncateSourceDouble
import pyOperator as Op


def forcing_term_op_init(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpMultiDouble.space_interp_multi_init_source(args)

	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpMultiOp = SpaceInterpMultiDouble.space_interp_multi(zCoord,xCoord,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)


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

	input = SepVector.getSepVector(irregSourceHyper,storage="dataDouble")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataDouble")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataDouble")
	sourceGridPositions = spaceInterpMultiOp.getRegPosUniqueVector()

	padTruncateSourceOp = PadTruncateSourceDouble.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions)

	#stagger op
	staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	output = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	wavefieldStaggerOp=Stagger.stagger_wfld(staggerDummyModel,output)

	#chain operators
	spaceInterpMultiOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpMultiOp = Op.Transpose(spaceInterpMultiOp)
	PK_adj = Op.ChainOperator(spaceInterpMultiOp,padTruncateSourceOp)
	SPK_adj = Op.ChainOperator(PK_adj,wavefieldStaggerOp)

	#read in source
	waveletDouble = SepVector.getSepVector(SPK_adj.getDomain().getHyper(),storage="dataDouble")
	prior = SepVector.getSepVector(SPK_adj.getRange().getHyper(),storage="dataDouble")
	waveletFile=parObject.getString("wavelet")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletFloat.getNdArray())
	waveletDMat=waveletDouble.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		waveletDMat[:,:,iShot] = waveletSMatT

	return SPK_adj,prior

#
def data_extraction_op_init(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpMultiDouble.space_interp_multi_init_rec(args)

	#interp operator instantiate
	#check which rec injection interp method
	recInterpMethod = parObject.getString("recInterpMethod","linear")
	recInterpNumFilters = parObject.getInt("recInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpMultiOp = SpaceInterpMultiDouble.space_interp_multi(zCoord,xCoord,centerHyper,nt,recInterpMethod,recInterpNumFilters)

	# pad truncate init
	dts = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dts)
	wfldAxis=Hypercube.axis(n=5,o=0.0,d=1)
	regRecAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceReg(),o=0.0,d=1)
	irregRecAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceIrreg(),o=0.0,d=1)
	regRecHyper=Hypercube.hypercube(axes=[regRecAxis,wfldAxis,tAxis])
	irregRecHyper=Hypercube.hypercube(axes=[irregRecAxis,wfldAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),wfldAxis,tAxis])

	output = SepVector.getSepVector(irregRecHyper,storage="dataDouble")
	padTruncateDummyModel = SepVector.getSepVector(regRecHyper,storage="dataDouble")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataDouble")
	recGridPositions = spaceInterpMultiOp.getRegPosUniqueVector()
	padTruncateRecOp = PadTruncateSourceDouble.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,recGridPositions)
	padTruncateRecOp = Op.Transpose(padTruncateRecOp)

	#stagger op
	staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	input = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataDouble")
	wavefieldStaggerOp=StaggerDouble.stagger_wfld(staggerDummyModel,input)
	wavefieldStaggerOp=Op.Transpose(wavefieldStaggerOp)

	#chain operators
	spaceInterpMultiOp.setDomainRange(padTruncateDummyModel,output)
	#spaceInterpMultiOp = Op.Transpose(spaceInterpMultiOp)
	P_adjS_adj = Op.ChainOperator(wavefieldStaggerOp,padTruncateRecOp)
	KP_adjS_adj = Op.ChainOperator(P_adjS_adj,spaceInterpMultiOp)

	#apply forward
	return KP_adjS_adj
