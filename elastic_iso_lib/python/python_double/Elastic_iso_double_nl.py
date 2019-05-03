#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyElastic_iso_double_nl
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

from pyElastic_iso_double_nl import spaceInterpGpu

############################ Acquisition geometry ##############################
# Build sources geometry
def buildSourceGeometry(parObject,elasticParam):

	# Horizontal axis
	nx=elasticParam.getHyper().axes[1].n
	dx=elasticParam.getHyper().axes[1].d
	ox=elasticParam.getHyper().axes[1].o

	# vertical axis
	nz=elasticParam.getHyper().axes[0].n
	dz=elasticParam.getHyper().axes[0].d
	oz=elasticParam.getHyper().axes[0].o

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nExp"),o=ox+oxSource*dx,d=spacingShots*dx)

	##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=elasticParam.getHyper().axes[0].n,o=elasticParam.getHyper().axes[0].o,d=elasticParam.getHyper().axes[0].d)
	zAxisShifted=Hypercube.axis(n=elasticParam.getHyper().axes[0].n,o=elasticParam.getHyper().axes[0].o-0.5*elasticParam.getHyper().axes[0].d,d=elasticParam.getHyper().axes[0].d)

	xAxis=Hypercube.axis(n=elasticParam.getHyper().axes[1].n,o=elasticParam.getHyper().axes[1].o,d=elasticParam.getHyper().axes[1].d)
	xAxisShifted=Hypercube.axis(n=elasticParam.getHyper().axes[1].n,o=elasticParam.getHyper().axes[1].o-0.5*elasticParam.getHyper().axes[1].d,d=elasticParam.getHyper().axes[1].d)

	paramAxis=Hypercube.axis(n=elasticParam.getHyper().axes[2].n,o=elasticParam.getHyper().axes[2].o,d=elasticParam.getHyper().axes[2].d)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,paramAxis])
	xGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,paramAxis])
	zGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,paramAxis])
	xzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxisShifted,paramAxis])

	#need a sources vector for centerGrid, x shifted, z shifted, and xz shifted grid
	sourcesVectorCenterGrid=[]
	sourcesVectorXGrid=[]
	sourcesVectorZGrid=[]
	sourcesVectorXZGrid=[]

	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

	for ishot in range(parObject.getInt("nExp")):
		# sources _zCoord and _xCoord
		sourceAxisVertical=Hypercube.axis(n=nzSource,o=0.0,d=1.0)
		zCoordHyper=Hypercube.hypercube(axes=[sourceAxisVertical])
		zCoordDouble=SepVector.getSepVector(zCoordHyper,storage="dataDouble")
		sourceAxisHorizontal=Hypercube.axis(n=nxSource,o=0.0,d=1.0)
		xCoordHyper=Hypercube.hypercube(axes=[sourceAxisHorizontal])
		xCoordDouble=SepVector.getSepVector(xCoordHyper,storage="dataDouble")

		#Setting z and x position of the source for the given experiment
		zCoordDouble.set(oz+ozSource*dz)
		xCoordDouble.set(ox+oxSource*dx)

		sourcesVectorCenterGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),centerGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorXGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),zGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorXZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xzGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		# sourcesVectorCenterGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),centerGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		# sourcesVectorXGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		# sourcesVectorZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),zGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		# sourcesVectorXZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xzGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))

		oxSource=oxSource+spacingShots # Shift source

	return sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis

# Build receivers geometry
def buildReceiversGeometry(parObject,elasticParam):

	# Horizontal axis
	nx=elasticParam.getHyper().axes[1].n
	dx=elasticParam.getHyper().axes[1].d
	ox=elasticParam.getHyper().axes[1].o

	# Vertical axis
	nz=elasticParam.getHyper().axes[0].n
	dz=elasticParam.getHyper().axes[0].d
	oz=elasticParam.getHyper().axes[0].o

	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzReceiver=0
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)
	nRecGeom=1; # Constant receivers' geometry

	##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=elasticParam.getHyper().axes[0].n,o=elasticParam.getHyper().axes[0].o,d=elasticParam.getHyper().axes[0].d)
	zAxisShifted=Hypercube.axis(n=elasticParam.getHyper().axes[0].n,o=elasticParam.getHyper().axes[0].o-0.5*elasticParam.getHyper().axes[0].d,d=elasticParam.getHyper().axes[0].d)

	xAxis=Hypercube.axis(n=elasticParam.getHyper().axes[1].n,o=elasticParam.getHyper().axes[1].o,d=elasticParam.getHyper().axes[1].d)
	xAxisShifted=Hypercube.axis(n=elasticParam.getHyper().axes[1].n,o=elasticParam.getHyper().axes[1].o-0.5*elasticParam.getHyper().axes[1].d,d=elasticParam.getHyper().axes[1].d)

	paramAxis=Hypercube.axis(n=elasticParam.getHyper().axes[2].n,o=elasticParam.getHyper().axes[2].o,d=elasticParam.getHyper().axes[2].d)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,paramAxis])
	xGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,paramAxis])
	zGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,paramAxis])
	xzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxisShifted,paramAxis])

	#need a receiver vector for centerGrid, x shifted, z shifted, and xz shifted grid
	recVectorCenterGrid=[]
	recVectorXGrid=[]
	recVectorZGrid=[]
	recVectorXZGrid=[]

	#check which source injection interp method
	recInterpMethod = parObject.getString("recInterpMethod","linear")
	recInterpNumFilters = parObject.getInt("recInterpNumFilters",4)

	# receiver _zCoord and _xCoord
	recAxisVertical=Hypercube.axis(n=nxReceiver,o=0.0,d=1.0)
	zCoordHyper=Hypercube.hypercube(axes=[recAxisVertical])
	zCoordDouble=SepVector.getSepVector(zCoordHyper,storage="dataDouble")
	zCoordDoubleNd = zCoordDouble.getNdArray()
	recAxisHorizontal=Hypercube.axis(n=nxReceiver,o=0.0,d=1.0)
	xCoordHyper=Hypercube.hypercube(axes=[recAxisHorizontal])
	xCoordDouble=SepVector.getSepVector(xCoordHyper,storage="dataDouble")
	xCoordDoubleNd = xCoordDouble.getNdArray()
	for irec in range(nxReceiver):
		zCoordDoubleNd[irec] = oz + ozReceiver*dz + dzReceiver*dz*irec
		xCoordDoubleNd[irec] = ox + oxReceiver*dx + dxReceiver*dx*irec

	for iRec in range(nRecGeom):
		recVectorCenterGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),centerGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorXGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),zGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorXZGrid.append(spaceInterpGpu(zCoordDouble.getCpp(),xCoordDouble.getCpp(),xzGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))

	return recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis

############################### Nonlinear ######################################
def nonlinearOpInitDouble(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)
	elasticParamDouble=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
	elasticParamDoubleNp=elasticParamDouble.getNdArray()
	elasticParamFloatNp=elasticParamFloat.getNdArray()
	elasticParamDoubleNp[:]=elasticParamFloatNp

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamDouble)
	recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamDouble)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model
	dummyAxis=Hypercube.axis(n=1)
	wavefieldAxis=Hypercube.axis(n=5)
	modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,wavefieldAxis,dummyAxis])
	modelDouble=SepVector.getSepVector(modelHyper,storage="dataDouble")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Outputs
	return modelDouble,dataDouble,elasticParamDouble,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid

class nonlinearPropElasticShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,elasticParam,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		self.pyOp = pyElastic_iso_double_nl.nonlinearPropElasticShotsGpu(elasticParam,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def getWavefield(self):
		wavefield = self.pyOp.getWavefield()
		return SepVector.doubleVector(fromCpp=wavefield)

	# def setVel(self,elasticParam):
	# 	#Checking if getCpp is present
	# 	if("getCpp" in dir(elasticParam)):
	# 		elasticParam = elasticParam.getCpp()
	# 	with pyElastic_iso_double_nl.ostream_redirect():
	# 		self.pyOp.setVel(elasticParam)
	# 	return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyElastic_iso_double_nl.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result