#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyElastic_iso_float_nl
import pyElastic_iso_float_born
import pyOperator as Op
import elasticParamConvertModule as ElaConv
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

from pyElastic_iso_float_nl import spaceInterpGpu

############################ Acquisition geometry ##############################
# Reads source or receiver x,y,z locations from parFile
def parsePosParFile(PosParFile):
	nDevices = None
	devCoords = []
	with open(PosParFile,"r") as fid:
		for line in fid:
			if("#" != line[0]):
				lineCur = line.split()
				if(len(lineCur)==1):
					nDevices=float(lineCur[0])
				elif(len(lineCur)==3):
					lineCur[0] = float(lineCur[0])
					lineCur[1] = float(lineCur[1])
					lineCur[2] = float(lineCur[2])
					devCoords.append(lineCur)
				else:
					raise ValueError("Error: Incorrectly formatted line, %s, in %s"%(lineCur,PosParFile))
	if(nDevices != None):
		if (len(devCoords) != nDevices): raise ValueError("ERROR: number of devices in parfile (%d) not the same as specified nDevices (%d)"%(len(a),nDevices))
	devCoordsNdArray = np.asarray(devCoords)
	return devCoordsNdArray[:,0],devCoordsNdArray[:,2]

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
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
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
		zCoordFloat=SepVector.getSepVector(zCoordHyper,storage="dataFloat")
		sourceAxisHorizontal=Hypercube.axis(n=nxSource,o=0.0,d=1.0)
		xCoordHyper=Hypercube.hypercube(axes=[sourceAxisHorizontal])
		xCoordFloat=SepVector.getSepVector(xCoordHyper,storage="dataFloat")

		#Setting z and x position of the source for the given experiment
		zCoordFloat.set(oz+ozSource*dz)
		xCoordFloat.set(ox+oxSource*dx)

		sourcesVectorCenterGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),centerGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorXGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),zGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))
		sourcesVectorXZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xzGridHyper.getCpp(),parObject.getInt("nts"),sourceInterpMethod,sourceInterpNumFilters))

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
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
	dzReceiver=0
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
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
	recParFile = parObject.getString("recParFile","none")
	if(recParFile != "none"):
		xCoord,zCoord = parsePosParFile(recParFile)

		recAxisVertical=Hypercube.axis(n=len(zCoord),o=0.0,d=1.0)
		zCoordHyper=Hypercube.hypercube(axes=[recAxisVertical])
		zCoordFloat=SepVector.getSepVector(zCoordHyper,storage="dataFloat")
		zCoordFloatNd = zCoordFloat.getNdArray()
		zCoordFloatNd[:] = zCoord
		recAxisHorizontal=Hypercube.axis(n=len(xCoord),o=0.0,d=1.0)
		xCoordHyper=Hypercube.hypercube(axes=[recAxisHorizontal])
		xCoordFloat=SepVector.getSepVector(xCoordHyper,storage="dataFloat")
		xCoordFloatNd = xCoordFloat.getNdArray()
		xCoordFloatNd[:] = xCoord

	else:
		recAxisVertical=Hypercube.axis(n=nxReceiver,o=0.0,d=1.0)
		zCoordHyper=Hypercube.hypercube(axes=[recAxisVertical])
		zCoordFloat=SepVector.getSepVector(zCoordHyper,storage="dataFloat")
		zCoordFloatNd = zCoordFloat.getNdArray()
		recAxisHorizontal=Hypercube.axis(n=nxReceiver,o=0.0,d=1.0)
		xCoordHyper=Hypercube.hypercube(axes=[recAxisHorizontal])
		xCoordFloat=SepVector.getSepVector(xCoordHyper,storage="dataFloat")
		xCoordFloatNd = xCoordFloat.getNdArray()
		for irec in range(nxReceiver):
			zCoordFloatNd[irec] = oz + ozReceiver*dz + dzReceiver*dz*irec
			xCoordFloatNd[irec] = ox + oxReceiver*dx + dxReceiver*dx*irec

	for iRec in range(nRecGeom):
		recVectorCenterGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),centerGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorXGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),zGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))
		recVectorXZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xzGridHyper.getCpp(),parObject.getInt("nts"),recInterpMethod,recInterpNumFilters))

	return recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis

############################### Nonlinear ######################################
def nonlinearOpInitFloat(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)
	#Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
	# 0 ==> correct parameterization
	# 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv.ElasticConv(elasticParamFloat,mod_par)
		elasticParamFloatTemp = elasticParamFloat.clone()
		convOp.forward(False,elasticParamFloatTemp,elasticParamFloat)
		del elasticParamFloatTemp

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
	recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model
	dummyAxis=Hypercube.axis(n=1)
	wavefieldAxis=Hypercube.axis(n=5)
	modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,wavefieldAxis,dummyAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid

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
		self.pyOp = pyElastic_iso_float_nl.nonlinearPropElasticShotsGpu(elasticParam,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def getWavefield(self):
		wavefield = self.pyOp.getWavefield()
		return SepVector.floatVector(fromCpp=wavefield)

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyElastic_iso_float_nl.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

def nonlinearFwiOpInitFloat(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
	recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model
	dummyAxis=Hypercube.axis(n=1)
	wavefieldAxis=Hypercube.axis(n=5)
	sourceHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,wavefieldAxis,dummyAxis])
	sourceFloat=SepVector.getSepVector(sourceHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return elasticParamFloat,dataFloat,sourceFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid

class nonlinearFwiPropElasticShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,sources,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
		# 0 ==> correct parameterization
		# 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
		domainTemp = domain
		mod_par = paramP.getInt("mod_par",0)
		if(mod_par != 0):
			convOp = ElaConv.ElasticConv(domain,mod_par)
			domainTemp = domain.clone()
			convOp.forward(False,domain,domainTemp)
		#Checking if getCpp is present
		if("getCpp" in dir(domainTemp)):
			domainTemp = domainTemp.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sources)):
			sources = sources.getCpp()
			self.sources = sources.clone()
		self.pyOp = pyElastic_iso_float_nl.nonlinearPropElasticShotsGpu(domainTemp,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def forward(self,add,model,data):
		#Setting elastic model parameters
		self.setBackground(model)
		#Checking if getCpp is present
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,self.sources,data)
		return

	def setBackground(self,elasticParam):
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.setBackground(elasticParam)
		return
################################### Born #######################################
def BornOpInitFloat(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)

	#Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
	# 0 ==> correct parameterization
	# 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv.ElasticConv(elasticParamFloat,mod_par)
		elasticParamFloatTemp = elasticParamFloat.clone()
		convOp.forward(False,elasticParamFloatTemp,elasticParamFloat)
		del elasticParamFloatTemp

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
	recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wavefieldAxis=Hypercube.axis(n=5)

	# Read sources signals
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		raise IOError("**** ERROR: User did not provide seismic sources file ****")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=3)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float3DReg slices

	# Allocate model
	modelFloat=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return modelFloat,dataFloat,elasticParamFloat,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid

class BornElasticShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for elastic Born propagator"""

	def __init__(self,domain,range,elasticParam,paramP,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyElastic_iso_float_born.BornElasticShotsGpu(elasticParam,paramP,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def setBackground(self,elasticParam):
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		with pyElastic_iso_float_nl.ostream_redirect():
			self.pyOp.setBackground(elasticParam)
		return
