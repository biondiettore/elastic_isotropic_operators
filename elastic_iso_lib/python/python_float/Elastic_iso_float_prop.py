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
import re

from pyElastic_iso_float_nl import spaceInterpGpu

############################ Dask interface ####################################
#Dask-related modules and functions
import dask.distributed as daskD
from dask_util import DaskClient
import pyDaskVector
import re

def create_client(parObject):
	"""
	   Function to create Dask client if requested
	"""
	hostnames = parObject.getString("hostnames","noHost")
	pbs_args = parObject.getString("pbs_args","noPBS")
	lsf_args = parObject.getString("lsf_args","noLSF")
	cluster_args = None
	if pbs_args != "noPBS":
		cluster_args = pbs_args
		cluster_name = "pbs_params"
	elif lsf_args != "noLSF":
		cluster_args = lsf_args
		cluster_name = "lsf_params"
	if hostnames != "noHost" and cluster_args is not None:
		raise ValueError("Only one interface can be used for a client! User provided both SSH and PBS/LSF parameters!")
	#Starting Dask client if requested
	client = None
	nWrks = None
	args = None
	if hostnames != "noHost":
		args = {"hostnames":hostnames.split(",")}
		scheduler_file = parObject.getString("scheduler_file","noFile")
		if scheduler_file != "noFile":
			args.update({"scheduler_file_prefix":scheduler_file})
		print("Starting Dask client using the following workers: %s"%(hostnames))
	elif cluster_args:
		n_wrks = parObject.getInt("n_wrks",1)
		n_jobs = parObject.getInt("n_jobs")
		args = {"n_jobs":n_jobs}
		args.update({"n_wrks":n_wrks})
		cluster_dict={elem.split(";")[0] : elem.split(";")[1] for elem in cluster_args.split(",")}
		if "cores" in cluster_dict.keys():
			cluster_dict.update({"cores":int(cluster_dict["cores"])})
		if "mem" in cluster_dict.keys():
			cluster_dict.update({"mem":int(cluster_dict["mem"])})
		if "ncpus" in cluster_dict.keys():
			cluster_dict.update({"ncpus":int(cluster_dict["ncpus"])})
		if "nanny" in cluster_dict.keys():
			nanny_flag = True
			if cluster_dict["nanny"] in "0falseFalse":
				nanny_flag = False
			cluster_dict.update({"nanny":nanny_flag})
		if "dashboard_address" in cluster_dict.keys():
			if cluster_dict["dashboard_address"] in "Nonenone":
				cluster_dict.update({"dashboard_address":None})
		if "env_extra" in cluster_dict.keys():
			cluster_dict.update({"env_extra":cluster_dict["env_extra"].split(":")})
		if "job_extra" in cluster_dict.keys():
			cluster_dict.update({"job_extra":cluster_dict["job_extra"].split("|")})
		cluster_dict={cluster_name:cluster_dict}
		args.update(cluster_dict)
		print("Starting jobqueue Dask client using %s workers on %s jobs"%(n_wrks,n_jobs))

	if args:
		client = DaskClient(**args)
		print("Client has started!")
		nWrks = client.getNworkers()
	return client, nWrks

def parfile2pars(args):
	"""Function to expand arguments in parfile to parameters"""
	#Check if par argument was provided
	par_arg = None
	reg_comp = re.compile("(^par=)(.*)")
	match = []
	for arg in args:
		find = reg_comp.search(arg)
		if find:
			match.append(find.group(2))
	if len(match) > 0:
		par_arg = "par="+match[-1] #Taking last par argument
	#If par was found expand arguments
	if par_arg:
		par_file = par_arg.split("=")[-1]
		with open(par_file) as fid:
			lines = fid.read().splitlines()
		#Substitute par with its arguments
		idx = args.index(par_arg)
		args = args[:idx] + lines + args[idx+1:]
	return args

def create_parObj(args):
	"""Function to call genericIO correctly"""
	obj = genericIO.io(params=args)
	return obj

def spreadParObj(client,args,par):
	"""Function to spread parameter object to workers"""
	#Spreading/Instantiating the parameter objects
	List_Exps = par.getInts("nExp",0)
	parObject = []
	args1=parfile2pars(args)
	#Finding index of nShot parameter
	idx_nshot = [ii for ii,el in enumerate(args1) if "nExp" in el]
	#Removing all other nExp parameters
	for idx in idx_nshot[:-1]:
		args1.pop(idx)
	#Correcting idx_nshot
	idx_nshot = [ii for ii,el in enumerate(args1) if "nExp" in el]
	for idx,wrkId in enumerate(client.getWorkerIds()):
		#Substituting nShot with the correct number of shots
		args1[idx_nshot[-1]]="nExp=%s"%(List_Exps[idx])
		parObject.append(client.getClient().submit(create_parObj,args1,workers=[wrkId],pure=False))
	daskD.wait(parObject)
	return parObject

def call_spaceInterpGpu(zAxes,xAxes,zCoord,xCoord,GridAxes,nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift):
	"""Function instantiate a spaceInterpGpu object"""
	GridHyper = Hypercube.hypercube(axes=GridAxes)
	xCoordFloat = SepVector.getSepVector(axes=xAxes)
	zCoordFloat = SepVector.getSepVector(axes=zAxes)
	xCoordFloat.getNdArray()[:] = xCoord
	zCoordFloat.getNdArray()[:] = zCoord
	obj = spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),GridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift)
	return obj

def get_axes(vecObj):
	"""Function to return Axes from vector"""
	return vecObj.getHyper().axes

def chunkData(dataVecLocal,dataSpaceRemote):
	"""Function to chunk and spread the data vector across dask workers"""
	dask_client = dataSpaceRemote.dask_client #Getting Dask client
	client = dask_client.getClient()
	wrkIds = dask_client.getWorkerIds()
	dataAxes = client.gather(client.map(get_axes,dataSpaceRemote.vecDask,pure=False)) #Getting hypercubes of remote vector chunks
	List_Exps = [axes[-1].n for axes in dataAxes]
	dataNd = dataVecLocal.getNdArray()
	if(np.sum(List_Exps) != dataNd.shape[0]):
		raise ValueError("Number of shot within provide data vector (%s) not consistent with total number of shots from nShot parameter (%s)"%(dataNd.shape[0],np.sum(List_Exps)))
	#Pointer-wise chunking
	dataArrays = []
	firstShot = 0
	for nExp in List_Exps:
		dataArrays.append(dataNd[firstShot:firstShot+nExp,:,:,:])
		firstShot += nExp
	#Copying the data to remove vector
	dataVecRemote = dataSpaceRemote.clone()
	for idx,wrkId in enumerate(wrkIds):
		arrD = client.scatter(dataArrays[idx],workers=[wrkId])
		daskD.wait(arrD)
		daskD.wait(client.submit(pyDaskVector.copy_from_NdArray,dataVecRemote.vecDask[idx],arrD,workers=[wrkId],pure=False))
	# daskD.wait(client.map(pyDaskVector.copy_from_NdArray,dataVecRemote.vecDask,dataArrays,pure=False))
	return dataVecRemote

############################ Acquisition geometry ##############################

# Build sources geometry
def buildSourceGeometry(parObject,elasticParam):


	#Dipole parameters
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getFloat("zDipoleShift",0.0)
	xDipoleShift = parObject.getFloat("xDipoleShift",0.0)
	nts = parObject.getInt("nts")
	nExp = parObject.getInt("nExp")

	# Horizontal axis
	nx=elasticParam.getHyper().axes[1].n
	dx=elasticParam.getHyper().axes[1].d
	ox=elasticParam.getHyper().axes[1].o

	# vertical axis
	nz=elasticParam.getHyper().axes[0].n
	dz=elasticParam.getHyper().axes[0].d
	oz=elasticParam.getHyper().axes[0].o

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

	sourceGeomFile = parObject.getString("sourceGeomFile","None")

	if sourceGeomFile != "None":

		# Read geometry file
		# 3 axes:
		# First (fastest) axis: experiment index
		# Second (slower) axis: simultaneous source points
		# Third (slowest) axis: spatial coordinates
		sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile,ndims=3).getNdArray()
		nExpFile = sourceGeomVectorNd.shape[2]
		nSimSou = sourceGeomVectorNd.shape[1]

		if nExp != nExpFile:
			raise ValueError("ERROR! nExp (%d) not consistent with number of shots provided within sourceGeomFile (%d)"%(nExp,nExpFile))

		sourceAxis=Hypercube.axis(n=parObject.getInt("nExp"),o=1.0,d=1.0)

		for iExp in range(nExp):
			zCoordFloat=SepVector.getSepVector(ns=[nSimSou],storage="dataFloat")
			xCoordFloat=SepVector.getSepVector(ns=[nSimSou],storage="dataFloat")

			#Setting z and x positions of the source for the given experiment
			zCoordFloat.getNdArray()[:] = sourceGeomVectorNd[2,:,iExp]
			xCoordFloat.getNdArray()[:] = sourceGeomVectorNd[0,:,iExp]

			sourcesVectorCenterGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),centerGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorXGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),zGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorXZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xzGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))

	else:
		# Sources geometry
		nzSource=1
		ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
		dzSource=1
		nxSource=1
		oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
		dxSource=1
		spacingShots=parObject.getInt("spacingShots")
		sourceAxis=Hypercube.axis(n=parObject.getInt("nExp"),o=ox+oxSource*dx,d=spacingShots*dx)

		for iExp in range(nExp):
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

			sourcesVectorCenterGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),centerGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorXGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),zGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
			sourcesVectorXZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xzGridHyper.getCpp(),nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift))

			oxSource=oxSource+spacingShots # Shift source

	return sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis

# Build sources geometry for Dask interface
def buildSourceGeometryDask(parObject,elasticParamHyper,client):

	#Dipole parameters
	dipole = parObject.getInt("dipole",0)
	List_Exps = parObject.getInts("nExp",0)
	zDipoleShift = parObject.getFloat("zDipoleShift",0.0)
	xDipoleShift = parObject.getFloat("xDipoleShift",0.0)
	nts = parObject.getInt("nts")

	#Checking if list of shots is consistent with number of workers
	nWrks = client.getNworkers()
	wrkIds = client.getWorkerIds()
	if len(List_Exps) != nWrks:
		raise ValueError("Number of workers (#nWrk=%s) not consistent with length of the provided list of experiments (nExp=%s)"%(nWrks,List_Exps))

	# Horizontal axis
	nx=elasticParamHyper.axes[1].n
	dx=elasticParamHyper.axes[1].d
	ox=elasticParamHyper.axes[1].o

	# vertical axis
	nz=elasticParamHyper.axes[0].n
	dz=elasticParamHyper.axes[0].d
	oz=elasticParamHyper.axes[0].o

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")

	#need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=elasticParamHyper.axes[0].n,o=elasticParamHyper.axes[0].o,d=elasticParamHyper.axes[0].d)
	zAxisShifted=Hypercube.axis(n=elasticParamHyper.axes[0].n,o=elasticParamHyper.axes[0].o-0.5*elasticParamHyper.axes[0].d,d=elasticParamHyper.axes[0].d)

	xAxis=Hypercube.axis(n=elasticParamHyper.axes[1].n,o=elasticParamHyper.axes[1].o,d=elasticParamHyper.axes[1].d)
	xAxisShifted=Hypercube.axis(n=elasticParamHyper.axes[1].n,o=elasticParamHyper.axes[1].o-0.5*elasticParamHyper.axes[1].d,d=elasticParamHyper.axes[1].d)

	paramAxis=Hypercube.axis(n=elasticParamHyper.axes[2].n,o=elasticParamHyper.axes[2].o,d=elasticParamHyper.axes[2].d)

	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

	#Dask-related lists
	sourcesVectorCenterGrid = [[] for ii in range(nWrks)]
	sourcesVectorXGrid = [[] for ii in range(nWrks)]
	sourcesVectorZGrid = [[] for ii in range(nWrks)]
	sourcesVectorXZGrid = [[] for ii in range(nWrks)]
	sourceAxis=[]

	centerGridAxes=[zAxis,xAxis,paramAxis]
	xGridAxes=[zAxis,xAxisShifted,paramAxis]
	zGridAxes=[zAxisShifted,xAxis,paramAxis]
	xzGridAxes=[zAxisShifted,xAxisShifted,paramAxis]

	oxExp = ox+oxSource*dx
	dxExp = spacingShots*dx
	for idx,nExp in enumerate(List_Exps):
		#Shot axis for given shots
		sourceAxis.append(Hypercube.axis(n=nExp,o=oxExp,d=dxExp))
		for ishot in range(nExp):
			# sources _zCoord and _xCoord
			zAxes=[Hypercube.axis(n=nzSource,o=0.0,d=1.0)]
			xAxes=[Hypercube.axis(n=nxSource,o=0.0,d=1.0)]
			#Setting z and x position of the source for the given experiment
			zCoord=oz+ozSource*dz
			xCoord=ox+oxSource*dx
			# Constructing injection/estraction operators
			sourcesVectorCenterGrid[idx].append(client.getClient().submit(call_spaceInterpGpu,zAxes,xAxes,zCoord,xCoord,centerGridAxes,nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[idx],pure=False))
			sourcesVectorXGrid[idx].append(client.getClient().submit(call_spaceInterpGpu,zAxes,xAxes,zCoord,xCoord,xGridAxes,nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[idx],pure=False))
			sourcesVectorZGrid[idx].append(client.getClient().submit(call_spaceInterpGpu,zAxes,xAxes,zCoord,xCoord,zGridAxes,nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[idx],pure=False))
			sourcesVectorXZGrid[idx].append(client.getClient().submit(call_spaceInterpGpu,zAxes,xAxes,zCoord,xCoord,xzGridAxes,nts,sourceInterpMethod,sourceInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[idx],pure=False))
			# Shift source
			oxSource=oxSource+spacingShots
		daskD.wait(sourcesVectorCenterGrid[idx]+sourcesVectorXGrid[idx]+sourcesVectorZGrid[idx]+sourcesVectorXZGrid[idx])
		#Adding shots offset to origin of shot axis
		oxExp += (nExp-1)*dxExp

	return sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis

# Build receivers geometry
def buildReceiversGeometry(parObject,elasticParam):

	#Dipole parameters
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getFloat("zDipoleShift",0.0)
	xDipoleShift = parObject.getFloat("xDipoleShift",0.0)
	nts = parObject.getInt("nts")

	# Horizontal axis
	nx=elasticParam.getHyper().axes[1].n
	dx=elasticParam.getHyper().axes[1].d
	ox=elasticParam.getHyper().axes[1].o

	# Vertical axis
	nz=elasticParam.getHyper().axes[0].n
	dz=elasticParam.getHyper().axes[0].d
	oz=elasticParam.getHyper().axes[0].o

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

	nRecGeom=1; # Constant receivers' geometry

	# receiver _zCoord and _xCoord
	recGeomFile = parObject.getString("recGeomFile","none")
	if(recGeomFile != "none"):
		# Read geometry file
		# 3 axes:
		# First (fastest) axis: experiment index [for now fixed for every source]
		# Second (slower) axis: receiver points
		# Third (slowest) axis: spatial coordinates [x,y,z]
		recGeomVectorNd = genericIO.defaultIO.getVector(recGeomFile,ndims=3).getNdArray()
		xCoord = recGeomVectorNd[0,:,0]
		zCoord = recGeomVectorNd[2,:,0]
		# Check if number of receivers is consistent with parameter file
		nRec = parObject.getInt("nReceiver")
		if nRec != len(xCoord):
			raise ValueError("ERROR [buildReceiversGeometry]: Number of receiver coordinates (%s) not consistent with parameter file (nReceiver=%d)"%(len(xCoord),nRec))
		receiverAxis=Hypercube.axis(n=nRec,o=1.0,d=1.0)

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

		nzReceiver=1
		ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
		dzReceiver=0
		nxReceiver=parObject.getInt("nReceiver")
		oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
		dxReceiver=parObject.getInt("dReceiver")
		receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)

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
		recVectorCenterGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),centerGridHyper.getCpp(),nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
		recVectorXGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xGridHyper.getCpp(),nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
		recVectorZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),zGridHyper.getCpp(),nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift))
		recVectorXZGrid.append(spaceInterpGpu(zCoordFloat.getCpp(),xCoordFloat.getCpp(),xzGridHyper.getCpp(),nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift))

	return recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis

# Build receivers geometry
def buildReceiversGeometryDask(parObject,elasticParamHyper,client):

	#Dipole parameters
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getFloat("zDipoleShift",0.0)
	xDipoleShift = parObject.getFloat("xDipoleShift",0.0)
	nts = parObject.getInt("nts")

	#Getting number of workers
	nWrks = client.getNworkers()
	wrkIds = client.getWorkerIds()

	# Horizontal axis
	nx=elasticParamHyper.axes[1].n
	dx=elasticParamHyper.axes[1].d
	ox=elasticParamHyper.axes[1].o

	# Vertical axis
	nz=elasticParamHyper.axes[0].n
	dz=elasticParamHyper.axes[0].d
	oz=elasticParamHyper.axes[0].o

	##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=elasticParamHyper.axes[0].n,o=elasticParamHyper.axes[0].o,d=elasticParamHyper.axes[0].d)
	zAxisShifted=Hypercube.axis(n=elasticParamHyper.axes[0].n,o=elasticParamHyper.axes[0].o-0.5*elasticParamHyper.axes[0].d,d=elasticParamHyper.axes[0].d)

	xAxis=Hypercube.axis(n=elasticParamHyper.axes[1].n,o=elasticParamHyper.axes[1].o,d=elasticParamHyper.axes[1].d)
	xAxisShifted=Hypercube.axis(n=elasticParamHyper.axes[1].n,o=elasticParamHyper.axes[1].o-0.5*elasticParamHyper.axes[1].d,d=elasticParamHyper.axes[1].d)

	paramAxis=Hypercube.axis(n=elasticParamHyper.axes[2].n,o=elasticParamHyper.axes[2].o,d=elasticParamHyper.axes[2].d)

	#check which source injection interp method
	recInterpMethod = parObject.getString("recInterpMethod","linear")
	recInterpNumFilters = parObject.getInt("recInterpNumFilters",4)

	nRecGeom=1; # Constant receivers' geometry

	# receiver _zCoord and _xCoord
	recGeomFile = parObject.getString("recGeomFile","none")
	if(recGeomFile != "none"):
		# Read geometry file
		# 3 axes:
		# First (fastest) axis: experiment index [for now fixed for every source]
		# Second (slower) axis: receiver points
		# Third (slowest) axis: spatial coordinates [x,y,z]
		recGeomVectorNd = genericIO.defaultIO.getVector(recGeomFile,ndims=3).getNdArray()
		xCoord = recGeomVectorNd[0,:,0]
		zCoord = recGeomVectorNd[2,:,0]

		# Check if number of receivers is consistent with parameter file
		nRec = parObject.getInt("nReceiver")
		if nRec != len(xCoord):
			raise ValueError("ERROR [buildReceiversGeometry]: Number of receiver coordinates (%s) not consistent with parameter file (nReceiver=%d)"%(len(xCoord),nRec))
		receiverAxis=Hypercube.axis(n=nRec,o=1.0,d=1.0)

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

		nzReceiver=1
		ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat",4)
		dzReceiver=0
		nxReceiver=parObject.getInt("nReceiver")
		oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat",4)
		dxReceiver=parObject.getInt("dReceiver")

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

	centerGridAxes=[zAxis,xAxis,paramAxis]
	xGridAxes=[zAxis,xAxisShifted,paramAxis]
	zGridAxes=[zAxisShifted,xAxis,paramAxis]
	xzGridAxes=[zAxisShifted,xAxisShifted,paramAxis]

	#need a receiver vector for centerGrid, x shifted, z shifted, and xz shifted grid
	recVectorCenterGrid=[[] for ii in range(nWrks)]
	recVectorXGrid=[[] for ii in range(nWrks)]
	recVectorZGrid=[[] for ii in range(nWrks)]
	recVectorXZGrid=[[] for ii in range(nWrks)]

	receiverAxis=[Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)]*nWrks
	for iwrk in range(nWrks):
		recVectorCenterGrid[iwrk].append(client.getClient().submit(call_spaceInterpGpu, [recAxisVertical],[recAxisHorizontal],zCoordFloatNd,xCoordFloatNd,centerGridAxes,nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[iwrk],pure=False))
		recVectorXGrid[iwrk].append(client.getClient().submit(call_spaceInterpGpu,[recAxisVertical],[recAxisHorizontal],zCoordFloatNd,xCoordFloatNd,xGridAxes,nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[iwrk],pure=False))
		recVectorZGrid[iwrk].append(client.getClient().submit(call_spaceInterpGpu,[recAxisVertical],[recAxisHorizontal],zCoordFloatNd,xCoordFloatNd,zGridAxes,nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[iwrk],pure=False))
		recVectorXZGrid[iwrk].append(client.getClient().submit(call_spaceInterpGpu,[recAxisVertical],[recAxisHorizontal],zCoordFloatNd,xCoordFloatNd,xzGridAxes,nts,recInterpMethod,recInterpNumFilters,dipole,zDipoleShift,xDipoleShift,workers=wrkIds[iwrk],pure=False))

	return recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis

############################### Nonlinear ######################################
def nonlinearOpInitFloat(args,client=None):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	dummyAxis=Hypercube.axis(n=1)
	wavefieldAxis=Hypercube.axis(n=5)
	sourceGeomFile = parObject.getString("sourceGeomFile","None")

	# Allocate model
	if sourceGeomFile != "None":
		sourceGeomVector = genericIO.defaultIO.getVector(sourceGeomFile,ndims=3)
		sourceSimAxis = sourceGeomVector.getHyper().getAxis(2)
		modelHyper=Hypercube.hypercube(axes=[timeAxis,sourceSimAxis,wavefieldAxis,dummyAxis])
	else:
		modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,wavefieldAxis,dummyAxis])

	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	#Local vector copy useful for dask interface
	modelFloatLocal = modelFloat

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

	#Setting variables if Dask is employed
	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()
		#Spreading domain vector (i.e., wavelet)
		modelFloat = pyDaskVector.DaskVector(client,vectors=[modelFloat]*nWrks)

		#Spreading velocity model to workers
		elasticParamHyper = elasticParamFloat.getHyper()
		elasticParamFloat = pyDaskVector.DaskVector(client,vectors=[elasticParamFloat]*nWrks).vecDask

		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometryDask(parObject,elasticParamHyper,client)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometryDask(parObject,elasticParamHyper,client)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],wavefieldAxis,sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:
		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

		# Allocate data
		dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
		dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return  modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,modelFloatLocal

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
		self.pyOp = pyElastic_iso_float_nl.nonlinearPropElasticShotsGpu(elasticParam,paramP.param,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def __str__(self):
		"""Name of the operator"""
		return " NLOper "

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

def nonlinearFwiOpInitFloat(args,client=None):
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
	elasticParamFloatConv = elasticParamFloat
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv.ElasticConv(elasticParamFloat,mod_par)
		elasticParamFloatConv = elasticParamFloat.clone()
		convOp.forward(False,elasticParamFloat,elasticParamFloatConv)

	elasticParamFloatLocal=elasticParamFloat

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

	# Read sources signals
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		raise IOError("**** ERROR: User did not provide seismic sources file ****")
	sourcesTemp=genericIO.defaultIO.getVector(sourcesFile,ndims=3)
	sourcesFMat=sourceFloat.getNdArray()
	sourcesTMat=sourcesTemp.getNdArray()
	sourcesFMat[0,:,0,:]=sourcesTMat
	del sourcesTemp

	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading source wavelet
		sourceFloat = pyDaskVector.DaskVector(client,vectors=[sourceFloat]*nWrks)

		#Spreading velocity model to workers
		elasticParamHyper = elasticParamFloat.getHyper()
		elasticParamFloat = pyDaskVector.DaskVector(client,vectors=[elasticParamFloat]*nWrks)
		elasticParamFloatConv = pyDaskVector.DaskVector(client,vectors=[elasticParamFloatConv]*nWrks)

		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometryDask(parObject,elasticParamHyper,client)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometryDask(parObject,elasticParamHyper,client)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],wavefieldAxis,sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)
	else:
		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

		# Allocate data
		dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
		dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return elasticParamFloat,elasticParamFloatConv,dataFloat,sourceFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,elasticParamFloatLocal

class nonlinearFwiPropElasticShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,sources,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sources)):
			sources = sources.getCpp()
			self.sources = sources.clone()
		self.pyOp = pyElastic_iso_float_nl.nonlinearPropElasticShotsGpu(domain,paramP.param,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def __str__(self):
		"""Name of the operator"""
		return " NLOper "

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
def BornOpInitFloat(args,client=None):
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
	modelFloatLocal=modelFloat

	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading source wavelet
		sourcesSignalsFloat = pyDaskVector.DaskVector(client,vectors=[sourcesSignalsFloat]*nWrks).vecDask
		sourcesSignalsVector = client.getClient().map((lambda x: [x]),sourcesSignalsFloat,pure=False)
		daskD.wait(sourcesSignalsVector)

		#Spreading velocity model to workers
		elasticParamHyper = elasticParamFloat.getHyper()
		elasticParamFloatD = pyDaskVector.DaskVector(client,vectors=[elasticParamFloat]*nWrks)
		elasticParamFloat = elasticParamFloatD.vecDask

		#Allocate model
		modelFloat = elasticParamFloatD.clone()
		modelFloat.zero()

		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometryDask(parObject,elasticParamHyper,client)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometryDask(parObject,elasticParamHyper,client)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],wavefieldAxis,sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)
	else:
		# Build sources/receivers geometry
		sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourceAxis=buildSourceGeometry(parObject,elasticParamFloat)
		recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,receiverAxis=buildReceiversGeometry(parObject,elasticParamFloat)

		# Allocate data
		dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
		dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Outputs
	return modelFloat,dataFloat,elasticParamFloat,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,recVectorCenterGrid,recVectorXGrid,recVectorZGrid,recVectorXZGrid,modelFloatLocal

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
		self.pyOp = pyElastic_iso_float_born.BornElasticShotsGpu(elasticParam,paramP.param,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorZGrid,sourcesVectorXZGrid,receiversVectorCenterGrid,receiversVectorXGrid,receiversVectorZGrid,receiversVectorXZGrid)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornOp "

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
