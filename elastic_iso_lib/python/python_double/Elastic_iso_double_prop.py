#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyElastic_iso_double_nl
import pyElastic_iso_double_born
import pyOperator as Op
import elasticParamConvertModule as ElaConv
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

from pyElastic_iso_double_nl import spaceInterpGpu


############################ Acquisition geometry ##############################
# Reads source or receiver x,y,z locations from parFile
def parsePosParFile(PosParFile):
  nDevices = None
  devCoords = []
  with open(PosParFile, "r") as fid:
    for line in fid:
      if ("#" not in line):
        lineCur = line.split()
        if (len(lineCur) == 1):
          nDevices = float(lineCur[0])
        elif (len(lineCur) == 3):
          lineCur[0] = float(lineCur[0])
          lineCur[1] = float(lineCur[1])
          lineCur[2] = float(lineCur[2])
          devCoords.append(lineCur)
        else:
          raise ValueError("Error: Incorrectly formatted line, %s, in %s" %
                           (lineCur, PosParFile))
  if (nDevices != None):
    if (len(devCoords) != nDevices):
      raise ValueError(
          "ERROR: number of devices in parfile (%d) not the same as specified nDevices (%d)"
          % (len(a), nDevices))
  devCoordsNdArray = np.asarray(devCoords)
  return devCoordsNdArray[:, 0], devCoordsNdArray[:, 2]


# Build sources geometry
def buildSourceGeometry(parObject, elasticParam):

  #Dipole parameters
  dipole = parObject.getInt("dipole", 0)
  zDipoleShift = parObject.getFloat("zDipoleShift", 0.0)
  xDipoleShift = parObject.getFloat("xDipoleShift", 0.0)
  nts = parObject.getInt("nts")
  nExp = parObject.getInt("nExp")

  # Horizontal axis
  nx = elasticParam.getHyper().axes[1].n
  dx = elasticParam.getHyper().axes[1].d
  ox = elasticParam.getHyper().axes[1].o

  # vertical axis
  nz = elasticParam.getHyper().axes[0].n
  dz = elasticParam.getHyper().axes[0].d
  oz = elasticParam.getHyper().axes[0].o

  ##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
  zAxis = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                         o=elasticParam.getHyper().axes[0].o,
                         d=elasticParam.getHyper().axes[0].d)
  zAxisShifted = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                                o=elasticParam.getHyper().axes[0].o -
                                0.5 * elasticParam.getHyper().axes[0].d,
                                d=elasticParam.getHyper().axes[0].d)

  xAxis = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                         o=elasticParam.getHyper().axes[1].o,
                         d=elasticParam.getHyper().axes[1].d)
  xAxisShifted = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                                o=elasticParam.getHyper().axes[1].o -
                                0.5 * elasticParam.getHyper().axes[1].d,
                                d=elasticParam.getHyper().axes[1].d)

  paramAxis = Hypercube.axis(n=elasticParam.getHyper().axes[2].n,
                             o=elasticParam.getHyper().axes[2].o,
                             d=elasticParam.getHyper().axes[2].d)

  centerGridHyper = Hypercube.hypercube(axes=[zAxis, xAxis, paramAxis])
  xGridHyper = Hypercube.hypercube(axes=[zAxis, xAxisShifted, paramAxis])
  zGridHyper = Hypercube.hypercube(axes=[zAxisShifted, xAxis, paramAxis])
  xzGridHyper = Hypercube.hypercube(
      axes=[zAxisShifted, xAxisShifted, paramAxis])

  #need a sources vector for centerGrid, x shifted, z shifted, and xz shifted grid
  sourcesVectorCenterGrid = []
  sourcesVectorXGrid = []
  sourcesVectorZGrid = []
  sourcesVectorXZGrid = []

  #check which source injection interp method
  sourceInterpMethod = parObject.getString("sourceInterpMethod", "linear")
  sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters", 4)

  sourceGeomFile = parObject.getString("sourceGeomFile", "None")

  if sourceGeomFile != "None":

    # Read geometry file
    # 3 axes:
    # First (fastest) axis: experiment index
    # Second (slower) axis: simultaneous source points
    # Third (slowest) axis: spatial coordinates
    sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile,
                                                       ndims=3).getNdArray()
    nExpFile = sourceGeomVectorNd.shape[2]
    nSimSou = sourceGeomVectorNd.shape[1]

    if nExp != nExpFile:
      raise ValueError(
          "ERROR! nExp (%d) not consistent with number of shots provided within sourceGeomFile (%d)"
          % (nExp, nExpFile))

    sourceAxis = Hypercube.axis(n=parObject.getInt("nExp"), o=1.0, d=1.0)

    for iExp in range(nExp):
      zCoordFloat = SepVector.getSepVector(ns=[nSimSou], storage="dataDouble")
      xCoordFloat = SepVector.getSepVector(ns=[nSimSou], storage="dataDouble")

      #Setting z and x positions of the source for the given experiment
      zCoordFloat.getNdArray()[:] = sourceGeomVectorNd[2, :, iExp]
      xCoordFloat.getNdArray()[:] = sourceGeomVectorNd[0, :, iExp]

      sourcesVectorCenterGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         centerGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorXGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         xGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorZGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         zGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorXZGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         xzGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))

  else:
    # Sources geometry
    nzSource = 1
    ozSource = parObject.getInt("zSource") - 1 + parObject.getInt(
        "zPadMinus") + parObject.getInt("fat", 4)
    dzSource = 1
    nxSource = 1
    oxSource = parObject.getInt("xSource") - 1 + parObject.getInt(
        "xPadMinus") + parObject.getInt("fat", 4)
    dxSource = 1
    spacingShots = parObject.getInt("spacingShots")
    sourceAxis = Hypercube.axis(n=parObject.getInt("nExp"),
                                o=ox + oxSource * dx,
                                d=spacingShots * dx)

    for iExp in range(nExp):
      # sources _zCoord and _xCoord
      sourceAxisVertical = Hypercube.axis(n=nzSource, o=0.0, d=1.0)
      zCoordHyper = Hypercube.hypercube(axes=[sourceAxisVertical])
      zCoordFloat = SepVector.getSepVector(zCoordHyper, storage="dataDouble")
      sourceAxisHorizontal = Hypercube.axis(n=nxSource, o=0.0, d=1.0)
      xCoordHyper = Hypercube.hypercube(axes=[sourceAxisHorizontal])
      xCoordFloat = SepVector.getSepVector(xCoordHyper, storage="dataDouble")

      #Setting z and x position of the source for the given experiment
      zCoordFloat.set(oz + ozSource * dz)
      xCoordFloat.set(ox + oxSource * dx)

      sourcesVectorCenterGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         centerGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorXGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         xGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorZGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         zGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))
      sourcesVectorXZGrid.append(
          spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                         xzGridHyper.getCpp(), nts, sourceInterpMethod,
                         sourceInterpNumFilters, dipole, zDipoleShift,
                         xDipoleShift))

      oxSource = oxSource + spacingShots  # Shift source

  return sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, sourceAxis


# Build receivers geometry
def buildReceiversGeometry(parObject, elasticParam):

  #Dipole parameters
  dipole = parObject.getInt("dipole", 0)
  zDipoleShift = parObject.getFloat("zDipoleShift", 0.0)
  xDipoleShift = parObject.getFloat("xDipoleShift", 0.0)
  nts = parObject.getInt("nts")

  # Horizontal axis
  nx = elasticParam.getHyper().axes[1].n
  dx = elasticParam.getHyper().axes[1].d
  ox = elasticParam.getHyper().axes[1].o

  # Vertical axis
  nz = elasticParam.getHyper().axes[0].n
  dz = elasticParam.getHyper().axes[0].d
  oz = elasticParam.getHyper().axes[0].o

  ##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
  zAxis = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                         o=elasticParam.getHyper().axes[0].o,
                         d=elasticParam.getHyper().axes[0].d)
  zAxisShifted = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                                o=elasticParam.getHyper().axes[0].o -
                                0.5 * elasticParam.getHyper().axes[0].d,
                                d=elasticParam.getHyper().axes[0].d)

  xAxis = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                         o=elasticParam.getHyper().axes[1].o,
                         d=elasticParam.getHyper().axes[1].d)
  xAxisShifted = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                                o=elasticParam.getHyper().axes[1].o -
                                0.5 * elasticParam.getHyper().axes[1].d,
                                d=elasticParam.getHyper().axes[1].d)

  paramAxis = Hypercube.axis(n=elasticParam.getHyper().axes[2].n,
                             o=elasticParam.getHyper().axes[2].o,
                             d=elasticParam.getHyper().axes[2].d)

  centerGridHyper = Hypercube.hypercube(axes=[zAxis, xAxis, paramAxis])
  xGridHyper = Hypercube.hypercube(axes=[zAxis, xAxisShifted, paramAxis])
  zGridHyper = Hypercube.hypercube(axes=[zAxisShifted, xAxis, paramAxis])
  xzGridHyper = Hypercube.hypercube(
      axes=[zAxisShifted, xAxisShifted, paramAxis])

  #need a receiver vector for centerGrid, x shifted, z shifted, and xz shifted grid
  recVectorCenterGrid = []
  recVectorXGrid = []
  recVectorZGrid = []
  recVectorXZGrid = []

  #check which source injection interp method
  recInterpMethod = parObject.getString("recInterpMethod", "linear")
  recInterpNumFilters = parObject.getInt("recInterpNumFilters", 4)

  nRecGeom = 1
  # Constant receivers' geometry

  # receiver _zCoord and _xCoord
  recGeomFile = parObject.getString("recGeomFile", "none")
  if (recGeomFile != "none"):
    # Read geometry file
    # 3 axes:
    # First (fastest) axis: experiment index [for now fixed for every source]
    # Second (slower) axis: receiver points
    # Third (slowest) axis: spatial coordinates [x,y,z]
    recGeomVectorNd = genericIO.defaultIO.getVector(recGeomFile,
                                                    ndims=3).getNdArray()
    xCoord = recGeomVectorNd[0, :, 0]
    zCoord = recGeomVectorNd[2, :, 0]
    # Check if number of receivers is consistent with parameter file
    nRec = parObject.getInt("nReceiver")
    if nRec != len(xCoord):
      raise ValueError(
          "ERROR [buildReceiversGeometry]: Number of receiver coordinates (%s) not consistent with parameter file (nReceiver=%d)"
          % (len(xCoord), nRec))
    receiverAxis = Hypercube.axis(n=nRec, o=1.0, d=1.0)

    recAxisVertical = Hypercube.axis(n=len(zCoord), o=0.0, d=1.0)
    zCoordHyper = Hypercube.hypercube(axes=[recAxisVertical])
    zCoordFloat = SepVector.getSepVector(zCoordHyper, storage="dataDouble")
    zCoordFloatNd = zCoordFloat.getNdArray()
    zCoordFloatNd[:] = zCoord
    recAxisHorizontal = Hypercube.axis(n=len(xCoord), o=0.0, d=1.0)
    xCoordHyper = Hypercube.hypercube(axes=[recAxisHorizontal])
    xCoordFloat = SepVector.getSepVector(xCoordHyper, storage="dataDouble")
    xCoordFloatNd = xCoordFloat.getNdArray()
    xCoordFloatNd[:] = xCoord

  else:

    nzReceiver = 1
    ozReceiver = parObject.getInt("depthReceiver") - 1 + parObject.getInt(
        "zPadMinus") + parObject.getInt("fat", 4)
    dzReceiver = 0
    nxReceiver = parObject.getInt("nReceiver")
    oxReceiver = parObject.getInt("oReceiver") - 1 + parObject.getInt(
        "xPadMinus") + parObject.getInt("fat", 4)
    dxReceiver = parObject.getInt("dReceiver")
    receiverAxis = Hypercube.axis(n=nxReceiver,
                                  o=ox + oxReceiver * dx,
                                  d=dxReceiver * dx)

    recAxisVertical = Hypercube.axis(n=nxReceiver, o=0.0, d=1.0)
    zCoordHyper = Hypercube.hypercube(axes=[recAxisVertical])
    zCoordFloat = SepVector.getSepVector(zCoordHyper, storage="dataDouble")
    zCoordFloatNd = zCoordFloat.getNdArray()
    recAxisHorizontal = Hypercube.axis(n=nxReceiver, o=0.0, d=1.0)
    xCoordHyper = Hypercube.hypercube(axes=[recAxisHorizontal])
    xCoordFloat = SepVector.getSepVector(xCoordHyper, storage="dataDouble")
    xCoordFloatNd = xCoordFloat.getNdArray()
    for irec in range(nxReceiver):
      zCoordFloatNd[irec] = oz + ozReceiver * dz + dzReceiver * dz * irec
      xCoordFloatNd[irec] = ox + oxReceiver * dx + dxReceiver * dx * irec

  for iRec in range(nRecGeom):
    recVectorCenterGrid.append(
        spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                       centerGridHyper.getCpp(), nts, recInterpMethod,
                       recInterpNumFilters, dipole, zDipoleShift, xDipoleShift))
    recVectorXGrid.append(
        spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                       xGridHyper.getCpp(), nts, recInterpMethod,
                       recInterpNumFilters, dipole, zDipoleShift, xDipoleShift))
    recVectorZGrid.append(
        spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                       zGridHyper.getCpp(), nts, recInterpMethod,
                       recInterpNumFilters, dipole, zDipoleShift, xDipoleShift))
    recVectorXZGrid.append(
        spaceInterpGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(),
                       xzGridHyper.getCpp(), nts, recInterpMethod,
                       recInterpNumFilters, dipole, zDipoleShift, xDipoleShift))

  return recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, receiverAxis


############################### Nonlinear ######################################
def nonlinearOpInitDouble(args):
  """Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
  # IO objects
  parObject = genericIO.io(params=sys.argv)

  # elatic params
  elasticParam = parObject.getString("elasticParam", "noElasticParamFile")
  if (elasticParam == "noElasticParamFile"):
    print("**** ERROR: User did not provide elastic parameter file ****\n")
    sys.exit()
  elasticParamFloat = genericIO.defaultIO.getVector(elasticParam)
  elasticParamDouble = SepVector.getSepVector(elasticParamFloat.getHyper(),
                                              storage="dataDouble")

  #Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
  # 0 ==> correct parameterization
  # 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
  mod_par = parObject.getInt("mod_par", 0)
  if (mod_par != 0):
    convOp = ElaConv.ElasticConv(elasticParamFloat, mod_par)
    elasticParamFloatTemp = elasticParamFloat.clone()
    convOp.forward(False, elasticParamFloatTemp, elasticParamFloat)
    del elasticParamFloatTemp

  #Conversion to double precision
  elasticParamDoubleNp = elasticParamDouble.getNdArray()
  elasticParamFloatNp = elasticParamFloat.getNdArray()
  elasticParamDoubleNp[:] = elasticParamFloatNp

  # Build sources/receivers geometry
  sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, sourceAxis = buildSourceGeometry(
      parObject, elasticParamDouble)
  recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, receiverAxis = buildReceiversGeometry(
      parObject, elasticParamDouble)

  # Time Axis
  nts = parObject.getInt("nts", -1)
  ots = parObject.getFloat("ots", 0.0)
  dts = parObject.getFloat("dts", -1.0)
  timeAxis = Hypercube.axis(n=nts, o=ots, d=dts)
  # Allocate model
  dummyAxis = Hypercube.axis(n=1)
  wavefieldAxis = Hypercube.axis(n=5)
  sourceGeomFile = parObject.getString("sourceGeomFile", "None")

  # Allocate model
  if sourceGeomFile != "None":
    sourceGeomVector = genericIO.defaultIO.getVector(sourceGeomFile, ndims=3)
    sourceSimAxis = sourceGeomVector.getHyper().getAxis(2)
    modelHyper = Hypercube.hypercube(
        axes=[timeAxis, sourceSimAxis, wavefieldAxis, dummyAxis])
  else:
    modelHyper = Hypercube.hypercube(
        axes=[timeAxis, dummyAxis, wavefieldAxis, dummyAxis])

  # Allocate model
  modelDouble = SepVector.getSepVector(modelHyper, storage="dataDouble")

  # Allocate data
  dataHyper = Hypercube.hypercube(
      axes=[timeAxis, receiverAxis, wavefieldAxis, sourceAxis])
  dataDouble = SepVector.getSepVector(dataHyper, storage="dataDouble")

  # Outputs
  return modelDouble, dataDouble, elasticParamDouble, parObject, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid


class nonlinearPropElasticShotsGpu(Op.Operator):
  """Wrapper encapsulating PYBIND11 module for non-linear propagator"""

  def __init__(self, domain, range, elasticParam, paramP,
               sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
               sourcesVectorXZGrid, receiversVectorCenterGrid,
               receiversVectorXGrid, receiversVectorZGrid,
               receiversVectorXZGrid):
    #Domain = source wavelet
    #Range = recorded data space
    self.setDomainRange(domain, range)
    #Checking if getCpp is present
    if ("getCpp" in dir(elasticParam)):
      elasticParam = elasticParam.getCpp()
    if ("getCpp" in dir(paramP)):
      paramP = paramP.getCpp()
    self.pyOp = pyElastic_iso_double_nl.nonlinearPropElasticShotsGpu(
        elasticParam, paramP, sourcesVectorCenterGrid, sourcesVectorXGrid,
        sourcesVectorZGrid, sourcesVectorXZGrid, receiversVectorCenterGrid,
        receiversVectorXGrid, receiversVectorZGrid, receiversVectorXZGrid)
    return

  def forward(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.forward(add, model, data)
    return

  def forwardWavefield(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.forwardWavefield(add, model, data)
    return

  def adjoint(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.adjoint(add, model, data)
    return

  def adjointWavefield(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.adjointWavefield(add, model, data)
    return

  def getWavefield(self):
    wavefield = self.pyOp.getWavefield()
    return SepVector.doubleVector(fromCpp=wavefield)

  def setBackground(self, elasticParam):
    #Checking if getCpp is present
    if ("getCpp" in dir(elasticParam)):
      elasticParam = elasticParam.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.setBackground(elasticParam)
    return

  def dotTestCpp(self, verb=False, maxError=.00001):
    """Method to call the Cpp class dot-product test"""
    with pyElastic_iso_double_nl.ostream_redirect():
      result = self.pyOp.dotTest(verb, maxError)
    return result


################################### Born #######################################
def BornOpInitDouble(args):
  """Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
  # IO objects
  parObject = genericIO.io(params=sys.argv)

  # elatic params
  elasticParam = parObject.getString("elasticParam", "noElasticParamFile")
  if (elasticParam == "noElasticParamFile"):
    print("**** ERROR: User did not provide elastic parameter file ****\n")
    sys.exit()
  elasticParamFloat = genericIO.defaultIO.getVector(elasticParam)
  elasticParamDouble = SepVector.getSepVector(elasticParamFloat.getHyper(),
                                              storage="dataDouble")

  #Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
  # 0 ==> correct parameterization
  # 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
  mod_par = parObject.getInt("mod_par", 0)
  if (mod_par != 0):
    convOp = ElaConv.ElasticConv(elasticParamFloat, mod_par)
    elasticParamFloatTemp = elasticParamFloat.clone()
    convOp.forward(False, elasticParamFloatTemp, elasticParamFloat)
    del elasticParamFloatTemp

  #Conversion to double precision
  elasticParamDoubleNp = elasticParamDouble.getNdArray()
  elasticParamFloatNp = elasticParamFloat.getNdArray()
  elasticParamDoubleNp[:] = elasticParamFloatNp

  # Build sources/receivers geometry
  sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, sourceAxis = buildSourceGeometry(
      parObject, elasticParamDouble)
  recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid, receiverAxis = buildReceiversGeometry(
      parObject, elasticParamDouble)

  # Time Axis
  nts = parObject.getInt("nts", -1)
  ots = parObject.getFloat("ots", 0.0)
  dts = parObject.getFloat("dts", -1.0)
  timeAxis = Hypercube.axis(n=nts, o=ots, d=dts)
  wavefieldAxis = Hypercube.axis(n=5)

  # Read sources signals
  sourcesFile = parObject.getString("sources", "noSourcesFile")
  if (sourcesFile == "noSourcesFile"):
    raise IOError("**** ERROR: User did not provide seismic sources file ****")
  sourcesSignalsFloat = genericIO.defaultIO.getVector(sourcesFile, ndims=3)
  sourcesSignalsDouble = SepVector.getSepVector(sourcesSignalsFloat.getHyper(),
                                                storage="dataDouble")
  sourcesSignalsDoubleNp = sourcesSignalsDouble.getNdArray()
  sourcesSignalsFloatNp = sourcesSignalsFloat.getNdArray()
  sourcesSignalsDoubleNp[:] = sourcesSignalsFloatNp
  sourcesSignalsVector = []
  sourcesSignalsVector.append(
      sourcesSignalsDouble)  # Create a vector of double3DReg slices

  # Allocate model
  modelDouble = SepVector.getSepVector(elasticParamDouble.getHyper(),
                                       storage="dataDouble")

  # Allocate data
  dataHyper = Hypercube.hypercube(
      axes=[timeAxis, receiverAxis, wavefieldAxis, sourceAxis])
  dataDouble = SepVector.getSepVector(dataHyper, storage="dataDouble")

  # Outputs
  return modelDouble, dataDouble, elasticParamDouble, parObject, sourcesSignalsVector, sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid, recVectorCenterGrid, recVectorXGrid, recVectorZGrid, recVectorXZGrid


class BornElasticShotsGpu(Op.Operator):
  """Wrapper encapsulating PYBIND11 module for elastic Born propagator"""

  def __init__(self, domain, range, elasticParam, paramP, sourcesSignalsVector,
               sourcesVectorCenterGrid, sourcesVectorXGrid, sourcesVectorZGrid,
               sourcesVectorXZGrid, receiversVectorCenterGrid,
               receiversVectorXGrid, receiversVectorZGrid,
               receiversVectorXZGrid):
    #Domain = source wavelet
    #Range = recorded data space
    self.setDomainRange(domain, range)
    #Checking if getCpp is present
    if ("getCpp" in dir(elasticParam)):
      elasticParam = elasticParam.getCpp()
    if ("getCpp" in dir(paramP)):
      paramP = paramP.getCpp()
    for idx, sourceSignal in enumerate(sourcesSignalsVector):
      if ("getCpp" in dir(sourceSignal)):
        sourcesSignalsVector[idx] = sourceSignal.getCpp()
    self.pyOp = pyElastic_iso_double_born.BornElasticShotsGpu(
        elasticParam, paramP, sourcesSignalsVector, sourcesVectorCenterGrid,
        sourcesVectorXGrid, sourcesVectorZGrid, sourcesVectorXZGrid,
        receiversVectorCenterGrid, receiversVectorXGrid, receiversVectorZGrid,
        receiversVectorXZGrid)
    return

  def forward(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.forward(add, model, data)
    return

  def adjoint(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.adjoint(add, model, data)
    return

  def setBackground(self, elasticParam):
    #Checking if getCpp is present
    if ("getCpp" in dir(elasticParam)):
      elasticParam = elasticParam.getCpp()
    with pyElastic_iso_double_nl.ostream_redirect():
      self.pyOp.setBackground(elasticParam)
    return
