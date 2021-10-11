#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pySpaceInterpMultiFloat
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys


def space_interp_multi_init_source(args):
  """Function to correctly initialize space interp for multiple component wflds
	   The function will return the necessary variables for operator construction
	"""
  # IO object
  parObject = genericIO.io(params=sys.argv)

  # elatic params
  elasticParamFile = parObject.getString("elasticParam", "noElasticParamFile")
  if (elasticParamFile == "noElasticParamFile"):
    print("**** ERROR: User did not provide elastic parameter file ****\n")
    sys.exit()
  elasticParam = genericIO.defaultIO.getVector(elasticParamFile)
  # elasticParam=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
  # elasticParamDoubleNp=elasticParam.getNdArray()

  # Horizontal axis
  nx = elasticParam.getHyper().axes[1].n
  dx = elasticParam.getHyper().axes[1].d
  ox = elasticParam.getHyper().axes[1].o

  # vertical axis
  nz = elasticParam.getHyper().axes[0].n
  dz = elasticParam.getHyper().axes[0].d
  oz = elasticParam.getHyper().axes[0].o

  # Sources geometry
  nzSource = 1
  ozSource = parObject.getInt("zSource") - 1 + parObject.getInt(
      "zPadMinus", 0) + parObject.getInt("fat")
  dzSource = 1
  nxSource = 1
  oxSource = parObject.getInt("xSource") - 1 + parObject.getInt(
      "xPadMinus", 0) + parObject.getInt("fat")
  dxSource = 1
  spacingShots = parObject.getInt("spacingShots")
  nExp = parObject.getInt("nExp")
  sourceAxis = Hypercube.axis(n=nExp, o=0, d=1)

  ##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
  zAxis = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                         o=elasticParam.getHyper().axes[0].o,
                         d=elasticParam.getHyper().axes[0].d)
  xAxis = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                         o=elasticParam.getHyper().axes[1].o,
                         d=elasticParam.getHyper().axes[1].d)
  paramAxis = Hypercube.axis(n=elasticParam.getHyper().axes[2].n,
                             o=elasticParam.getHyper().axes[2].o,
                             d=elasticParam.getHyper().axes[2].d)

  centerGridHyper = Hypercube.hypercube(axes=[zAxis, xAxis, paramAxis])

  #check which source injection interp method
  # sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
  # sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

  # sources _zCoord and _xCoord
  zCoordHyper = Hypercube.hypercube(axes=[sourceAxis])
  zCoordFloat = SepVector.getSepVector(zCoordHyper, storage="dataFloat")
  xCoordHyper = Hypercube.hypercube(axes=[sourceAxis])
  xCoordFloat = SepVector.getSepVector(xCoordHyper, storage="dataFloat")

  xCoordDMat = xCoordFloat.getNdArray()
  zCoordDMat = zCoordFloat.getNdArray()

  for ishot in range(nExp):
    #Setting z and x position of the source for the given experiment
    zCoordDMat[ishot] = oz + ozSource * dz
    xCoordDMat[ishot] = ox + oxSource * dx
    oxSource = oxSource + spacingShots  # Shift source

  return zCoordFloat, xCoordFloat, centerGridHyper


def space_interp_multi_init_rec(args):
  """Function to correctly initialize space interp for multiple component wflds
	   The function will return the necessary variables for operator construction
	"""
  # IO object
  parObject = genericIO.io(params=sys.argv)

  # elatic params
  elasticParamFile = parObject.getString("elasticParam", "noElasticParamFile")
  if (elasticParamFile == "noElasticParamFile"):
    print("**** ERROR: User did not provide elastic parameter file ****\n")
    sys.exit()
  elasticParam = genericIO.defaultIO.getVector(elasticParamFile)
  # elasticParam=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataFloat")
  # elasticParamDoubleNp=elasticParam.getNdArray()

  # Horizontal axis
  nx = elasticParam.getHyper().axes[1].n
  dx = elasticParam.getHyper().axes[1].d
  ox = elasticParam.getHyper().axes[1].o

  # vertical axis
  nz = elasticParam.getHyper().axes[0].n
  dz = elasticParam.getHyper().axes[0].d
  oz = elasticParam.getHyper().axes[0].o

  # rec geometry
  nzReceiver = 1
  ozReceiver = parObject.getInt("depthReceiver") - 1 + parObject.getInt(
      "zPadMinus", 0) + parObject.getInt("fat")
  dzReceiver = 0
  nxReceiver = parObject.getInt("nReceiver")
  oxReceiver = parObject.getInt("oReceiver") - 1 + parObject.getInt(
      "xPadMinus", 0) + parObject.getInt("fat")
  dxReceiver = parObject.getInt("dReceiver")
  receiverAxis = Hypercube.axis(n=nxReceiver,
                                o=ox + oxReceiver * dx,
                                d=dxReceiver * dx)
  nRecGeom = 1
  # Constant receivers' geometry

  ##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
  zAxis = Hypercube.axis(n=elasticParam.getHyper().axes[0].n,
                         o=elasticParam.getHyper().axes[0].o,
                         d=elasticParam.getHyper().axes[0].d)
  xAxis = Hypercube.axis(n=elasticParam.getHyper().axes[1].n,
                         o=elasticParam.getHyper().axes[1].o,
                         d=elasticParam.getHyper().axes[1].d)
  paramAxis = Hypercube.axis(n=elasticParam.getHyper().axes[2].n,
                             o=elasticParam.getHyper().axes[2].o,
                             d=elasticParam.getHyper().axes[2].d)

  centerGridHyper = Hypercube.hypercube(axes=[zAxis, xAxis, paramAxis])

  #check which source injection interp method
  # sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
  # sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

  # sources _zCoord and _xCoord
  zCoordHyper = Hypercube.hypercube(axes=[receiverAxis])
  zCoordFloat = SepVector.getSepVector(zCoordHyper, storage="dataFloat")
  xCoordHyper = Hypercube.hypercube(axes=[receiverAxis])
  xCoordFloat = SepVector.getSepVector(xCoordHyper, storage="dataFloat")

  xCoordDMat = xCoordFloat.getNdArray()
  zCoordDMat = zCoordFloat.getNdArray()

  for irec in range(nxReceiver):
    #Setting z and x position of the source for the given experiment
    zCoordDMat[irec] = oz + ozReceiver * dz + dzReceiver * dz * irec
    xCoordDMat[irec] = ox + oxReceiver * dx + dxReceiver * dx * irec

  return zCoordFloat, xCoordFloat, centerGridHyper


class space_interp_multi(Op.Operator):
  """Wrapper encapsulating PYBIND11 module"""

  def __init__(self, zCoord, xCoord, elasticParamHypercube, nt, interpMethod,
               nFilt):
    #Checking if getCpp is present
    # self.setDomainRange(domain,range)
    # if("getCpp" in dir(domain)):
    # 	domain = domain.getCpp()
    # if("getCpp" in dir(range)):
    # 	range = range.getCpp()
    self.pyOp = pySpaceInterpMultiFloat.spaceInterpMulti(
        zCoord.getCpp(), xCoord.getCpp(), elasticParamHypercube.getCpp(), nt,
        interpMethod, nFilt)
    return

  def forward(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pySpaceInterpMultiFloat.ostream_redirect():
      self.pyOp.forward(add, model, data)
    return

  def adjoint(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pySpaceInterpMultiFloat.ostream_redirect():
      self.pyOp.adjoint(add, model, data)
    return

  def dotTestCpp(self, verb=False, maxError=.00001):
    """Method to call the Cpp class dot-product test"""
    with pySpaceInterpMultiFloat.ostream_redirect():
      result = self.pyOp.dotTest(verb, maxError)
    return result

  def getNDeviceIrreg(self):
    with pySpaceInterpMultiFloat.ostream_redirect():
      result = self.pyOp.getNDeviceIrreg()
    return result

  def getNDeviceReg(self):
    with pySpaceInterpMultiFloat.ostream_redirect():
      result = self.pyOp.getNDeviceReg()
    return result

  def getRegPosUniqueVector(self):
    with pySpaceInterpMultiFloat.ostream_redirect():
      result = self.pyOp.getRegPosUniqueVector()
    return result
