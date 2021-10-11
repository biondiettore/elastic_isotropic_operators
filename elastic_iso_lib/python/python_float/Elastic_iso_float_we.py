#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyElastic_iso_float_we
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys


############################# Wave Equation ####################################
def waveEquationOpInitFloat(args):
  """Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
  # IO object
  parObject = genericIO.io(params=sys.argv)

  # elatic params
  elasticParam = parObject.getString("elasticParam", "noElasticParamFile")
  if (elasticParam == "noElasticParamFile"):
    print("**** ERROR: User did not provide elastic parameter file ****\n")
    sys.exit()
  elasticParamFloat = genericIO.defaultIO.getVector(elasticParam)
  # elasticParamDouble=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
  # elasticParamDoubleNp=elasticParamDouble.getNdArray()
  # elasticParamFloatNp=elasticParamFloat.getNdArray()
  # elasticParamDoubleNp[:]=elasticParamFloatNp

  # Time Axis
  nts = parObject.getInt("nts", -1)
  ots = parObject.getFloat("ots", 0.0)
  dts = parObject.getFloat("dts", -1.0)
  timeAxis = Hypercube.axis(n=nts, o=ots, d=dts)

  # z Axis
  nz = parObject.getInt("nz", -1)
  oz = parObject.getFloat("oz", -1.0)
  dz = parObject.getFloat("dz", -1.0)
  zAxis = Hypercube.axis(n=nz, o=oz, d=dz)

  # x axis
  nx = parObject.getInt("nx", -1)
  ox = parObject.getFloat("ox", -1.0)
  dx = parObject.getFloat("dx", -1.0)
  xAxis = Hypercube.axis(n=nx, o=ox, d=dx)

  #wavefield axis
  wavefieldAxis = Hypercube.axis(n=5)

  # Allocate model
  modelHyper = Hypercube.hypercube(axes=[zAxis, xAxis, wavefieldAxis, timeAxis])
  modelFloat = SepVector.getSepVector(modelHyper, storage="dataFloat")

  # Allocate data
  dataHyper = Hypercube.hypercube(axes=[zAxis, xAxis, wavefieldAxis, timeAxis])
  dataFloat = SepVector.getSepVector(dataHyper, storage="dataFloat")

  # Outputs
  return modelFloat, dataFloat, elasticParamFloat, parObject


class waveEquationElasticGpu(Op.Operator):
  """Wrapper encapsulating PYBIND11 module for elastic wave equation"""

  def __init__(self, domain, range, elasticParam, paramP):
    #Domain = source wavelet
    #Range = recorded data space
    self.setDomainRange(domain, range)
    #Checking if getCpp is present
    if ("getCpp" in dir(elasticParam)):
      elasticParam = elasticParam.getCpp()
    if ("getCpp" in dir(paramP)):
      paramP = paramP.getCpp()
    if ("getCpp" in dir(domain)):
      domain = domain.getCpp()
    if ("getCpp" in dir(range)):
      range = range.getCpp()
    self.pyOp = pyElastic_iso_float_we.waveEquationElasticGpu(
        domain, range, elasticParam, paramP)
    return

  def forward(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_float_we.ostream_redirect():
      self.pyOp.forward(add, model, data)
    return

  def adjoint(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyElastic_iso_float_we.ostream_redirect():
      self.pyOp.adjoint(add, model, data)
    return

  def dotTestCpp(self, verb=False, maxError=.00001):
    """Method to call the Cpp class dot-product test"""
    with pyElastic_iso_float_we.ostream_redirect():
      result = self.pyOp.dotTest(verb, maxError)
    return result
