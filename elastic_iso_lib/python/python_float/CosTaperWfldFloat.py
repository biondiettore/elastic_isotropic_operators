#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyCosTaperWfld
import pyOperator as Op


class cos_taper_wfld_float(Op.Operator):
  """Wrapper encapsulating PYBIND11 module"""

  def __init__(self, domain, range, bz, bx, width, alpha, beta):
    #Checking if getCpp is present
    self.setDomainRange(domain, range)
    if ("getCpp" in dir(domain)):
      domain = domain.getCpp()
    if ("getCpp" in dir(range)):
      range = range.getCpp()
    self.pyOp = pyCosTaperWfld.cosTaperWfld(domain, range, bz, bx, width, alpha,
                                            beta)
    return

  def forward(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyCosTaperWfld.ostream_redirect():
      self.pyOp.forward(add, model, data)
    return

  def adjoint(self, add, model, data):
    #Checking if getCpp is present
    if ("getCpp" in dir(model)):
      model = model.getCpp()
    if ("getCpp" in dir(data)):
      data = data.getCpp()
    with pyCosTaperWfld.ostream_redirect():
      self.pyOp.adjoint(add, model, data)
    return

  def dotTestCpp(self, verb=False, maxError=.00001):
    """Method to call the Cpp class dot-product test"""
    with pyCosTaperWfld.ostream_redirect():
      result = self.pyOp.dotTest(verb, maxError)
    return result
