#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pySP_Float
import pyOperator as Op


class sp_op(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range,gridPointIndexUnique):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pySP_Float.SP(domain,range,gridPointIndexUnique)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySP_Float.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySP_Float.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySP_Float.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
