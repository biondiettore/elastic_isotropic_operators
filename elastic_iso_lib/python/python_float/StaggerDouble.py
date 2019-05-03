#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyStagger
import pyOperator as Op

#
# class stagger_x(Op.Operator):
# 	"""Wrapper encapsulating PYBIND11 module"""
#
# 	def __init__(self,domain,range):
# 		#Checking if getCpp is present
# 		self.setDomainRange(domain,range)
# 		if("getCpp" in dir(domain)):
# 			domain = domain.getCpp()
# 		if("getCpp" in dir(range)):
# 			range = range.getCpp()
# 		self.pyOp = pyStagger.staggerX(domain,range)
# 		return
#
# 	def forward(self,add,model,data):
# 		#Checking if getCpp is present
# 		if("getCpp" in dir(model)):
# 			model = model.getCpp()
# 		if("getCpp" in dir(data)):
# 			data = data.getCpp()
# 		with pyStagger.ostream_redirect():
# 			self.pyOp.forward(add,model,data)
# 		return
#
# 	def adjoint(self,add,model,data):
# 		#Checking if getCpp is present
# 		if("getCpp" in dir(model)):
# 			model = model.getCpp()
# 		if("getCpp" in dir(data)):
# 			data = data.getCpp()
# 		with pyStagger.ostream_redirect():
# 			self.pyOp.adjoint(add,model,data)
# 		return
#
# 	def dotTestCpp(self,verb=False,maxError=.00001):
# 		"""Method to call the Cpp class dot-product test"""
# 		with pyStagger.ostream_redirect():
# 			result=self.pyOp.dotTest(verb,maxError)
# 		return result
#
# class stagger_z(Op.Operator):
# 	"""Wrapper encapsulating PYBIND11 module"""
#
# 	def __init__(self,domain,range):
# 		#Checking if getCpp is present
# 		self.setDomainRange(domain,range)
# 		if("getCpp" in dir(domain)):
# 			domain = domain.getCpp()
# 		if("getCpp" in dir(range)):
# 			range = range.getCpp()
# 		self.pyOp = pyStagger.staggerZ(domain,range)
# 		return
#
# 	def forward(self,add,model,data):
# 		#Checking if getCpp is present
# 		if("getCpp" in dir(model)):
# 			model = model.getCpp()
# 		if("getCpp" in dir(data)):
# 			data = data.getCpp()
# 		with pyStagger.ostream_redirect():
# 			self.pyOp.forward(add,model,data)
# 		return
#
# 	def adjoint(self,add,model,data):
# 		#Checking if getCpp is present
# 		if("getCpp" in dir(model)):
# 			model = model.getCpp()
# 		if("getCpp" in dir(data)):
# 			data = data.getCpp()
# 		with pyStagger.ostream_redirect():
# 			self.pyOp.adjoint(add,model,data)
# 		return
#
# 	def dotTestCpp(self,verb=False,maxError=.00001):
# 		"""Method to call the Cpp class dot-product test"""
# 		with pyStagger.ostream_redirect():
# 			result=self.pyOp.dotTest(verb,maxError)
# 		return result

class stagger_wfld(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyStagger.staggerWfld(domain,range)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyStagger.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyStagger.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyStagger.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
