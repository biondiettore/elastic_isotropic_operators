#Module containing definition for elastic parameter conversion
import pyOperator
import numpy as np

class ElasticConv(pyOperator.Operator):
	"""
	   Operator class to convert elastic parameters
	"""


	def __init__(self,domain,conv_type):
		"""
		   Constructor for elastic parameter conversion:
		   domain    	= [no default] - vector class; Vector for defining domain of the transformation
		   conv_type    = [no default] - int; Conversion kind
			1 = VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
		   -1 = RhoLameMu to VpVsRho (m/s|m/s|kg/m3 <- kg/m3|Pa|Pa)
		"""
		self.setDomainRange(domain,domain)
		self.conv_type=conv_type
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		#Getting pointers to input arrays
		modelNd = model.getNdArray()
		dataNd = data.getNdArray()

		if(not add): data.zero()
		if(self.conv_type == 1):
			#VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
			dataNd[0,:,:] += modelNd[2,:,:] #rho
			dataNd[1,:,:] += modelNd[2,:,:]*(modelNd[0,:,:]*modelNd[0,:,:]-2.0*modelNd[1,:,:]*modelNd[1,:,:]) #lame
			dataNd[2,:,:] += modelNd[2,:,:]*modelNd[1,:,:]*modelNd[1,:,:] #mu
		elif(self.conv_type == -1):
			#RhoLameMu to VpVsRho (kg/m3|Pa|Pa -> m/s|m/s|kg/m3)
			dataNd[0,:,:] += np.sqrt(np.divide((modelNd[1,:,:]+2*modelNd[2,:,:]),modelNd[0,:,:])) #vp
			dataNd[1,:,:] += np.sqrt(np.divide(modelNd[2,:,:],modelNd[0,:,:])) #vs
			dataNd[2,:,:] += modelNd[0,:,:] #rho
		else:
			raise ValueError("ERROR! Unsupported conv_type (current value: %s). Please redefine it!"%(self.conv_type))
		return
