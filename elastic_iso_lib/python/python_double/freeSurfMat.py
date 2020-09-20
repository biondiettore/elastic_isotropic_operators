#!/usr/bin/env python3.5

# testing for finding the adjoint of the free-surface operators
# import sympy as sym
# import numpy as np
#
# # Define stencil coefficient
# a = sym.Symbol('a')
# b = sym.Symbol('b')
# c = sym.Symbol('c')
# d = sym.Symbol('d')
#
# zero_row = [0]*18
# top = [zero_row]*8
# stencil_free =  [[0,0,0,0,0,-d,-c,-b,-a,a,b,c,d,0,0,0,0,0]]
# # stencil_free =  [[0,0,0,0,0,0,0,0,-a,a,0,0,0,0,0,0,0,0]]
# # stencil_free =  [[0,0,0,0,0,0,0,0,-a,0,0,0,0,0,0,0,0,0]]
# # stencil_free =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# stencil_free1 = [[0,0,0,0,0,0,-d,-c,-b,-a,a,b,c,d,0,0,0,0]]
# stencil_free2 = [[0,0,0,0,0,0,0,-d,-c,-b,-a,a,b,c,d,0,0,0]]
# stencil_free3 = [[0,0,0,0,0,0,0,0,-d,-c,-b,-a,a,b,c,d,0,0]]
# stencil_free4 = [[0,0,0,0,0,0,0,0,0,-d,-c,-b,-a,a,b,c,d,0]]
# stencil_free5 = [[0,0,0,0,0,0,0,0,0,0,-d,-c,-b,-a,a,b,c,d]]
#
# Dz_p = sym.Matrix(top+stencil_free+stencil_free1+stencil_free2+stencil_free3+stencil_free4+stencil_free5)


# Creating simple elastic stepping rules to find elastic adjoint free-surface conditions
import SepVector
import pyOperator as Op
import numpy as np


class elasticStep(Op.Operator):

	def __init__(self,wavefield,lamb,mu,rho,dz,dx,fat=4):
		"""Constructor for elastic stepper"""
		self.setDomainRange(wavefield,wavefield)
		self.lambNd = lamb.clone().getNdArray()
		self.muNd = mu.clone().getNdArray()
		self.rhoNd = rho.clone().getNdArray()
		self.piNd = lamb.clone().getNdArray() + 2.0*mu.clone().getNdArray()
		self.zCoeff0 = 1.196289062541883 / dz
		self.zCoeff1 = -0.079752604188901 / dz
		self.zCoeff2 = 0.009570312506634 / dz
		self.zCoeff3 = -6.975446437140719e-04 / dz
		self.xCoeff0 = 1.196289062541883 / dx
		self.xCoeff1 = -0.079752604188901 / dx
		self.xCoeff2 = 0.009570312506634 / dx
		self.xCoeff3 = -6.975446437140719e-04 / dx
		self.fat  = fat

	def forward(self,add,model,data):
		"""Forward elastic stepping rule"""
		self.checkDomainRange(model,data)
		nx = model.shape[1]
		nz = model.shape[2]
		if not add:
			data.zero()
		m_arr = model.clone().getNdArray()
		# Applying masking on the FAT region
		m_arr[:,:,:self.fat]=0.0
		m_arr[:,:,-self.fat:]=0.0
		m_arr[:,:self.fat,:]=0.0
		m_arr[:,-self.fat:,:]=0.0
		d_arr = data.getNdArray()
		for ix in range(self.fat,nx-self.fat):
			for iz in range(self.fat,nz-self.fat):
				# updating vx
				d_arr[0,ix,iz] += self.rhoNd[ix,iz]*(self.xCoeff0*(m_arr[2][ix][iz]-m_arr[2][ix-1][iz])+\
													 self.xCoeff1*(m_arr[2][ix+1][iz]-m_arr[2][ix-2][iz])+\
													 self.xCoeff2*(m_arr[2][ix+2][iz]-m_arr[2][ix-3][iz])+\
													 self.xCoeff3*(m_arr[2][ix+3][iz]-m_arr[2][ix-4][iz])+\
													 self.zCoeff0*(m_arr[4][ix][iz+1]-m_arr[4][ix][iz])+\
 													 self.zCoeff1*(m_arr[4][ix][iz+2]-m_arr[4][ix][iz-1])+\
 													 self.zCoeff2*(m_arr[4][ix][iz+3]-m_arr[4][ix][iz-2])+\
 													 self.zCoeff3*(m_arr[4][ix][iz+4]-m_arr[4][ix][iz-3])\
				)
				# updating vz
				d_arr[1,ix,iz] += self.rhoNd[ix,iz]*(self.zCoeff0*(m_arr[3][ix][iz]-m_arr[3][ix][iz-1])+\
													 self.zCoeff1*(m_arr[3][ix][iz+1]-m_arr[3][ix][iz-2])+\
													 self.zCoeff2*(m_arr[3][ix][iz+2]-m_arr[3][ix][iz-3])+\
													 self.zCoeff3*(m_arr[3][ix][iz+3]-m_arr[3][ix][iz-4])+\
													 self.xCoeff0*(m_arr[4][ix+1][iz]-m_arr[4][ix][iz])+\
 													 self.xCoeff1*(m_arr[4][ix+2][iz]-m_arr[4][ix-1][iz])+\
 													 self.xCoeff2*(m_arr[4][ix+3][iz]-m_arr[4][ix-2][iz])+\
 													 self.xCoeff3*(m_arr[4][ix+4][iz]-m_arr[4][ix-3][iz])\
				)
				# updating sigma-xx
				d_arr[2,ix,iz] += self.piNd[ix,iz]*(self.xCoeff0*(m_arr[0][ix+1][iz]-m_arr[0][ix][iz])+\
													self.xCoeff1*(m_arr[0][ix+2][iz]-m_arr[0][ix-1][iz])+\
													self.xCoeff2*(m_arr[0][ix+3][iz]-m_arr[0][ix-2][iz])+\
													self.xCoeff3*(m_arr[0][ix+4][iz]-m_arr[0][ix-3][iz]))+\
							    self.lambNd[ix,iz]*(self.zCoeff0*(m_arr[1][ix][iz+1]-m_arr[1][ix][iz])+\
  													self.zCoeff1*(m_arr[1][ix][iz+2]-m_arr[1][ix][iz-1])+\
													self.zCoeff2*(m_arr[1][ix][iz+3]-m_arr[1][ix][iz-2])+\
													self.zCoeff3*(m_arr[1][ix][iz+4]-m_arr[1][ix][iz-3]))
				# updating sigma-zz
				d_arr[3,ix,iz] += self.lambNd[ix,iz]*(self.xCoeff0*(m_arr[0][ix+1][iz]-m_arr[0][ix][iz])+\
													self.xCoeff1*(m_arr[0][ix+2][iz]-m_arr[0][ix-1][iz])+\
													self.xCoeff2*(m_arr[0][ix+3][iz]-m_arr[0][ix-2][iz])+\
													self.xCoeff3*(m_arr[0][ix+4][iz]-m_arr[0][ix-3][iz]))+\
							       self.piNd[ix,iz]*(self.zCoeff0*(m_arr[1][ix][iz+1]-m_arr[1][ix][iz])+\
  													self.zCoeff1*(m_arr[1][ix][iz+2]-m_arr[1][ix][iz-1])+\
													self.zCoeff2*(m_arr[1][ix][iz+3]-m_arr[1][ix][iz-2])+\
													self.zCoeff3*(m_arr[1][ix][iz+4]-m_arr[1][ix][iz-3]))
				# updating sigma-xz
				d_arr[4,ix,iz] += self.muNd[ix,iz]*(self.zCoeff0*(m_arr[0][ix][iz]-m_arr[0][ix][iz-1])+\
													self.zCoeff1*(m_arr[0][ix][iz+1]-m_arr[0][ix][iz-2])+\
													self.zCoeff2*(m_arr[0][ix][iz+2]-m_arr[0][ix][iz-3])+\
													self.zCoeff3*(m_arr[0][ix][iz+3]-m_arr[0][ix][iz-4])+\
							       					self.xCoeff0*(m_arr[1][ix][iz]-m_arr[1][ix-1][iz])+\
  													self.xCoeff1*(m_arr[1][ix+1][iz]-m_arr[1][ix-2][iz])+\
													self.xCoeff2*(m_arr[1][ix+2][iz]-m_arr[1][ix-3][iz])+\
													self.xCoeff3*(m_arr[1][ix+3][iz]-m_arr[1][ix-4][iz])\
			    )
		return

	def adjoint(self,add,model,data):
		"""Adjoint elastic stepping rule"""
		self.checkDomainRange(model,data)
		nx = model.shape[1]
		nz = model.shape[2]
		if not add:
			model.zero()
		d_arr = data.clone().getNdArray()
		# Applying masking on the FAT region
		d_arr[:,:,:self.fat]=0.0
		d_arr[:,:,-self.fat:]=0.0
		d_arr[:,:self.fat,:]=0.0
		d_arr[:,-self.fat:,:]=0.0
		m_arr = model.getNdArray()
		vx_rho = self.rhoNd*d_arr[0,:,:]
		vz_rho = self.rhoNd*d_arr[1,:,:]
		sigmaxx_pi = self.piNd*d_arr[2,:,:]
		sigmaxx_lamb = self.lambNd*d_arr[2,:,:]
		sigmazz_pi = self.piNd*d_arr[3,:,:]
		sigmazz_lamb = self.lambNd*d_arr[3,:,:]
		sigmaxz_mu = self.muNd*d_arr[4,:,:]
		for ix in range(self.fat,nx-self.fat):
			for iz in range(self.fat,nz-self.fat):
				# updating vx
				m_arr[0,ix,iz] -= (self.xCoeff0*(sigmaxx_pi[ix][iz]-sigmaxx_pi[ix-1][iz])+\
								   self.xCoeff1*(sigmaxx_pi[ix+1][iz]-sigmaxx_pi[ix-2][iz])+\
								   self.xCoeff2*(sigmaxx_pi[ix+2][iz]-sigmaxx_pi[ix-3][iz])+\
								   self.xCoeff3*(sigmaxx_pi[ix+3][iz]-sigmaxx_pi[ix-4][iz])+\
								   self.xCoeff0*(sigmazz_lamb[ix][iz]-sigmazz_lamb[ix-1][iz])+\
   								   self.xCoeff1*(sigmazz_lamb[ix+1][iz]-sigmazz_lamb[ix-2][iz])+\
   								   self.xCoeff2*(sigmazz_lamb[ix+2][iz]-sigmazz_lamb[ix-3][iz])+\
   								   self.xCoeff3*(sigmazz_lamb[ix+3][iz]-sigmazz_lamb[ix-4][iz])+\
								   self.zCoeff0*(sigmaxz_mu[ix][iz+1]-sigmaxz_mu[ix][iz])+\
   								   self.zCoeff1*(sigmaxz_mu[ix][iz+2]-sigmaxz_mu[ix][iz-1])+\
   								   self.zCoeff2*(sigmaxz_mu[ix][iz+3]-sigmaxz_mu[ix][iz-2])+\
   								   self.zCoeff3*(sigmaxz_mu[ix][iz+4]-sigmaxz_mu[ix][iz-3])
				)
				# updating vz
				m_arr[1,ix,iz] -= (self.zCoeff0*(sigmaxx_lamb[ix][iz]-sigmaxx_lamb[ix][iz-1])+\
								   self.zCoeff1*(sigmaxx_lamb[ix][iz+1]-sigmaxx_lamb[ix][iz-2])+\
								   self.zCoeff2*(sigmaxx_lamb[ix][iz+2]-sigmaxx_lamb[ix][iz-3])+\
								   self.zCoeff3*(sigmaxx_lamb[ix][iz+3]-sigmaxx_lamb[ix][iz-4])+\
								   self.zCoeff0*(sigmazz_pi[ix][iz]-sigmazz_pi[ix][iz-1])+\
   								   self.zCoeff1*(sigmazz_pi[ix][iz+1]-sigmazz_pi[ix][iz-2])+\
   								   self.zCoeff2*(sigmazz_pi[ix][iz+2]-sigmazz_pi[ix][iz-3])+\
   								   self.zCoeff3*(sigmazz_pi[ix][iz+3]-sigmazz_pi[ix][iz-4])+\
								   self.xCoeff0*(sigmaxz_mu[ix+1][iz]-sigmaxz_mu[ix][iz])+\
   								   self.xCoeff1*(sigmaxz_mu[ix+2][iz]-sigmaxz_mu[ix-1][iz])+\
   								   self.xCoeff2*(sigmaxz_mu[ix+3][iz]-sigmaxz_mu[ix-2][iz])+\
   								   self.xCoeff3*(sigmaxz_mu[ix+4][iz]-sigmaxz_mu[ix-3][iz])
				)
				# updating sigma-xx
				m_arr[2,ix,iz] -= (self.xCoeff0*(vx_rho[ix+1][iz]-vx_rho[ix][iz])+\
								   self.xCoeff1*(vx_rho[ix+2][iz]-vx_rho[ix-1][iz])+\
								   self.xCoeff2*(vx_rho[ix+3][iz]-vx_rho[ix-2][iz])+\
								   self.xCoeff3*(vx_rho[ix+4][iz]-vx_rho[ix-3][iz])\
				)
				# updating sigma-zz
				m_arr[3,ix,iz] -= (self.zCoeff0*(vz_rho[ix][iz+1]-vz_rho[ix][iz])+\
								   self.zCoeff1*(vz_rho[ix][iz+2]-vz_rho[ix][iz-1])+\
								   self.zCoeff2*(vz_rho[ix][iz+3]-vz_rho[ix][iz-2])+\
								   self.zCoeff3*(vz_rho[ix][iz+4]-vz_rho[ix][iz-3])\
				)
				# updating sigma-xz
				m_arr[4,ix,iz] -= (self.zCoeff0*(vx_rho[ix][iz]-vx_rho[ix][iz-1])+\
								   self.zCoeff1*(vx_rho[ix][iz+1]-vx_rho[ix][iz-2])+\
								   self.zCoeff2*(vx_rho[ix][iz+2]-vx_rho[ix][iz-3])+\
								   self.zCoeff3*(vx_rho[ix][iz+3]-vx_rho[ix][iz-4])+\
								   self.xCoeff0*(vz_rho[ix][iz]-vz_rho[ix-1][iz])+\
   								   self.xCoeff1*(vz_rho[ix+1][iz]-vz_rho[ix-2][iz])+\
   								   self.xCoeff2*(vz_rho[ix+2][iz]-vz_rho[ix-3][iz])+\
   								   self.xCoeff3*(vz_rho[ix+3][iz]-vz_rho[ix-4][iz])\
				)
		return



if __name__ == '__main__':
	nx=200
	nz=300
	lamb = SepVector.getSepVector(ns=[nz,nx],storage="dataDouble")
	mu = SepVector.getSepVector(ns=[nz,nx],storage="dataDouble")
	rho = SepVector.getSepVector(ns=[nz,nx],storage="dataDouble")
	wavefield = SepVector.getSepVector(ns=[nz,nx,5],storage="dataDouble")
	# Setting model parameters
	lamb.set(2.0)+lamb.clone().rand()
	mu.set(1.0)+mu.clone().rand()
	rho.set(2.0)+rho.clone().rand()
	# Creating elastic step operator
	ela_step = elasticStep(wavefield,lamb,mu,rho,0.05,0.05)
	ela_step.dotTest(True)
