#!/usr/bin/env python3
import sys,os
import numpy as np
import Hypercube,SepVector,genericIO
import math

if __name__ == '__main__':

	# IO object
	par=genericIO.io(params=sys.argv)

	# filename for model
	model_filename = par.getString("model")
	model = genericIO.defaultIO.getVector(model_filename)

	# data filename
	data_filename = par.getString("data")

	# Model parameters
	nz = model.getHyper().getAxis(1).n
	nx = model.getHyper().getAxis(2).n
	if model.getHyper().getNdim() == 3:
		nPar = model.getHyper().getAxis(3).n
		oPar = model.getHyper().getAxis(3).o
		dPar = model.getHyper().getAxis(3).d
	else:
		nPar = 1
		oPar = 0
		dPar = 1

	# Parfile
	zPad = par.getInt("zPad")
	xPad = par.getInt("xPad")
	fat = par.getInt("fat", 4)
	blockSize = par.getInt("blockSize", 16)
	surfaceCondition = par.getInt("surfaceCondition",0)

	# Compute size of zPadPlus
	if surfaceCondition == 0:
		nzTotal = zPad * 2 + nz
		ratioz = np.float32(nzTotal) / np.float32(blockSize)
		ratioz = math.ceil(ratioz)
		nbBlockz = ratioz
		zPadPlus = nbBlockz * blockSize - nz - zPad
		nzNew = zPad + zPadPlus + nz
		nzNewTotal = nzNew + 2*fat

	elif surfaceCondition==1:
		nzTotal = zPad + nz + fat
		ratioz = np.float32(nzTotal) / np.float32(blockSize)
		ratioz = math.ceil(ratioz)
		nbBlockz = ratioz
		zPad=fat
		zPadPlus = nbBlockz * blockSize - nz - zPad
		nzNew = zPad + zPadPlus + nz
		nzNewTotal = nzNew + 2*fat
	else:
		print( "ERROR UNKNOWN SURFACE CONDITION PARAMETER" )


	# Compute size of xPadPlus
	nxTotal = xPad * 2 + nx
	ratiox = np.float32(nxTotal) / np.float32(blockSize)
	ratiox = math.ceil(ratiox)
	nbBlockx = ratiox
	xPadPlus = nbBlockx * blockSize - nx - xPad
	nxNew = xPad + xPadPlus + nx
	nxNewTotal = nxNew + 2*fat

	# Compute parameters
	dz = model.getHyper().getAxis(1).d
	oz = model.getHyper().getAxis(1).o - (fat + zPad) * dz
	dx = model.getHyper().getAxis(2).d
	ox = model.getHyper().getAxis(2).o - (fat + xPad) * dx

	# Data
	zAxis = Hypercube.axis(n=nzNewTotal, o=oz, d=dz)
	xAxis = Hypercube.axis(n=nxNewTotal, o=ox, d=dx)
	extAxis = 	Hypercube.axis(n=nPar, o=oPar, d=dPar)
	dataHyper = Hypercube.hypercube(axes=[zAxis, xAxis, extAxis])
	data = SepVector.getSepVector(dataHyper)
	dataFile = par.getString("data")

	# copy central part
	data.getNdArray()[:,fat+xPad:fat+xPad+nx,fat+zPad:fat+zPad+nz] = model.getNdArray()

	# copy top central part
	for iz in np.arange(fat+zPad):
		data.getNdArray()[:,fat+xPad:fat+xPad+nx,iz] = model.getNdArray()[:,:,0]

	# copy bottom central part
	for iz in np.arange(zPadPlus+fat):
		data.getNdArray()[:,fat+xPad:fat+xPad+nx,iz+fat+zPad+nz] = model.getNdArray()[:,:,-1]

	# copy left part
	for ix in np.arange(xPad+fat):
		data.getNdArray()[:,ix,:] = data.getNdArray()[:,xPad+fat,:]

	# copy right part
	for ix in np.arange(xPadPlus+fat):
		for iz in np.arange(nzNewTotal):
			data.getNdArray()[:,ix+fat+nx+xPad,iz] = data.getNdArray()[:,fat+xPad+nx-1,iz]

	# Write model
	data.writeVec(dataFile)

	# Display info
	print(" ")
	print("------------------------ Model padding program --------------------")
	print("Chosen surface condition parameter: ")
	if surfaceCondition == 0: print("(0) no free surface condition",'\n')
	elif surfaceCondition==1: print("(1) free surface condition from Robertsson (1998) chosen.",'\n')

	print("Original nz = ",nz," [samples]")
	print("Original nx = ",nx," [samples]")
	print(" ")
	print("zPadMinus = ",zPad," [samples]")
	print("zPadPlus = ",zPadPlus," [samples]")
	print("xPadMinus = ",xPad," [samples]")
	print("xPadPlus = ",xPadPlus," [samples]")
	print(" ")
	print("blockSize = ",blockSize," [samples]")
	print("FAT = ",fat," [samples]")
	print(" ")
	print("New nz = ",nzNewTotal," [samples including padding and FAT]")
	print("New nx = ",nxNewTotal," [samples including padding and FAT]")
	print("-------------------------------------------------------------------")
	print(" ")
