nts=2000  #Number of time samples
dts=0.004 #Coarse sampling [s]
sub=10	  #Ratio between coarse and propagation samplings
nz=920    #Number of samples in the z direction
dz=5      #Sampling of z direction [m]
nx=3624   #Number of samples in the x direction
dx=5      #Sampling of x direction [m]
fMax=16   #Maximum propagated frequency [Hz]
mod_par=1 #Model parameterization (1 = VpVsRho)

#The following numbers are provided when the model is padded
zPadMinus=100               #Number of padding samples on top of the model
zPadPlus=111	            #Number of padding samples on bottom of the model
xPadMinus=100               #Number of padding samples on left of the model
xPadPlus=115                #Number of padding samples on right of the model

#Source geometry
nExp=1                      #Number of shots
zSource=3                   #Shots' depth
xSource=850                 #X position of the first shot
spacingShots=10	            #Shot interval

#Receiver geometry
nReceiver=3401              #Number of receivers
depthReceiver=3             #Receivers' depth
oReceiver=1                 #X position of the first receiver
dReceiver=1                 #Receiver interval

#GPU related variables
nGpu=1                      #Number of GPU cards to parallelize over shots
#iGpu=0                     #GPU-card IDs to parallelize over shots (it can be a comma-separated list). Overwrites nGpu!
blockSize=16                #BlockSize of GPU grid
info=1                      #Verbosity
