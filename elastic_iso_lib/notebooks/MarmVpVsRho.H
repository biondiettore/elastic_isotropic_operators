Input from a tty

Segy2sep:   stuart@jarvis   Thu Jan 31 13:35:34 2019
		sets next: in="/net/storm/scr1/stuart/Segy2sep@gCCK1j"
		junkME=0
		hff="Dat/MODEL_P-WAVE_VELOCITY_1.25m.H@@"
		same_record_number=1
	 	n2=1 o2=1 d2=1 label2="trace number" 

n3=1 n4=1 n5=1 n6=1 n7=1 n8=1 n9=1 

	n1=2801  o1=0.000000  d1=0.001250   label1="none"   unit1="none"
		gff="-1"
		esize=4
		data_format="xdr_float"
		junkME=0
		same_record_number=1
	 	n2=13601 o2=1 d2=1 label2="trace number" 

n3=1 n4=1 n5=1 n6=1 n7=1 n8=1 n9=1 

	n1=2801  o1=0.000000  d1=0.001250   label1="none"   unit1="none"
		gff="-1"
		esize=4


Window:   stuart@jarvis   Thu Jan 31 13:35:58 2019
		sets next: in="/net/storm/scr1/stuart/Window@IGTvtU"
		In:	float	Fortran (2801,13601,1,1)	C [1][1][13601][2801]
		Out:	float	Fortran (701,3401,1,1)	C [1][1][3401][701]
	 From History:
		 n4=1 n3=1 n2=13601 n1=2801 esize=4
	 From Par:
		 n1=701  #f1=0  #j1=4  #min1=0.000000  #max1=3.5
		 n2=3401  #f2=0  #j2=4  #min2=1.000000  #max2=13601
		 n3=1  #f3=0  #j3=1  #min3=0.000000  #max3=0
		 n4=1  #f4=0  #j4=1  #min4=0  #max4=0
	 Make header useful for further processing:
		 n1=701 	 n2=3401 	 n3=1 	 n4=1
		 o1=0.000000 	 o2=1.000000 	 o3=0.000000 	 o4=0.000000
		 d1=0.005 	 d2=4 	 d3=1 	 d4=1
		 label1="none" 	 label2="trace number" 	 label3="" 	 label4=""
		data_format="xdr_float"

hff=-1 o2=0.0 d1=5.000 d2=5.000 label1='z [m]' label2='x [m]' label3='P-WAVE_VELOCITY'


Cat3d:   stuart@jarvis   Thu Jan 31 13:38:34 2019
		sets next: in="/net/storm/scr1/stuart/Cat3d@ebZXdT"
		axis=3
		file1="./Dat/P-WAVE_VELOCITY-true.H"
		file2="./Dat/S-WAVE_VELOCITY-true.H"
		file3="./Dat/DENSITY-true.H"
		junkME=0
	n2=3401  o2=0.000000  d2=5.000000   label2="x [m]"   unit2="Undefined"
	n3=3  o3=0.000000  d3=1.000000
		gff="-1"
		hff="-1"
hff=-1 gff=-1  n4=1  n5=1  n6=1  n7=1  n8=1  n9=1 
	n1=701  o1=0.000000  d1=5.000000   label1="z [m]"   unit1="none"
		hff="-1"
		gff="-1"
		esize=4
		data_format="xdr_float"



Cp:   ettore@jarvis   Thu Feb  7 19:33:46 2019
		sets next: in="/homes/sep/ettore/research/elastic_tests/Elastic_gpu_tests/Dat_NR/VpVsRho_true.H@"
		copy_intag="/homes/sep/ettore/research/packages/elastic_gpu/elastic_iso_double_lib/test/Elastic_Marmousi_test/Dat/VPVSRHO-true.H"
		data_format="xdr_float"


Cp:   ettore@thanos   Fri Jul 12 13:33:11 2019
		sets next: in="/homes/sep/ettore/elastic_modeling/Vel/MarmVpVsRho.H@"
		copy_intag="/homes/sep/ettore/research/elastic_tests/Elastic_gpu_tests/Dat_NR/VpVsRho_true.H"
		data_format="xdr_float"
in=./MarmVpVsRho.H@
