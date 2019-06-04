cmake_minimum_required(VERSION 2.8)
project(ELASTIC_ISO_DOUBLE)

#Adding include files
#include_directories(${hypercube_DIRNAME_include})
#include_directories(${genericIO_DIRNAME_include})
include_directories(include)
#include_directories(../prop/include)

#Files to be compiled
set(SRC_DIR src)
set(SOURCE ${SRC_DIR}/staggerWfld.cpp
					 ${SRC_DIR}/padTruncateSource.cpp
				   ${SRC_DIR}/waveEquationElasticGpu.cpp
				 	 # ${SRC_DIR}/kernelsGpuWaveEquationElastic.cu
				   ${SRC_DIR}/waveEquationElasticGpuFunctions.cu
			     # ${SRC_DIR}/cudaErrors.cu
				   ${SRC_DIR}/fdParamElasticWaveEquation.cpp
				 	 ${SRC_DIR}/spaceInterpMulti.cpp
				 ${SRC_DIR}/stagger.cpp)

# #Executables to be compiled
# set(EXECS  ${SRC_DIR}/nonlinearPropElasticShotsGpuMain.cpp)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS ${INC_DIR}/staggerWfld.h
						${INC_DIR}/padTruncateSource.h
					  ${INC_DIR}/waveEquationElasticGpu.h
					  ${INC_DIR}/varDeclareWaveEquation.h
					  ${INC_DIR}/waveEquationElasticGpuFunctions.h
					  ${INC_DIR}/fdParamElasticWaveEquation.h
					  ${INC_DIR}/spaceInterpMulti.h
						${SRC_DIR}/kernelsGpuWaveEquationElastic.cu
						${SRC_DIR}/waveEquationElasticGpuFunctions.cu
						${SRC_DIR}/cudaErrors.cu
					${INC_DIR}/stagger.h)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
								 ${hypercube_BINARY_DIR}
								 ${buffers-lib_BINARY_DIR}
								 ${json_BINARY_DIR}
							   ${sepVector-lib_BINARY_DIR}
								 ${SEPlib_LIBDIR}
							  )

#Compiling the project library
add_library(ELASTIC_ISO_DOUBLE_WE ${SOURCE})

#Compiling the project executables
#add_executable(nonlinearPropElasticShotsGpuMain ${EXECS})
#target_link_libraries(nonlinearPropElasticShotsGpuMain ELASTIC_ISO_DOUBLE_NL genericCpp OpenMP::OpenMP_CXX)
target_link_libraries(ELASTIC_ISO_DOUBLE_WE genericCpp sepVector sep3d sep OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS ELASTIC_ISO_DOUBLE_WE DESTINATION lib)
#install(TARGETS nonlinearPropElasticShotsGpuMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
