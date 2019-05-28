cmake_minimum_required(VERSION 2.8)
project(SEIS_UTILS_FLOAT-lib)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../elastic_iso_float/include)

#Adding python programs
#add_subdirectory(python)

#Adding include files
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
#Files for nonLinear operator library
set(SOURCE ${SRC_DIR}/padElasticFileGpuMain.cpp

)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
)

#Compiling the project library (if any)

# Compiling the project executables (if any)
add_executable(padElasticFileGpuMain ${SRC_DIR}/padElasticFileGpuMain.cpp)


#Adding libray and header files to install destination
target_link_libraries(padElasticFileGpuMain genericCpp OpenMP::OpenMP_CXX)
install(TARGETS padElasticFileGpuMain DESTINATION bin)
