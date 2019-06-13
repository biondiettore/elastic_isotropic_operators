cmake_minimum_required(VERSION 3.8)
project(ELASTIC_ISO_DOUBLE)

#Adding include files
#include_directories(${hypercube_DIRNAME_include})
#include_directories(${genericIO_DIRNAME_include})
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
set(SOURCE_NL ${SRC_DIR}/fdParamElastic.cpp
              ${SRC_DIR}/nonlinearPropElasticShotsGpu.cpp
              ${SRC_DIR}/nonlinearPropElasticGpu.cpp
              ${SRC_DIR}/nonlinearPropElasticGpuFunctions.cu
              ${SRC_DIR}/spaceInterpGpu.cpp
           # ${SRC_DIR}/spaceSincInterpGpu.cpp
              ${SRC_DIR}/interpTimeLinTbb.cpp
         	  ${SRC_DIR}/stagger.cpp)

set(SOURCE_BORN ${SRC_DIR}/BornElasticGpuFunctions.cu
                ${SRC_DIR}/BornElasticGpu.cpp)

#Executables to be compiled
# set(EXECS  ${SRC_DIR}/nonlinearPropElasticShotsGpuMain.cpp)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS ${INC_DIR}/fdParamElastic.h
            ${INC_DIR}/nonlinearPropElasticGpu.h
            ${INC_DIR}/nonlinearPropElasticGpuFunctions.h
            ${SRC_DIR}/nonlinearPropElasticGpuFunctions.cu
            ${SRC_DIR}/kernelsGpuElastic.cu
            ${SRC_DIR}/cudaErrors.cu
            ${INC_DIR}/operator.h
            ${INC_DIR}/spaceInterpGpu.h
            # ${INC_DIR}/spaceSincInterpGpu.h
            ${INC_DIR}/interpTimeLinTbb.h
            ${INC_DIR}/seismicElasticOperator2D.h
            ${INC_DIR}/varDeclare.h
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
add_library(ELASTIC_ISO_DOUBLE_NL ${SOURCE_NL})
add_library(ELASTIC_ISO_DOUBLE_BORN ${SOURCE_BORN})

#Compiling the project executables
# add_executable(padElasticFileGpuMain ${SRC_DIR}/padElasticFileGpuMain.cpp)
# target_link_libraries(padElasticFileGpuMain genericCpp OpenMP::OpenMP_CXX)
target_link_libraries(ELASTIC_ISO_DOUBLE_NL genericCpp sepVector sep3d sep OpenMP::OpenMP_CXX)
target_link_libraries(ELASTIC_ISO_DOUBLE_BORN genericCpp sepVector sep3d sep OpenMP::OpenMP_CXX)


#Adding libray and header files to install destination
install(TARGETS ELASTIC_ISO_DOUBLE_NL DESTINATION lib)
install(TARGETS ELASTIC_ISO_DOUBLE_BORN DESTINATION lib)
# install(TARGETS padElasticFileGpuMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
