cmake_minimum_required(VERSION 2.8)
project(pyElastic_iso_double)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ELASTIC_ISO_DOUBLE library
include_directories(../../elastic_iso_double/prop/include)
include_directories(../../elastic_iso_double/wave/include)

#PYBIND11 library headers
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
#library headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../elastic_isotropic_operators")

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${GENERIC_EXPORT_INC_DIRS})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
			 ${hypercube_BINARY_DIR}
			 ${buffers-lib_BINARY_DIR}
			 ${json_BINARY_DIR}
			 ${sepVector-lib_BINARY_DIR}
			 ${SEPlib_LIBDIR}
)

set(LIBS genericCpp sepVector sep3d sep buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

#Compiling and adding module to library
pybind11_add_module(pyElastic_iso_double_nl MODULE pyElastic_iso_double_nl.cpp)
target_link_libraries(pyElastic_iso_double_nl PRIVATE ELASTIC_ISO_DOUBLE_NL ${LIBS})

pybind11_add_module(pyElastic_iso_double_born MODULE pyElastic_iso_double_born.cpp)
target_link_libraries(pyElastic_iso_double_born PRIVATE ELASTIC_ISO_DOUBLE_BORN ${LIBS})

pybind11_add_module(pyElastic_iso_double_we MODULE pyElastic_iso_double_we.cpp)
target_link_libraries(pyElastic_iso_double_we PRIVATE ELASTIC_ISO_DOUBLE_WE ELASTIC_ISO_DOUBLE_NL ${LIBS})

pybind11_add_module(pyStaggerDouble MODULE pyStaggerDouble.cpp)
target_link_libraries(pyStaggerDouble PRIVATE ELASTIC_ISO_DOUBLE_WE ELASTIC_ISO_DOUBLE_NL ${LIBS})

# pybind11_add_module(pySpaceInterpMultiDouble MODULE pySpaceInterpMultiDouble.cpp)
# target_link_libraries(pySpaceInterpMultiDouble PRIVATE ELASTIC_ISO_DOUBLE_WE ELASTIC_ISO_DOUBLE_NL ${LIBS})

pybind11_add_module(pyPadTruncateSourceDouble MODULE pyPadTruncateSourceDouble.cpp)
target_link_libraries(pyPadTruncateSourceDouble PRIVATE ELASTIC_ISO_DOUBLE_WE ELASTIC_ISO_DOUBLE_NL ${LIBS})

#Installing library
set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install( TARGETS pyElastic_iso_double_nl DESTINATION ${PYTHON_DIR}) #Module for non-linear operator
install( TARGETS pyElastic_iso_double_born DESTINATION ${PYTHON_DIR}) #Module for Born operator
install( FILES Elastic_iso_double_prop.py DESTINATION ${PYTHON_DIR})

install( TARGETS pyElastic_iso_double_we DESTINATION ${PYTHON_DIR}) #Module for wave-equation operator
install( FILES Elastic_iso_double_we.py DESTINATION ${PYTHON_DIR})

install( TARGETS pyStaggerDouble  DESTINATION ${PYTHON_DIR})
install( FILES StaggerDouble.py DESTINATION ${PYTHON_DIR})

# install( TARGETS pySpaceInterpMultiDouble  DESTINATION ${PYTHON_DIR})
# install( FILES SpaceInterpMultiDouble.py DESTINATION ${PYTHON_DIR})

install( TARGETS pyPadTruncateSourceDouble  DESTINATION ${PYTHON_DIR})
install( FILES PadTruncateSourceDouble.py DESTINATION ${PYTHON_DIR})

install( FILES wriUtilDouble.py DESTINATION ${PYTHON_DIR})


#Installing Python executables
install( FILES nonlinearPythonElasticDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES BornPythonElasticDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationPythonElasticDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveReconMainDouble.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
