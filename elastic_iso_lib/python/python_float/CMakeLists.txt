cmake_minimum_required(VERSION 2.8)
project(pyElastic_iso_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ELASTIC_ISO_FLOAT library
include_directories(../../elastic_iso_float/prop/include)
include_directories(../../elastic_iso_float/wave/include)

#PYBIND11 library headers
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
#library headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../elastic_isotropic_operators")

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})
find_package(Boost 1.40 COMPONENTS program_options unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

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

# message("Boost: ${Boost_LIBRARY_DIRS}")
# set(LIBS ${Boost_LIBRARY_DIRS} genericCpp sepVector sep3d sep buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})
set(LIBS  genericCpp sepVector sep3d sep buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

#Compiling and adding module to library
pybind11_add_module(pyElastic_iso_float_nl MODULE pyElastic_iso_float_nl.cpp)
target_link_libraries(pyElastic_iso_float_nl PRIVATE ELASTIC_ISO_FLOAT_NL ${LIBS})

pybind11_add_module(pyElastic_iso_float_we MODULE pyElastic_iso_float_we.cpp)
target_link_libraries(pyElastic_iso_float_we PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})

pybind11_add_module(pyElastic_iso_float_born MODULE pyElastic_iso_float_born.cpp)
target_link_libraries(pyElastic_iso_float_born PRIVATE ELASTIC_ISO_FLOAT_BORN ${LIBS})
#
pybind11_add_module(pyStaggerFloat MODULE pyStaggerFloat.cpp)
target_link_libraries(pyStaggerFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})
#
pybind11_add_module(pySpaceInterpMultiFloat MODULE pySpaceInterpMultiFloat.cpp)
target_link_libraries(pySpaceInterpMultiFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})

pybind11_add_module(pyPadTruncateSourceFloat MODULE pyPadTruncateSourceFloat.cpp)
target_link_libraries(pyPadTruncateSourceFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})

#Installing library
set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install( TARGETS pyElastic_iso_float_nl DESTINATION ${PYTHON_DIR}) #Module for non-linear operator
install( TARGETS pyElastic_iso_float_born DESTINATION ${PYTHON_DIR}) #Module for Born operator
install( FILES Elastic_iso_float_prop.py DESTINATION ${PYTHON_DIR})

install( TARGETS pyElastic_iso_float_we DESTINATION ${PYTHON_DIR}) #Module for non-linear operator
install( FILES Elastic_iso_float_we.py DESTINATION ${PYTHON_DIR})
#
install( TARGETS pyStaggerFloat  DESTINATION ${PYTHON_DIR})
install( FILES StaggerFloat.py DESTINATION ${PYTHON_DIR})

install( TARGETS pySpaceInterpMultiFloat  DESTINATION ${PYTHON_DIR})
install( FILES SpaceInterpMultiFloat.py DESTINATION ${PYTHON_DIR})

install( TARGETS pyPadTruncateSourceFloat  DESTINATION ${PYTHON_DIR})
install( FILES PadTruncateSourceFloat.py DESTINATION ${PYTHON_DIR})

install( FILES wriUtilFloat.py DESTINATION ${PYTHON_DIR})

#Inversion-related files
install( FILES inversionUtils.py DESTINATION ${PYTHON_DIR})
install( FILES efwiMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

#Installing Python executables
install( FILES waveEqInvFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveReconMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES nonlinearPythonElasticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ DESTINATION bin)
install( FILES nonlinearPythonElasticFloatSourceEstMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ DESTINATION bin)
install( FILES BornPythonElasticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationPythonElasticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install(FILES waveletFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ DESTINATION bin)
install(FILES forcingTermOpsTest.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install(FILES priorTest.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
