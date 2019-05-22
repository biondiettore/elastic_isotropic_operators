cmake_minimum_required(VERSION 2.8)
project(pyElastic_iso_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ELASTIC_ISO_FLOAT library
include_directories(../../elastic_iso_float/prop/include)
include_directories(../../elastic_iso_float/wave/include)

# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
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
  pybind11_add_module(pyElastic_iso_float_nl MODULE pyElastic_iso_float_nl.cpp)
  target_link_libraries(pyElastic_iso_float_nl PRIVATE ELASTIC_ISO_FLOAT_NL ${LIBS})

  pybind11_add_module(pyElastic_iso_float_we MODULE pyElastic_iso_float_we.cpp)
  target_link_libraries(pyElastic_iso_float_we PRIVATE ELASTIC_ISO_FLOAT_WE ${LIBS})
  #
  pybind11_add_module(pyStaggerFloat MODULE pyStaggerFloat.cpp)
  target_link_libraries(pyStaggerFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})
  #
  pybind11_add_module(pySpaceInterpMultiFloat MODULE pySpaceInterpMultiFloat.cpp)
  target_link_libraries(pySpaceInterpMultiFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})

  pybind11_add_module(pyPadTruncateSourceFloat MODULE pyPadTruncateSourceFloat.cpp)
  target_link_libraries(pyPadTruncateSourceFloat PRIVATE ELASTIC_ISO_FLOAT_WE ELASTIC_ISO_FLOAT_NL ${LIBS})
  # pybind11_add_module(pyElastic_iso_float_born MODULE pyElastic_iso_float_born.cpp)
  # target_link_libraries(pyElastic_iso_float_born PRIVATE ELASTIC_ISO_FLOAT_BORN ${LIBS})
  # pybind11_add_module(pyElastic_iso_float_born_ext MODULE pyElastic_iso_float_born_ext.cpp)
  # target_link_libraries(pyElastic_iso_float_born_ext PRIVATE ELASTIC_ISO_FLOAT_BORN_EXT ${LIBS})
  # pybind11_add_module(pyElastic_iso_float_tomo MODULE pyElastic_iso_float_tomo.cpp)
  # target_link_libraries(pyElastic_iso_float_tomo PRIVATE ELASTIC_ISO_FLOAT_TOMO_EXT ${LIBS})
  # pybind11_add_module(pyElastic_iso_float_wemva MODULE pyElastic_iso_float_wemva.cpp)
  # target_link_libraries(pyElastic_iso_float_wemva PRIVATE ELASTIC_ISO_FLOAT_WEMVA_EXT ${LIBS})

  #Installing library
  install( TARGETS pyElastic_iso_float_nl DESTINATION lib/python) #Module for non-linear operator
  install( FILES Elastic_iso_float_nl.py DESTINATION lib/python)

  install( TARGETS pyElastic_iso_float_we DESTINATION lib/python) #Module for non-linear operator
  install( FILES Elastic_iso_float_we.py DESTINATION lib/python)
  #
  install( TARGETS pyStaggerFloat  DESTINATION lib/python)
  install( FILES StaggerFloat.py DESTINATION lib/python)

  install( TARGETS pySpaceInterpMultiFloat  DESTINATION lib/python)
  install( FILES SpaceInterpMultiFloat.py DESTINATION lib/python)

  install( TARGETS pyPadTruncateSourceFloat  DESTINATION lib/python)
  install( FILES PadTruncateSourceFloat.py DESTINATION lib/python)

  install( FILES wriUtilFloat.py DESTINATION lib/python)


  #Installing Python executables
  install( FILES waveReconMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES nonlinearPythonElasticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES waveEquationPythonElasticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)
