project(python-seis_utils)

set(PythonMods elasticParamConvertModule.py dataCompModule.py)
set(PythonMains elasticParamConvertMain.py dataCompMain.py)

set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install(FILES ${PythonMods}  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ DESTINATION ${PYTHON_DIR})
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ DESTINATION bin)
