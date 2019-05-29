project(python-seis_utils)

set(PythonMods elasticParamConvertModule.py)
set(PythonMains elasticParamConvertMain.py)

install(FILES ${PythonMods}  DESTINATION lib/python)
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ  DESTINATION bin)
