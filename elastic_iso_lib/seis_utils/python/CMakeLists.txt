project(python-seis_utils)


set(PythonFiles elasticParamConvertModule.py elasticParamConvertMain.py)

install(FILES ${PythonFiles}  DESTINATION lib/python)
