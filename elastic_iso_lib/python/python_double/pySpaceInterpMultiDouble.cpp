/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "spaceInterpMulti.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySpaceInterpMulti, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<spaceInterpMulti, std::shared_ptr<spaceInterpMulti>>(clsGeneric,"spaceInterpMulti")  //
      .def(py::init<const std::shared_ptr<double1DReg>, const std::shared_ptr<double1DReg>, const std::shared_ptr<SEP::hypercube>, int&, std::string , int>(),"Initlialize spaceInterpMulti")

      .def("forward",(void (spaceInterpMulti::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &spaceInterpMulti::forward,"Forward")

      .def("adjoint",(void (spaceInterpMulti::*)(const bool, std::shared_ptr<double3DReg>, const std::shared_ptr<double3DReg>)) &spaceInterpMulti::adjoint,"Adjoint")

      .def("dotTest",(bool (spaceInterpMulti::*)(const bool, const float)) &spaceInterpMulti::dotTest,"Dot-Product Test")

			.def("getNDeviceReg",(int (spaceInterpMulti::*)())&spaceInterpMulti::getNDeviceReg,"Get number of regular devices")

			.def("getNDeviceIrreg",(int (spaceInterpMulti::*)())&spaceInterpMulti::getNDeviceIrreg,"Get number of regular devices")

			.def("getRegPosUniqueVector",(std::vector<int> (spaceInterpMulti::*)())&spaceInterpMulti::getRegPosUniqueVector,"Get vector of unique grid point locations")
    ;

}
