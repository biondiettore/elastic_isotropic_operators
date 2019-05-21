/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
// #include "stagger.h"
#include "SP.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySP_Float, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

		py::class_<SP, std::shared_ptr<SP>>(clsGeneric,"SP")  //
      .def(py::init<std::shared_ptr<float3DReg>, std::shared_ptr<float4DReg>,std::vector<int>>(),"Initlialize SP")

      .def("forward",(void (SP::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float4DReg>)) &SP::forward,"Forward")

      .def("adjoint",(void (SP::*)(const bool, std::shared_ptr<float3DReg>, const std::shared_ptr<float4DReg>)) &SP::adjoint,"Adjoint")

      .def("dotTest",(bool (SP::*)(const bool, const float)) &SP::dotTest,"Dot-Product Test")
    ;
}
