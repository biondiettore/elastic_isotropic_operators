/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
// #include "stagger.h"
#include "cosTaperWfld.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyCosTaperWfld, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");


		py::class_<cosTaperWfld, std::shared_ptr<cosTaperWfld>>(clsGeneric,"cosTaperWfld")  //
      .def(py::init<std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>,int , int , int , float ,float>(),"Initlialize cosTaperWfld")

      .def("forward",(void (cosTaperWfld::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &cosTaperWfld::forward,"Forward")

      .def("adjoint",(void (cosTaperWfld::*)(const bool, std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>)) &cosTaperWfld::adjoint,"Adjoint")

      .def("dotTest",(bool (cosTaperWfld::*)(const bool, const float)) &cosTaperWfld::dotTest,"Dot-Product Test")
    ;
}
