/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
// #include "stagger.h"
#include "staggerWfld.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyStaggerFloat, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    // py::class_<staggerX, std::shared_ptr<staggerX>>(clsGeneric,"staggerX")  //
    //   .def(py::init<std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>>(),"Initlialize staggerX")
		//
    //   .def("forward",(void (staggerX::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &staggerX::forward,"Forward")
		//
    //   .def("adjoint",(void (staggerX::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &staggerX::adjoint,"Adjoint")
		//
    //   .def("dotTest",(bool (staggerX::*)(const bool, const float)) &staggerX::dotTest,"Dot-Product Test")
    // ;
		//
		//
    // py::class_<staggerZ, std::shared_ptr<staggerZ>>(clsGeneric,"staggerZ")  //
    //   .def(py::init<std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>>(),"Initlialize staggerZ")
		//
    //   .def("forward",(void (staggerZ::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &staggerZ::forward,"Forward")
		//
    //   .def("adjoint",(void (staggerZ::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &staggerZ::adjoint,"Adjoint")
		//
    //   .def("dotTest",(bool (staggerZ::*)(const bool, const float)) &staggerZ::dotTest,"Dot-Product Test")
    // ;

		py::class_<staggerWfld, std::shared_ptr<staggerWfld>>(clsGeneric,"staggerWfld")  //
      .def(py::init<std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>>(),"Initlialize staggerWfld")

      .def("forward",(void (staggerWfld::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &staggerWfld::forward,"Forward")

      .def("adjoint",(void (staggerWfld::*)(const bool, std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>)) &staggerWfld::adjoint,"Adjoint")

      .def("dotTest",(bool (staggerWfld::*)(const bool, const float)) &staggerWfld::dotTest,"Dot-Product Test")
    ;
}
