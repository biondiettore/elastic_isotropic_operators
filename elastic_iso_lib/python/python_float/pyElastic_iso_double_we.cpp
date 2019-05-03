#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "waveEquationElasticGpu.h"
//#include "spaceInterpGpu.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyElastic_iso_double_we, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<waveEquationElasticGpu, std::shared_ptr<waveEquationElasticGpu>>(clsGeneric,"waveEquationElasticGpu")
      .def(py::init<std::shared_ptr<SEP::double4DReg> , std::shared_ptr<SEP::double4DReg>, std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>  >(), "Initialize a waveEquationElasticGpu")

      .def("forward", (void (waveEquationElasticGpu::*)(const bool, const std::shared_ptr<SEP::double4DReg>, std::shared_ptr<SEP::double4DReg>)) &waveEquationElasticGpu::forward, "Forward")

      .def("adjoint", (void (waveEquationElasticGpu::*)(const bool, const std::shared_ptr<SEP::double4DReg>, std::shared_ptr<SEP::double4DReg>)) &waveEquationElasticGpu::adjoint, "Adjoint")

      .def("dotTest",(bool (waveEquationElasticGpu::*)(const bool, const float)) &waveEquationElasticGpu::dotTest,"Dot-Product Test")

      ;
}
