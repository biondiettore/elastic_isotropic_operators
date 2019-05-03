#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "nonlinearPropElasticShotsGpu.h"
//#include "spaceInterpGpu.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyElastic_iso_double_nl, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<spaceInterpGpu, std::shared_ptr<spaceInterpGpu>>(clsGeneric, "spaceInterpGpu")
      .def(py::init<const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<SEP::hypercube>, int &,std::string , int >(), "Initialize a spaceInterpGpu object using location, velocity, and nt")

      .def(py::init<const std::shared_ptr<double1DReg> , const std::shared_ptr<double1DReg> , const std::shared_ptr<SEP::hypercube> , int&,std::string , int >(), "Initialize a spaceInterpGpu object using zcoord, xcoord, velocity, and nt")

      // .def(py::init<const std::vector<int> &, const std::vector<int> &, const std::shared_ptr<SEP::hypercube>, int &>(), "Initialize a spaceInterpGpu object using coordinates and nt")
      //
      // .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &, const std::shared_ptr<SEP::hypercube>, int &>(), "Initialize a spaceInterpGpu object using sampling in z and x axes, velocity, and nt")

      .def("getInfo", (void (spaceInterpGpu::*)()) &spaceInterpGpu::getInfo, "getInfo")

      ;

  py::class_<nonlinearPropElasticShotsGpu, std::shared_ptr<nonlinearPropElasticShotsGpu>>(clsGeneric,"nonlinearPropElasticShotsGpu")
      .def(py::init<std::shared_ptr<SEP::double3DReg> , std::shared_ptr<paramObj> ,
  			 														std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>> , std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>> ,
  																	std::vector<std::shared_ptr<spaceInterpGpu>>  >(), "Initialize a nonlinearPropElasticShotsGpu")

      .def("forward", (void (nonlinearPropElasticShotsGpu::*)(const bool, const std::shared_ptr<SEP::double4DReg>, std::shared_ptr<SEP::double4DReg>)) &nonlinearPropElasticShotsGpu::forward, "Forward")

      .def("adjoint", (void (nonlinearPropElasticShotsGpu::*)(const bool, const std::shared_ptr<SEP::double4DReg>, std::shared_ptr<SEP::double4DReg>)) &nonlinearPropElasticShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (nonlinearPropElasticShotsGpu::*)(const bool, const std::shared_ptr<double4DReg>, std::shared_ptr<double4DReg>)) &nonlinearPropElasticShotsGpu::forwardWavefield, "Forward with wavefield")

      .def("adjointWavefield",(void (nonlinearPropElasticShotsGpu::*)(const bool, const std::shared_ptr<double4DReg>, std::shared_ptr<double4DReg>)) &nonlinearPropElasticShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("getWavefield", (std::shared_ptr<SEP::double4DReg> (nonlinearPropElasticShotsGpu::*)()) &nonlinearPropElasticShotsGpu::getWavefield, "get wavefield")

      .def("dotTest",(bool (nonlinearPropElasticShotsGpu::*)(const bool, const float)) &nonlinearPropElasticShotsGpu::dotTest,"Dot-Product Test")

      ;
}
