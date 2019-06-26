#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "BornElasticShotsGpu.h"
//#include "spaceInterpGpu.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyElastic_iso_float_born, clsGeneric) {

	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

	py::class_<BornElasticShotsGpu, std::shared_ptr<BornElasticShotsGpu>>(clsGeneric,"BornElasticShotsGpu")
			.def(py::init<std::shared_ptr<SEP::float3DReg> ,
				 std::shared_ptr<paramObj> ,
				 std::vector<std::shared_ptr<SEP::float3DReg>>,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>> ,
				 std::vector<std::shared_ptr<spaceInterpGpu>>  >(), "Initialize a BornElasticShotsGpu")

			.def("forward", (void (BornElasticShotsGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float4DReg>)) &BornElasticShotsGpu::forward, "Forward")

      		.def("adjoint", (void (BornElasticShotsGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float4DReg>)) &BornElasticShotsGpu::adjoint, "Adjoint")

			.def("setBackground", (void (BornElasticShotsGpu::*)(std::shared_ptr<SEP::float3DReg>)) &BornElasticShotsGpu::setBackground, "setBackground")

			;

}
