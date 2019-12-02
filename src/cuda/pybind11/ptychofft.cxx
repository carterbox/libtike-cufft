#include <pybind11/pybind11.h>
#include "ptychofft.cuh"

namespace py = pybind11;

PYBIND11_MODULE(ptychofft, m){

  py::class_<ptychofft>(m, "ptychofft")
    .def(py::init<int, int, int, int, int, int>(),
      py::arg("ntheta"),
      py::arg("nz"),
      py::arg("n"),
      py::arg("nscan"),
      py::arg("detector_shape"),
      py::arg("probe_shape")
    )
    .def_readonly("ntheta", &ptychofft::ntheta)
    .def_readonly("nz", &ptychofft::nz)
    .def_readonly("n", &ptychofft::n)
    .def_readonly("nscan", &ptychofft::nscan)
    .def_readonly("detector_shape", &ptychofft::detector_shape)
    .def_readonly("probe_shape", &ptychofft::probe_shape)
    .def("fwd", &ptychofft::fwd)
    .def("adj", &ptychofft::adj)
    .def("free", &ptychofft::free)
    ;
}
