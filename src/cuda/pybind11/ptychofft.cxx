#include <pybind11/pybind11.h>
#include "ptychofft.cuh"

namespace py = pybind11;

// Get the array pointer from a cupy array.
size_t get_gpu_ptr(py::handle cupy_array)
{
  py::dict info = cupy_array.attr("__cuda_array_interface__");
  py::tuple data = info["data"];
  return data[0].cast<size_t>();
  // printf("%zu", (size_t)address);
}

// Get the cupy array constructor
namespace cp
{
  auto zeros = py::module::import("cupy").attr("zeros");
}

PYBIND11_MODULE(ptychofft, m){
  m.doc() = "A module for ptychography solvers.";

  py::class_<ptychofft>(m, "ptychofft")

    .def(py::init<int, int, int, int, int, int>(),
      py::arg("detector_shape"), py::arg("probe_shape"),
      py::arg("nscan"), py::arg("nz"), py::arg("n"), py::arg("ntheta") = 1
    )

    .def("__enter__", [](ptychofft& self){
      return self;
    })

    .def("__exit__", [](ptychofft& self, py::args args){
      self.free();
    })

    .def("fwd", [](ptychofft& self, py::handle probe, py::handle scan,
                   py::handle psi){
      auto farfield = cp::zeros(
        py::make_tuple(
          self.ntheta, self.nscan, self.detector_shape, self.detector_shape
        ),
        "complex64"
      );
      self.fwd(
        get_gpu_ptr(farfield),
        get_gpu_ptr(psi),
        get_gpu_ptr(scan),
        get_gpu_ptr(probe)
      );
      return farfield;
      },
      py::arg("probe"), py::arg("scan"), py::arg("psi")
    )

    .def("adj", [](ptychofft& self, py::handle farfield, py::handle probe,
                   py::handle scan){
      auto psi = cp::zeros(
        py::make_tuple(self.ntheta, self.nz, self.n),
        "complex64"
      );
      self.adj(
        get_gpu_ptr(farfield),
        get_gpu_ptr(psi),
        get_gpu_ptr(scan),
        get_gpu_ptr(probe),
        0
      );
      return psi;
    },
    py::arg("farplane"), py::arg("probe"), py::arg("scan")
    )
    .def("adj_probe", [](ptychofft& self, py::handle farfield,
                         py::handle scan, py::handle psi){
      auto probe = cp::zeros(
       py::make_tuple(self.ntheta, self.probe_shape, self.probe_shape),
       "complex64"
      );
      self.adj(
        get_gpu_ptr(farfield),
        get_gpu_ptr(psi),
        get_gpu_ptr(scan),
        get_gpu_ptr(probe),
        1
      );
      return probe;
    },
    py::arg("farplane"), py::arg("scan"), py::arg("psi")
    )
    ;
}
