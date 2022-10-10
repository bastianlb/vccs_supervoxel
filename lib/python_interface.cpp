#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/vccs_supervoxel.h"
#include "codelibrary/geometry/io/xyz_io.h"

namespace py = pybind11;

namespace vccs_py {
  PYBIND11_MODULE(mkv_extractor, m) {
    py::class_<VCCSSupervoxel>(m, "VCCSSupervoxel")
      .def(py::init(&VCCSSupervoxel::create));
  }
}
