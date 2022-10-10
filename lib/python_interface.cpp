#include <vector>
#include <random>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "include/vccs_supervoxel.h"
#include "codelibrary/geometry/io/xyz_io.h"

namespace py = pybind11;

cl::Array<cl::RGB32Color> random_colors(cl::Array<cl::RPoint3D>& points, cl::Array<int> labels, int n_supervoxels) {
  // make some colors
  cl::Array<cl::RGB32Color> colors(points.size());
  std::mt19937 random;
  cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
  for (int i = 0; i < n_supervoxels; ++i) {
      supervoxel_colors[i] = cl::RGB32Color(random());
  }
  for (int i = 0; i < points.size(); ++i) {
      colors[i] = supervoxel_colors[labels[i]];
  }
  return colors;
}

py::array py_segment(py::array_t<double, py::array::c_style | py::array::forcecast>& array,
                     const float voxel_resolution, const float resolution)
{
  // check input dimensions
  if ( array.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");
  if ( array.shape()[1] != 6 )
    throw std::runtime_error("Input should have size [N,6] i.e. [N, xyzrgb]");

  size_t N = array.shape()[0];
  size_t W = array.shape()[1];
  std::vector<double> pos(array.size());
  // copy py::array -> std::vector
  std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));
  // allocate std::vector (to pass to the C++ function)
  cl::Array<cl::RPoint3D> points;

  // copy py::array -> cl::Array
  py::print("copying ", N," points to cl::RPoint3d");
  for (int i = 0; i < N; i++) {
    // copy XYZ data into RPoints
    points.push_back(cl::RPoint3D(pos.data()[i * W + 0], pos.data()[i * W + 1], pos.data()[i * W + 2]));
  }

  // call pure C++ function
  //py::gil_scoped_release release;
  py::print("Calling supervoxel segmentation");
  VCCSSupervoxel vccs(points.begin(), points.end(), voxel_resolution, resolution);
  cl::Array<int> vccs_labels;
  cl::Array<VCCSSupervoxel::Supervoxel> vccs_supervoxels;
  vccs.Segment(&vccs_labels, &vccs_supervoxels);
  //py::gil_scoped_acquire acquire;

  // TODO: replace this with actual pointcloud colors
  cl::Array<cl::RGB32Color> colors = random_colors(points, vccs_labels, vccs_supervoxels.size());

  // todo, we can allocate this explicitely
  std::vector<double> result;

  py::print("Copying to output");
  for (int i = 0; i < points.size(); i++) {
    result.push_back(points[i].x);
    result.push_back(points[i].y);
    result.push_back(points[i].z);
    result.push_back(pos[i * W + 3]);
    result.push_back(pos[i * W + 4]);
    result.push_back(pos[i * W + 5]);
    result.push_back(static_cast<double>(colors[i].red()));
    result.push_back(static_cast<double>(colors[i].green()));
    result.push_back(static_cast<double>(colors[i].blue()));
    // add the label from the segmentation as output..
    result.push_back(static_cast<double>(vccs_labels[i]));
  }

  // also copy standard colors
  W = W + 3;

  // returns same shape but with one additional point i.e. supervoxel cluster
  std::vector<ssize_t> shape   = { N, W + 1};
  std::vector<ssize_t> strides = { sizeof(double)*(W + 1) , sizeof(double) };
  py::print("Returning values..");
  py::print(result.size(), shape);

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    2,                                       /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
}

PYBIND11_MODULE(vccs_supervoxel, m) {
  m.doc() = R"pbdoc(
      Pybind11 VCCS Supervoxel Wrapper
      -----------------------
      .. currentmodule:: vccs_supervoxel
      .. autosummary::
         :toctree: _generate
         VCCSSupervoxel
  )pbdoc";
  m.def("segment", &py_segment, "basic supervoxel clustering using vccs");
}
