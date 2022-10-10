cmake_minimum_required(VERSION 3.13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(vccs VERSION 1.0.0 LANGUAGES CXX)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")

SET(APP_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/point_cloud/grid_sample.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/point_cloud/pca_estimate_normals.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/algorithm.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/array.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/bits.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/equal.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/log.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/macros.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/message.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/object_pool.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/base/string_printf.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/io/xyz_io.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/box_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/box_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/circle_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/line_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/line_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/plane_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/point_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/point_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/point_nd.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/polygon_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/segment_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/segment_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/sphere_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/triangle_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/kernel/triangle_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/transform/project_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/util/center_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/util/center_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/util/compare_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/util/compare_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/util/distance_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/point_octree.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix/decompose/tridiagonal_decompose.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix/matrix.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix/matrix_util.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix/symmetric_eigen.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix/triplet_matrix.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/vector/vector.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/vector/vector_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/vector/vector_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/angle.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/matrix.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/math/vector.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/kernel/covariance.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/kernel/deviation.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/kernel/mean.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/kernel/median.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/kernel/ranking.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/regression/linear_least_squares_fitting.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/principal_component_analysis_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/statistics/principal_component_analysis_3d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/array/array_2d.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/angular.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/cosine.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/euclidean.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/jaccard.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/manhattan.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/metric/squared_euclidean.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/set/disjoint_set.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/tree/kd_tree.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/util/tree/octree.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/visualization/color/rgb32_color.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
)

add_library(vccs_interface
  "${CMAKE_CURRENT_SOURCE_DIR}/include/vccs_knn_supervoxel.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/vccs_supervoxel.h"
)

target_include_directories(vccs_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIRECTORES} include)
set_target_properties(vccs_interface PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET vccs_interface PROPERTY CXX_STANDARD 17)


SET(APP_SOURCES
      "main.cc"
)

SET(TARGET_NAME vccs)
add_executable(${TARGET_NAME} ${APP_SOURCES} ${APP_HEADERS} )
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE vccs_interface)

add_subdirectory(lib/pybind11)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

SET(LIB_NAME vccs_supervoxel)
pybind11_add_module(${LIB_NAME} ${APP_HEADERS} lib/python_interface.cpp)
