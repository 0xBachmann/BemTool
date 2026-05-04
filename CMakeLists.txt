cmake_minimum_required(VERSION 3.20)

project(BEMTool)

include_directories(bemtool)

file(COPY mesh DESTINATION ${CMAKE_BINARY_DIR})

find_package(MPI REQUIRED)
link_libraries(MPI::MPI_CXX)

find_package(Eigen3 REQUIRED)
link_libraries(Eigen3::Eigen)

add_executable(test2D test2D.cpp)
add_executable(test3D test3D.cpp)

add_executable(pec_cylinder rotation/pec_cylinder.cpp)
add_executable(dielectric_cylinder rotation/dielectric_cylinder.cpp)


enable_testing()
add_subdirectory(tests)

