cmake_minimum_required(VERSION 3.20)

project(BEMTool)

include_directories(bemtool)

file(COPY mesh DESTINATION ${CMAKE_BINARY_DIR})


include(FetchContent)
# -----------------------
# Complex Bessel (FetchContent)
# -----------------------
FetchContent_Declare(
        complex_bessel
        GIT_REPOSITORY https://github.com/joeydumont/complex_bessel.git
)
FetchContent_MakeAvailable(complex_bessel)
link_libraries(complex_bessel::complex_bessel)

find_package(MPI REQUIRED)
link_libraries(MPI::MPI_CXX)

find_package(Eigen3 REQUIRED)
link_libraries(Eigen3::Eigen)

add_executable(test2D test2D.cpp)
add_executable(test3D test3D.cpp)
add_executable(cylinder2D cylinder2d.cpp)
add_executable(scatter_TE_PEC_two_trace scatter_TE_PEC_two_trace.cpp)

add_executable(rotating_dielectric_scatter tests/rotating_dielectric_scatter_2D.cpp)



# -----------------------
# GoogleTest (FetchContent)
# -----------------------
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: prevent overriding parent project compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(rh_converges_to_he
        tests/rh_converges_to_he.cpp
)

target_link_libraries(rh_converges_to_he
        GTest::gtest_main
)

add_executable(rh_potential_converges_to_he
        tests/rh_potential_converges_to_he.cpp
)

target_link_libraries(rh_potential_converges_to_he
        GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(rh_converges_to_he rh_potential_converges_to_he)