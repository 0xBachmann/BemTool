include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------

add_executable(test_kernel_normal_derivative normal_derivative_G.cpp)
target_link_libraries(test_kernel_normal_derivative PRIVATE GTest::gtest_main)
gtest_discover_tests(test_kernel_normal_derivative PROPERTIES LABELS "slow")

add_executable(test_zero_omega_matches_helmholtz rh_equals_he_nonrotating.cpp)
target_link_libraries(test_zero_omega_matches_helmholtz PRIVATE GTest::gtest_main)
gtest_discover_tests(test_zero_omega_matches_helmholtz PROPERTIES LABELS "slow")

# ------------------------------------------------------------------
# Fast rotating-kernel property tests
# ------------------------------------------------------------------

add_executable(test_rotating_kernel_properties rotating_kernel_properties.cpp)
target_link_libraries(test_rotating_kernel_properties PRIVATE GTest::gtest_main)
gtest_discover_tests(test_rotating_kernel_properties)

# ------------------------------------------------------------------
# Small-mesh operator tests
# ------------------------------------------------------------------

add_executable(test_rotating_operator_symmetry rotating_operator_symmetry.cpp)
target_link_libraries(test_rotating_operator_symmetry PRIVATE GTest::gtest_main)
gtest_discover_tests(test_rotating_operator_symmetry)