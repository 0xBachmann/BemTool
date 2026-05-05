enable_testing()

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_zero_omega_matches_helmholtz
        rh_equals_he_nonrotating.cpp
)
target_link_libraries(test_zero_omega_matches_helmholtz
        PRIVATE GTest::gtest_main
)

add_executable(test_kernel_normal_derivative
        normal_derivative_G.cpp
)
target_link_libraries(test_kernel_normal_derivative
        PRIVATE GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_zero_omega_matches_helmholtz)
gtest_discover_tests(test_kernel_normal_derivative)