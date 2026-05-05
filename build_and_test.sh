mkdir -p build
cd build || exit

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16

cd tests || exit
echo "Testing Derivatives of rotating Green's functino"
./test_kernel_normal_derivative

echo "Testing stationary operators and layer potential matching Helmholtz formulations from BemTool"
./test_zero_omega_matches_helmholtz
