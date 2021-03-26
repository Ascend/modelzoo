export DDK_PATH=${install_path}/ascend-toolkit/latest
export NPU_HOST_LIB=${install_path}/ascend-toolkit/latest/acllib/lib64/stub

g="g++"
cd Benchmark
rm -rf build
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=${g} -DCMAKE_SKIP_RPATH=TRUE
make
