rm -rf build
cp ./src/CMakeLists_device_1910.txt ./src/CMakeLists.txt
mkdir -p build/intermediates/device
cd build/intermediates/device
cmake ../../../src/ -DCMAKE_CXX_COMPILER=/usr/local/Ascend/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
make clean
make install
cd -
