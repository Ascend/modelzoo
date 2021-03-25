rm -rf build
cp -rp ./src/CMakeLists_host_C75.txt ./src/CMakeLists.txt
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src/ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make clean
make install
cd -
