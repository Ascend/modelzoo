rm -rf ./Benchmark/build

mkdir -p Benchmark/build/intermediates/host
cd Benchmark/build/intermediates/host
cmake ../../../../Benchmark/ -DCAMKE_CXX_COMPILER=g++
make clean
make install
cd -
cd Benchmark/out/
