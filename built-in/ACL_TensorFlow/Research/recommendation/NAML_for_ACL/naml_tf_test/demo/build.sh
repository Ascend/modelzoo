build_dir=./build/intermediates/host
if [  ! -d "$build_dir" ];then
    mkdir -p $build_dir
fi

cd $build_dir
rm -rf *
cmake3 ../../../ -DCMAKE_CXX_COMPILER=g++
make install
cd -

