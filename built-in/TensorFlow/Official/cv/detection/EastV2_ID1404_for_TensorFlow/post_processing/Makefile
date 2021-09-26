#CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)
tmpFLAGS = -I/home/xhyang/anaconda3/envs/tf/include/python3.6m -I/home/xhyang/anaconda3/envs/tf/include/python3.6m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -O3 -ffunction-sections -pipe -isystem /home/xhyang/anaconda3/envs/tf/include -fdebug-prefix-map=/tmp/build/80754af9/python_1588903631989/work=/usr/local/src/conda/python-3.6.10 -fdebug-prefix-map=/home/xhyang/anaconda3/envs/tf=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall
CXXFLAGS = -I include -std=c++11 -O3 $(tmpFLAGS)

LDFLAGS = $(shell python3-config --ldflags)

DEPS = $(shell find include -xtype f)
CXX_SOURCES = pse.cpp

LIB_SO = pse.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
