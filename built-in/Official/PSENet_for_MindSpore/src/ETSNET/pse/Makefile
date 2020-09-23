CXXFLAGS = -I include  -std=c++11 -O3

CXX_SOURCES = adaptor.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
