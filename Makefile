INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)


BUILD_TYPE?=
# keep standard at C11 and C++11
CFLAGS   = -I./ncnn -I./ncnn/src -I./ncnn/build/src/ -I. -I./stable-diffusion/x86/vs2019_opencv-mobile_ncnn-dll_demo/vs2019_opencv-mobile_ncnn-dll_demo -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I./ncnn -I./ncnn/src -I./ncnn/build/src/ -I. -I./stable-diffusion/x86/vs2019_opencv-mobile_ncnn-dll_demo/vs2019_opencv-mobile_ncnn-dll_demo  -O3 -DNDEBUG -std=c++11 -fPIC
LDFLAGS  = 

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
#
# Print build information
#

$(info I llama.cpp build info: )

ncnn/build/src/libncnn.a:
	cd ncnn && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON .. && make -j$(shell nproc)

ncnn/net.h:
	cd ncnn && cp -rfv src/* ./

stablediffusion.o:
	$(CXX) $(CXXFLAGS) stablediffusion.cpp -o stablediffusion.o -c $(LDFLAGS)

unpack: ncnn/build/src/libncnn.a
	mkdir -p unpack && cd unpack && ar x ../ncnn/build/src/libncnn.a

libstablediffusion.a: stablediffusion.o unpack $(EXTRA_TARGETS)
	ar src libstablediffusion.a stablediffusion.o ncnn/build/src/libncnn.a $(shell ls unpack/* | xargs echo)

example/main: libstablediffusion.a
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go build -x -o example/main ./example

clean:
	rm -rf *.o
	rm -rf *.a
	rm -rf unpack