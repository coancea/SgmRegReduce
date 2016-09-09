EXECUTABLE = sgm-reg-red
SOURCES_CPP= MainTests.cu
CXX = nvcc
CXXFLAGS = -O3
INCLUDES = GenTypes.cu.h
#ParBBHost.cu.h ParBBKernels.cu.h

default: all

all: $(INCLUDES)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(SOURCES_CPP)

clean:
	rm -f $(EXECUTABLE) ./*.ptx 
	@# clean nVidia compiler cache
	rm -rf $(HOME)/.nv/ComputeCache/*

