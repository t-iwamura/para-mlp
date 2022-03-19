cpp = g++

cflags = -O3 -fPIC -DARMA_NO_DEBUG -DNDEBUG -Wno-deprecated -mtune=native -march=native -std=c++11 #-mfpmath=both -flto
ompflags = -fopenmp

include_common = -I./src -I/home/iwamura/.pyenv/versions/miniconda3-4.7.12/envs/py37/include/python3.7m
include_boost = -I/home/iwamura/usr/local/boost/include
include_pybind11 = -I/home/iwamura/.pyenv/versions/miniconda3-4.7.12/envs/py37/include
#include_arma = -I/home/seko/usr/include
include_gsl = -I/usr/include
include_eigen3 = -I/usr/include/eigen3
INCLUDE = $(include_common) $(include_boost) $(include_gsl) $(include_eigen3) $(include_pybind11)

loader = $(cpp)

libboost = -L/home/iwamura/usr/local/boost/lib
libarma = -L/usr/lib/ -larmadillo
libgsl = -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas
libomp = -L/usr/lib/x86_64-linux-gnu -lgomp

libs = $(libarma) $(libboost) $(libgsl) $(libomp)

VPATH = src/features src/longrange src/projector src/python
source = gtinv_data.o read_gtinv.o model.o basis_function.o local.o \
    features.o model_params.o neighbor.o projector.o pybind11_mlp.o \
    model_py.o model_single_py.o ewald.o

all: mkdir $(addprefix lib/,$(source))

	$(loader) $(ldflags) -shared $(addprefix lib/,$(source)) \
        $(libs) -o lib/mlpcpp.so

mkdir: 
	@if [ ! -d lib ]; then \
        echo "mkdir lib"; mkdir lib; \
    fi 

lib/%.o: %.cpp
	$(cpp) -c $(cflags) $(ompflags) $(INCLUDE) -o $@ $<

clean:
	rm -rf lib

