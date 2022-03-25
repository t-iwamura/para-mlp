#!/bin/zsh

for i in {basis_function.cpp,basis_function.h,read_gtinv.cpp,read_gtinv.h,gtinv_data.cpp,gtinv_data.h,features.cpp,features.h,model_params.cpp,model_params.h};do
    sed -e 's/features\///g' ./src/features/$i | \
    sed -e 's/#include "/#include "mlip_/g'| \
    sed -e 's/#ifndef __/#ifndef __MLIP_/g'| \
    sed -e 's/#define __/#define __MLIP_/g' >| ./lammps/src/USER-MLIP/mlip_$i
    echo " $i is copied. "
done
#cp ./src/mlpcpp.h ./lammps/src/USER-MLIP/mlip_mlpcpp.h
