/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip,PairMLIP)

#else

#ifndef LMP_PAIR_MLIP_H
#define LMP_PAIR_MLIP_H

#include "pair.h"

#include "mlip_pymlcpp.h"
#include "mlip_read_gtinv.h"
#include "mlip_model_params.h"
#include "mlip_features.h"

namespace LAMMPS_NS {

class PairMLIP : public Pair {
 public:
  PairMLIP(class LAMMPS *);
  virtual ~PairMLIP();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  
  virtual double init_one(int, int);
 /* virtual void init_style();
  */

 protected:

  virtual void allocate();

  std::vector<std::string> ele;
  double cutmax;
  vector1d reg_coeffs, scale, mass;
  vector1i types;
  vector2i lm_info;

  struct feature_params fp;
  ModelParams modelp;

  virtual void compute_pair(int, int);
  virtual void compute_gtinv(int, int);

  barray3dc compute_anlm
    (const vector1d& dis_a, const vector2d& diff_a, const vector1i& atom2_a);
  vector4dc compute_anlm_products
    (const barray3dc& anlm, vector1d& dn_array);
//  void compute_fn_ylm_products
//    (const double& dis, const vector1d& diff, 
//     barray2dc& fn_ylm, barray2dc& fn_ylm_dx, 
//     barray2dc& fn_ylm_dy, barray2dc& fn_ylm_dz);

  void polynomial_model1_pair
    (double& fpair, double& evdwl, const int& type1, const int& type2, 
     const vector1d& fn_array, const vector1d& fn_d_array, 
     const vector1d& dn_array, int& eflag);

  void polynomial_model2_pair
      (double& fpair, double& evdwl, const int& type2,
       const vector1d& fn_array, const vector1d& fn_d_array, 
       const vector2d& prod_poly, int& eflag);

  void polynomial_model1_gtinv
      (double& fx, double& fy, double& fz, double& evdwl, 
       const int& type1, const int& type2, 
       const vector1d& fn, const vector1d& fn_dx, 
       const vector1d& fn_dy, const vector1d& fn_dz, 
       const vector1d& dn_array, int& eflag);

  void polynomial_model2_gtinv
      (double& fx, double& fy, double& fz, double& evdwl, const int& type2,
       const vector1d& fn_e, const vector1d& fn_dx, 
       const vector1d& fn_dy, const vector1d& fn_dz, 
       const vector2d& prod_poly, int& eflag);

  void polynomial_model2_products
      (const int& type1, const vector1d& dn_array, vector2d& prod_poly);

  double dot(const vector1d& a, const vector1d& b, const int& sindex);
  double prod_real(const dc& val1, const dc& val2);

  void modify_neighbor
    (vector2d& dis_array, vector3d& diff_array, vector2i& atom2_array);

  void read_pot(char *);
  template<typename T> T get_value(std::ifstream& input);
  template<typename T> std::vector<T> get_value_array
    (std::ifstream& input, const int& size);

};

}

#endif
#endif

