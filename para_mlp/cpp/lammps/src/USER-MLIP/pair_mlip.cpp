/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Atsuto Seko
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

#include "pair_mlip.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLIP::PairMLIP(LAMMPS *lmp) : Pair(lmp)
{}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIP::~PairMLIP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIP::compute(int eflag, int vflag){
    if (fp.des_type == "pair") compute_pair(eflag, vflag);
    else if (fp.des_type == "gtinv") compute_gtinv(eflag, vflag);
}

void PairMLIP::compute_pair(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    vector2d dis_array;
    vector3d diff_array;
    vector2i atom2_array;
    modify_neighbor(dis_array, diff_array, atom2_array);

    int inum = list->inum;

    int *ilist; 
    ilist = list->ilist;
    tagint *tag = atom->tag;

    vector2d evdwl_array(inum), fpair_array(inum);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        evdwl_array[ii].resize(dis_array[ii].size());
        fpair_array[ii].resize(dis_array[ii].size());
    
        int i,j,type1,type2,sindex0,sindex,sindex_dn;
        double fpair,dis,evdwl;

        const int n_fn = modelp.get_n_fn(), n_des = modelp.get_n_des();
        const int n_type = fp.n_type;
        vector1d fn(n_fn), fn_d(n_fn);

        i = ilist[ii], type1 = types[tag[i]-1];
 
        // first part of polynomial model correction
        vector1d dn_array(n_des, 0.0);
        vector2d prod_poly(n_type);
        if (fp.maxp > 1){
            for (int jj = 0; jj < dis_array[ii].size(); ++jj){
                j = atom2_array[ii][jj], type2 = types[tag[j]-1];
                sindex_dn = type2 * n_fn;
                get_fn(dis_array[ii][jj], fp, fn);
                for (int n = 0; n < n_fn; ++n) dn_array[sindex_dn+n] += fn[n];
            }
            if (fp.model_type == 2){
                polynomial_model2_products(type1, dn_array, prod_poly);
            }
        }
        // end: first part of polynomial model correction 
        
        sindex0 = type1 * modelp.get_n_coeff();
        for (int jj = 0; jj < dis_array[ii].size(); ++jj) {
            j = atom2_array[ii][jj], type2 = types[tag[j]-1];
            dis = dis_array[ii][jj];
            get_fn(dis, fp, fn, fn_d);

            sindex = sindex0 + 1 + (type2 * modelp.get_n_fn());
            fpair = dot(fn_d, reg_coeffs, sindex);
            if (eflag) evdwl = dot(fn, reg_coeffs, sindex);

            // polynomial model correction
            if (fp.maxp > 1 and fp.model_type == 1){
                polynomial_model1_pair
                    (fpair, evdwl, type1, type2, fn, fn_d, dn_array, eflag);
            }
            else if (fp.maxp > 1 and fp.model_type == 2){
                polynomial_model2_pair
                    (fpair, evdwl, type2, fn, fn_d, prod_poly, eflag);
            }
            // polynomial model correction: end

            fpair *= - 1.0 / dis;
            evdwl_array[ii][jj] = evdwl;
            fpair_array[ii][jj] = fpair;
        }
    }

    int i,j,type1,sindex0;
    double delx,dely,delz,fpair,evdwl;
    double **f = atom->f;
    for (int ii = 0; ii < inum; ii++) {
        i = ilist[ii], type1 = types[tag[i]-1];
        sindex0 = type1 * modelp.get_n_coeff();
        for (int jj = 0; jj < dis_array[ii].size(); ++jj) {
            j = atom2_array[ii][jj];
            delx = diff_array[ii][jj][0];
            dely = diff_array[ii][jj][1];
            delz = diff_array[ii][jj][2];
            fpair = fpair_array[ii][jj];
            evdwl = evdwl_array[ii][jj];

            f[i][0] += delx*fpair;
            f[i][1] += dely*fpair;
            f[i][2] += delz*fpair;
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
            if (evflag){
                ev_tally_full(i,evdwl,0.0,fpair,delx,dely,delz);
                ev_tally_full(j,evdwl,0.0,fpair,delx,dely,delz);
            }
        }
        if (eflag) ev_tally_full(i,2*reg_coeffs[sindex0],0.0,0.0,0.0,0.0,0.0);
    }
}

 
void PairMLIP::compute_gtinv(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    vector2d dis_array;
    vector3d diff_array;
    vector2i atom2_array;
    modify_neighbor(dis_array, diff_array, atom2_array);

    int *ilist; 
    int inum = list->inum;
    ilist = list->ilist;

    vector2d evdwl_array(inum),fx_array(inum),fy_array(inum),fz_array(inum);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided), \
        shared(dis_array,diff_array,atom2_array)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        evdwl_array[ii].resize(dis_array[ii].size());
        fx_array[ii].resize(dis_array[ii].size());
        fy_array[ii].resize(dis_array[ii].size());
        fz_array[ii].resize(dis_array[ii].size());

        int i,j,type1,type2,sindex0,sindex_reg,m,lm1,lm2,dindex;
        double delx,dely,delz,dis,evdwl,fx,fy,fz,
            costheta,sintheta,cosphi,sinphi,coeff,cc,sum,sumx,sumy,sumz,regc;
        dc f1,ylm_dphi,d0,d1,d2,term1,term2;

        tagint *tag = atom->tag;
        i = ilist[ii], type1 = types[tag[i]-1];

        // anlm : order parameters for {type,n,l,m}
        const barray3dc &anlm = compute_anlm
            (dis_array[ii], diff_array[ii], atom2_array[ii]);

        // products of anlm[1:], dn_array used for polynomial correction
        vector1d dn_array;
        const vector4dc &prod_array = compute_anlm_products(anlm, dn_array);

        // products in polynomial correction (model 2)
        vector2d prod_poly;
        if (fp.model_type == 2 or fp.model_type == 3){
            polynomial_model2_products(type1, dn_array, prod_poly);
        }

        const int n_fn = modelp.get_n_fn(), n_des = modelp.get_n_des(), 
              n_lm = lm_info.size(), n_lm_all = 2 * n_lm - fp.maxl - 1;

        vector1d fn,fn_d,fn_e,fn_dx,fn_dy,fn_dz;
        vector1dc ylm,ylm_dtheta;
        barray2dc fn_ylm(boost::extents[n_fn][n_lm_all]),
                  fn_ylm_dx(boost::extents[n_fn][n_lm_all]),
                  fn_ylm_dy(boost::extents[n_fn][n_lm_all]),
                  fn_ylm_dz(boost::extents[n_fn][n_lm_all]);

        sindex0 = type1 * modelp.get_n_coeff();
        for (int jj = 0; jj < dis_array[ii].size(); ++jj) {
            j = atom2_array[ii][jj], type2 = types[tag[j]-1];
            dis = dis_array[ii][jj];
            delx = diff_array[ii][jj][0];
            dely = diff_array[ii][jj][1];
            delz = diff_array[ii][jj][2];

            const vector1d &sph = cartesian_to_spherical(diff_array[ii][jj]);
            get_fn(dis, fp, fn, fn_d);
            get_ylm(sph, lm_info, ylm, ylm_dtheta);
            
            costheta = cos(sph[0]), sintheta = sin(sph[0]);
            cosphi = cos(sph[1]), sinphi = sin(sph[1]);
            fabs(sintheta) > 1e-15 ? (coeff = 1.0 / sintheta) : (coeff = 0);
            for (int lm = 0; lm < n_lm; ++lm) {
                m = lm_info[lm][1], lm1 = lm_info[lm][2], lm2 = lm_info[lm][3];
                cc = pow(-1, m); 
               
                ylm_dphi = dc{0.0,1.0} * double(m) * ylm[lm];
                term1 = ylm_dtheta[lm] * costheta;
                term2 = coeff * ylm_dphi;
                d0 = term1 * cosphi - term2 * sinphi;
                d1 = term1 * sinphi + term2 * cosphi;
                d2 = - ylm_dtheta[lm] * sintheta;
                
                for (int n = 0; n < n_fn; ++n) {
                    fn_ylm[n][lm1] = fn[n] * ylm[lm];
                    fn_ylm[n][lm2] = cc * std::conj(fn_ylm[n][lm1]);
                    f1 = fn_d[n] * ylm[lm];
                    fn_ylm_dx[n][lm1] = - (f1 * delx + fn[n] * d0) / dis;
                    fn_ylm_dy[n][lm1] = - (f1 * dely + fn[n] * d1) / dis;
                    fn_ylm_dz[n][lm1] = - (f1 * delz + fn[n] * d2) / dis;
                    fn_ylm_dx[n][lm2] = cc * std::conj(fn_ylm_dx[n][lm1]);
                    fn_ylm_dy[n][lm2] = cc * std::conj(fn_ylm_dy[n][lm1]);
                    fn_ylm_dz[n][lm2] = cc * std::conj(fn_ylm_dz[n][lm1]);
                }
            }

            evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
            if (fp.maxp > 1){
                fn_e.resize(n_des); 
                fn_dx.assign(n_des, 0.0);
                fn_dy.assign(n_des, 0.0);
                fn_dz.assign(n_des, 0.0);
            }

            const auto &info = modelp.get_invariant_info(type2);
            for (int n = 0; n < n_fn; ++n) {
                sindex_reg = sindex0 + 1 + n * fp.lm_array.size();
                for (int t1 = 0; t1 < info.size(); ++t1){
                    const auto &prod = prod_array[type2][n][t1];
                    const auto &lm0 = info[t1].lm0;
                    sum = 0.0, sumx = 0.0, sumy = 0.0, sumz = 0.0;
                    for (int t2 = 0; t2 < lm0.size(); ++t2){
                        sumx += prod_real(prod[t2], fn_ylm_dx[n][lm0[t2]]);
                        sumy += prod_real(prod[t2], fn_ylm_dy[n][lm0[t2]]);
                        sumz += prod_real(prod[t2], fn_ylm_dz[n][lm0[t2]]);
                        if (info[t1].energy == true)
                            sum += prod_real(prod[t2], fn_ylm[n][lm0[t2]]);
                    }
                    regc = reg_coeffs[sindex_reg+info[t1].reg_index];
                    fx += sumx * regc;
                    fy += sumy * regc;
                    fz += sumz * regc;
                    if (info[t1].energy == true) evdwl += sum * regc;

                    if (fp.maxp > 1){
                        dindex = n * fp.lm_array.size() + info[t1].reg_index;
                        fn_dx[dindex] += sumx;
                        fn_dy[dindex] += sumy;
                        fn_dz[dindex] += sumz;
                        if (info[t1].energy == true) fn_e[dindex] = sum;
                    }
                }
            }

            // polynomial model correction
            if (fp.maxp > 1){
                if (fp.model_type == 1){
                    polynomial_model1_gtinv(fx, fy, fz, evdwl, type1, type2, 
                            fn_e, fn_dx, fn_dy, fn_dz, dn_array, eflag);
                }
                else if (fp.model_type == 2 or fp.model_type == 3){
                    polynomial_model2_gtinv(fx, fy, fz, evdwl, type2, 
                        fn_e, fn_dx, fn_dy, fn_dz, prod_poly, eflag);
                }
            }
            // polynomial model correction: end

            evdwl_array[ii][jj] = evdwl;
            fx_array[ii][jj] = fx;
            fy_array[ii][jj] = fy;
            fz_array[ii][jj] = fz;
        }
    }

    double **f = atom->f;

    int i,j,type1,sindex0;
    double fx,fy,fz,evdwl,delx,dely,delz;
    tagint *tag = atom->tag;
    for (int ii = 0; ii < inum; ii++) {
        i = ilist[ii], type1 = types[tag[i]-1];
        sindex0 = type1 * modelp.get_n_coeff();
        for (int jj = 0; jj < dis_array[ii].size(); ++jj) {
            j = atom2_array[ii][jj];
            delx = diff_array[ii][jj][0];
            dely = diff_array[ii][jj][1];
            delz = diff_array[ii][jj][2];
 
            evdwl = evdwl_array[ii][jj];
            fx = fx_array[ii][jj]; 
            fy = fy_array[ii][jj]; 
            fz = fz_array[ii][jj]; 
            f[i][0] += fx, f[i][1] += fy, f[i][2] += fz;
            f[j][0] -= fx, f[j][1] -= fy, f[j][2] -= fz;

            if (evflag){
                ev_tally_xyz_full(i,evdwl,0.0,fx,fy,fz,delx,dely,delz);
                ev_tally_xyz_full(j,evdwl,0.0,fx,fy,fz,delx,dely,delz);
            }
        }
        if (eflag) ev_tally_full(i,2*reg_coeffs[sindex0],0.0,0.0,0.0,0.0,0.0);
    }
    
}
barray3dc PairMLIP::compute_anlm
(const vector1d& dis_a, const vector2d& diff_a, const vector1i& atom2_a){

    const int n_fn = modelp.get_n_fn(), n_lm = lm_info.size(), 
        n_lm_all = 2 * n_lm - fp.maxl - 1, n_type = fp.n_type;

    int j,type2;
    double dis, cc;
    tagint *tag = atom->tag;

    vector1d fn;
    vector1dc ylm;

    barray3dc anlm(boost::extents[n_type][n_fn][n_lm_all]);
    std::fill(anlm.data(), anlm.data() + anlm.num_elements(), 0.0);
    for (int jj = 0; jj < dis_a.size(); ++jj) {
        j = atom2_a[jj], type2 = types[tag[j]-1];
        dis = dis_a[jj];
        const vector1d &sph = cartesian_to_spherical(diff_a[jj]);
        get_fn(dis, fp, fn);
        get_ylm(sph, lm_info, ylm);
        for (int n = 0; n < n_fn; ++n) {
            for (int lm = 0; lm < n_lm; ++lm) {
                anlm[type2][n][lm_info[lm][2]] += fn[n] * ylm[lm];
            }
        }
    }

    for (int type2 = 0; type2 < n_type; ++type2) {
        for (int n = 0; n < n_fn; ++n) {
            for (int lm = 0; lm < n_lm; ++lm) {
                cc = pow(-1, lm_info[lm][1]); 
                anlm[type2][n][lm_info[lm][3]]
                    = cc * std::conj(anlm[type2][n][lm_info[lm][2]]);
            }
        }
    }
    return anlm;
}

vector4dc PairMLIP::compute_anlm_products
(const barray3dc& anlm, vector1d& dn_array){

    const int n_fn = modelp.get_n_fn(), n_des = modelp.get_n_des();
    const int n_type = fp.n_type;
    int dindex, t1;

    vector4dc prod_array(n_type);
    dn_array = vector1d(n_des, 0.0);
    for (int type2 = 0; type2 < n_type; ++type2) {
        const int size = modelp.get_invariant_size(type2);
        prod_array[type2] = vector3dc(n_fn, vector2dc(size));
        for (int n = 0; n < n_fn; ++n) {
            t1 = 0;
            for (const auto& inv: modelp.get_invariant_info(type2)){
                const vector1i &tc = inv.types;
                prod_array[type2][n][t1] = inv.coeffs;
                for (int s1 = 0; s1 < tc.size(); ++s1){
                    for (int t2 = 0; t2 < inv.lm.size(); ++t2){
                        prod_array[type2][n][t1][t2] 
                            *= anlm[tc[s1]][n][inv.lm[t2][s1]];
                    }
                }
                if (fp.maxp > 1 and inv.energy == true){
                    const vector1i &lm0 = inv.lm0;
                    dindex = n * fp.lm_array.size() + inv.reg_index;
                    for (int t2 = 0; t2 < lm0.size(); ++t2){
                        dn_array[dindex] += prod_real
                            (prod_array[type2][n][t1][t2], 
                             anlm[type2][n][lm0[t2]]);
                    }
                }
                ++t1;
            }
        }
    }
    return prod_array;
}


void PairMLIP::polynomial_model1_pair
(double& fpair, double& evdwl, const int& type1, const int& type2, 
 const vector1d& fn_array, const vector1d& fn_d_array, 
 const vector1d& dn_array, int& eflag){

    int sindex_dn = type2 * modelp.get_n_fn();
    int sindex = type1 * modelp.get_n_coeff() 
        + type2 * modelp.get_n_fn() + 1;

    for (int p = 2; p < fp.maxp+1; ++p){
        int sindex_m = sindex + modelp.get_n_des() * (p-1);
        for (int n = 0; n < modelp.get_n_fn(); ++n){
            double prod = reg_coeffs[sindex_m+n] * dn_array[sindex_dn+n];
            for (int p1 = 3; p1 < p+1; ++p1) prod *= dn_array[sindex_dn+n];
            fpair += double(p) * prod * fn_d_array[n];
            if (eflag) evdwl += prod * fn_array[n];
        }
    }
}

void PairMLIP::polynomial_model2_pair
(double& fpair, double& evdwl, const int& type2,
 const vector1d& fn_array, const vector1d& fn_d_array, 
 const vector2d& prod_poly, int& eflag){

    const auto &ct = modelp.get_comb_type(type2);
    const auto &prod = prod_poly[type2];
    for (int i = 0; i < prod.size(); ++i){
        fpair += prod[i] * fn_d_array[ct[i][1]];
        if (ct[i][2] == 1) evdwl += prod[i] * fn_array[ct[i][1]];
    }
}

void PairMLIP::polynomial_model1_gtinv
(double& fx, double& fy, double& fz, double& evdwl, 
 const int& type1, const int& type2, 
 const vector1d& fn_e, const vector1d& fn_dx, 
 const vector1d& fn_dy, const vector1d& fn_dz, 
 const vector1d& dn_array, int& eflag){

    int sindex = type1 * modelp.get_n_coeff() + 1;

    for (int p = 2; p < fp.maxp+1; ++p){
        sindex += modelp.get_n_des();
        for (int n = 0; n < dn_array.size(); ++n){
            double prod = reg_coeffs[sindex+n] * dn_array[n];
            for (int p1 = 3; p1 < p+1; ++p1) prod *= dn_array[n];
            fx += double(p) * prod * fn_dx[n];
            fy += double(p) * prod * fn_dy[n];
            fz += double(p) * prod * fn_dz[n];
            if (eflag) evdwl += prod * fn_e[n];
        }
    }
}

void PairMLIP::polynomial_model2_gtinv
(double& fx, double& fy, double& fz, double& evdwl, const int& type2,
 const vector1d& fn_e, const vector1d& fn_dx, 
 const vector1d& fn_dy, const vector1d& fn_dz, 
 const vector2d& prod_poly, int& eflag){

    const auto &ct = modelp.get_comb_type(type2);
    const auto &prod = prod_poly[type2];
    for (int i = 0; i < prod.size(); ++i){
        fx += prod[i] * fn_dx[ct[i][1]];
        fy += prod[i] * fn_dy[ct[i][1]];
        fz += prod[i] * fn_dz[ct[i][1]];
        if (ct[i][2] == 1) evdwl += prod[i] * fn_e[ct[i][1]];
    }
}

void PairMLIP::polynomial_model2_products
(const int& type1, const vector1d& dn_array, vector2d& prod_poly){

    const int n_type = fp.n_type;
    prod_poly.resize(n_type);

    const int sindex_m = type1 * modelp.get_n_coeff() + 1 + modelp.get_n_des();
    for (int type2 = 0; type2 < n_type; ++type2){
        const auto& ct = modelp.get_comb_type(type2);
        prod_poly[type2].resize(ct.size());
        for (auto i = 0; i < ct.size(); ++i){
            const auto& ct1 = ct[i];
            prod_poly[type2][i] = reg_coeffs[sindex_m+ct1[0]];
        }
        for (auto i = 0; i < ct.size(); ++i){
            const auto& ct1 = ct[i];
            for (int n = 3; n < ct1.size(); ++n) 
                prod_poly[type2][i] *= dn_array[ct1[n]];
        }
    }
}
 

double PairMLIP::dot(const vector1d& a, const vector1d& b, const int& sindex){
    double val(0.0);
    for (int n = 0; n < a.size(); ++n) val += a[n] * b[sindex+n];
    return val;
}

double PairMLIP::prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

void PairMLIP::modify_neighbor
(vector2d& dis_array, vector3d& diff_array, vector2i& atom2_array){

    int i,j,ii,jj,inum,jnum;
    double xtmp,ytmp,ztmp,delx,dely,delz,dis;
    int *ilist,*jlist,*numneigh,**firstneigh;

    inum = list->inum, ilist = list->ilist;
    numneigh = list->numneigh, firstneigh = list->firstneigh;

    double **x = atom->x;
    int nlocal = atom->nlocal;

    dis_array = vector2d(inum);
    diff_array = vector3d(inum);
    atom2_array = vector2i(inum);

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        xtmp = x[i][0], ytmp = x[i][1], ztmp = x[i][2];
        jlist = firstneigh[i], jnum = numneigh[i];
        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = xtmp-x[j][0], dely = ytmp-x[j][1], delz = ztmp-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                dis_array[i].emplace_back(dis);
                diff_array[i].emplace_back(vector1d{delx,dely,delz});
                atom2_array[i].emplace_back(j);
                if (j < nlocal){
                    dis_array[j].emplace_back(dis);
                    diff_array[j].emplace_back(vector1d{-delx,-dely,-delz});
                    atom2_array[j].emplace_back(i);
                }
            }
        }
    }
}


/* ---------------------------------------------------------------------- */

void PairMLIP::allocate()
{

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
      for (int j = i; j <= n; j++)
      setflag[i][j] = 0;


  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");



}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLIP::settings(int narg, char **arg)
{
  force->newton_pair = 0;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLIP::coeff(int narg, char **arg)
{
    if (!allocated) allocate();

    if (narg != 3 + atom->ntypes)
        error->all(FLERR,"Incorrect args for pair coefficients");

    // insure I,J args are * *
    if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
        error->all(FLERR,"Incorrect args for pair coefficients");

    read_pot(arg[2]);
    modelp = ModelParams(fp, true);

    // read args that map atom types to elements in potential file
    // map[i] = which element the Ith atom type is, -1 if NULL
    std::vector<int> map(atom->ntypes);
    for (int i = 3; i < narg; i++) {
        for (int j = 0; j < ele.size(); j++){
            if (strcmp(arg[i],ele[j].c_str()) == 0){
                map[i-3] = j;
                break;
            }
        }
    }

    for (int i = 1; i <= atom->ntypes; ++i){
        atom->set_mass(FLERR,i,mass[map[i-1]]);
        for (int j = 1; j <= atom->ntypes; ++j) setflag[i][j] = 1;
    }

    for (int i = 0; i < atom->natoms; ++i){
        types.emplace_back(map[(atom->type)[i]-1]);
    }
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMLIP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairMLIP::read_pot(char *file)
{
    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlip file: " << file << "\n";
        exit(8);
    }

    std::stringstream ss;
    std::string line, tmp;

    // line 1: elements
    ele.clear();
    std::getline( input, line );
    ss << line;
    while (!ss.eof()){
        ss >> tmp;
        ele.push_back(tmp);
    }
    ele.erase(ele.end()-1);
    ele.erase(ele.end()-1);
    ss.str("");
    ss.clear(std::stringstream::goodbit);

    // line 2-4: cutoff radius, pair type, descriptor type
    // line 5-7: model_type, max power, max l
    cutmax = cutforce = get_value<double>(input);
    std::string pair_type = get_value<std::string>(input);
    std::string des_type = get_value<std::string>(input);
    int model_type = get_value<int>(input);
    int maxp = get_value<int>(input);
    int maxl = get_value<int>(input);

    // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
    vector3i lm_array;
    vector2i type_comb;
    vector2d lm_coeffs_r;
    if (des_type == "gtinv"){
        int gtinv_order = get_value<int>(input);
        int size = gtinv_order - 1;
        vector1i gtinv_maxl = get_value_array<int>(input, size);
        std::vector<bool> gtinv_sym = get_value_array<bool>(input, size);

        Readgtinv rgt(gtinv_order, gtinv_maxl, gtinv_sym, ele.size());
        lm_array = rgt.get_lm_seq();
        type_comb = rgt.get_type_comb();
        lm_coeffs_r = rgt.get_lm_coeffs_r(); 
        lm_info = get_lm_info(maxl);
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    int n_reg_coeffs = get_value<int>(input);
    reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) reg_coeffs[i] /= scale[i];

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    int n_params = get_value<int>(input);
    vector2d des_params(n_params);
    for (int i = 0; i < n_params; ++i)
        des_params[i] = get_value_array<double>(input, 2);
    
    mass = get_value_array<double>(input, ele.size());

    bool force = true;
    fp = {int(ele.size()), force, des_params, cutmax, pair_type,
        des_type, model_type, maxp, maxl,
        lm_array, type_comb, lm_coeffs_r};
}

template<typename T>
T PairMLIP::get_value(std::ifstream& input)
{
    std::string line;
    std::stringstream ss;

    T val;
    std::getline( input, line );
    ss << line;
    ss >> val;

    return val;
}

template<typename T>
std::vector<T> PairMLIP::get_value_array
(std::ifstream& input, const int& size)
{
    std::string line;
    std::stringstream ss;

    std::vector<T> array(size);

    std::getline( input, line );
    ss << line;
    T val;
    for (int i = 0; i < array.size(); ++i){
        ss >> val;
        array[i] = val;
    }

    return array;
   
}

/*
void PairMLIP::compute_fn_ylm_products
(const double& dis, const vector1d& diff, 
 barray2dc& fn_ylm, barray2dc& fn_ylm_dx, 
 barray2dc& fn_ylm_dy, barray2dc& fn_ylm_dz){

    const int n_fn = modelp.get_n_fn(), n_lm = lm_info.size(); 
    const double delx(diff[0]),dely(diff[1]),delz(diff[2]);
    const vector1d &sph = cartesian_to_spherical(diff);

    int m,lm1,lm2;
    double costheta,sintheta,cosphi,sinphi,coeff,cc;
    dc f1,ylm_dphi,d0,d1,d2,term1,term2;

    vector1d fn, fn_d;
    vector1dc ylm, ylm_dtheta;

    get_fn(dis, fp, fn, fn_d);
    get_ylm(sph, lm_info, ylm, ylm_dtheta);

    costheta = cos(sph[0]), sintheta = sin(sph[0]);
    cosphi = cos(sph[1]), sinphi = sin(sph[1]);
    fabs(sintheta) > 1e-15 ? (coeff = 1.0 / sintheta) : (coeff = 0);
    for (int lm = 0; lm < n_lm; ++lm) {
        m = lm_info[lm][1], lm1 = lm_info[lm][2], lm2 = lm_info[lm][3];
        cc = pow(-1, m); 

        ylm_dphi = dc{0.0,1.0} * double(m) * ylm[lm];
        term1 = ylm_dtheta[lm] * costheta;
        term2 = coeff * ylm_dphi;
        d0 = term1 * cosphi - term2 * sinphi;
        d1 = term1 * sinphi + term2 * cosphi;
        d2 = - ylm_dtheta[lm] * sintheta;

        for (int n = 0; n < n_fn; ++n) {
            fn_ylm[n][lm1] = fn[n] * ylm[lm];
            fn_ylm[n][lm2] = cc * std::conj(fn_ylm[n][lm1]);
            f1 = fn_d[n] * ylm[lm];
            fn_ylm_dx[n][lm1] = - (f1 * delx + fn[n] * d0) / dis;
            fn_ylm_dy[n][lm1] = - (f1 * dely + fn[n] * d1) / dis;
            fn_ylm_dz[n][lm1] = - (f1 * delz + fn[n] * d2) / dis;
            fn_ylm_dx[n][lm2] = cc * std::conj(fn_ylm_dx[n][lm1]);
            fn_ylm_dy[n][lm2] = cc * std::conj(fn_ylm_dy[n][lm1]);
            fn_ylm_dz[n][lm2] = cc * std::conj(fn_ylm_dz[n][lm1]);
        }
    }
}
*/

