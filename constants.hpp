#pragma once
#include <array>
#include <string>


// ======================================


const bool is_periodic_orthogonal = false;

const int TWO = 2;
const int THREE = 3;
const int SIX = 6;

int nu = 3; // 1,2,3,4 // PP, PA, AA, AP // xy
// PP, PA, AA, AP

const double m = 0.0;

const long Lx = 6*32; // 6; // ; * 20;
const long Ly = Lx;

// -------------------

const int nparallel = 12;

// const double Mu = 1.0;
// double Mu = 1.0;
// constexpr long double tt = -0.3; // 48.0l/(Lx*Ly*2.0l/3.0l);
// constexpr double ell1 = 1.0;
// constexpr double ell2 = 0.5;

// const double abs_tautil = 1.0; // 1.0 <= abs
// const double arg_tautil = 3.5/9.0 * M_PI; // pi/3.0 <= arg <= pi/2.0 // 3.35
const double abs_tautil = 1.2; // 1.0 <= abs
const double arg_tautil = 4.0/9.0 * M_PI; // pi/3.0 <= arg <= pi/2.0 // 3.35
// const double abs_tautil = 1.0; // 1.0 <= abs
// const double arg_tautil = 3.5/9.0 * M_PI; // pi/3.0 <= arg <= pi/2.0 // 3.35
const bool tautil_default = false; // true->below
double tautil1 = 0.3420201433256688;
double tautil2 = 0.9396926207859083 + 0.00001;

const int xs = 0;
const int ys = 0;

const int xP = Lx/2, yP = 0;


double ell0[2]; // = {1.0, 0.0};
double ell1[2]; // = -omega;
double ell2[2]; // -ell2 - ell0

double ell[3]; // {|ell0|, |ell1|, |ell2|}

double ell_star0[2];
double ell_star1[2];
double ell_star2[2];

double e0[2];
double e1[2];
double e2[2];


double kappa[3];


// constexpr double kappa[3] = {
//   2.0/3.0 + 2.0/3.0*tt,
//   2.0/3.0 - 1.0/3.0*tt,
//   2.0/3.0 - 1.0/3.0*tt
//   // 2.0 / (1.0 + 2.0*ell2/ell1),
//   // 2.0 / (1.0 + 2.0*ell1/ell2),
//   // 2.0 / (1.0 + 2.0*ell1/ell2)

// };
// constexpr double kappa[3] = {
//   2.0/3.0 + 2.0/3.0*tt,
//   2.0/3.0 - 1.0/3.0*tt,
//   2.0/3.0 - 1.0/3.0*tt
// };
// const double kappa = (2.0/3.0) * 2.0 / alat / my;
// const double kappa = 0.0;

// std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
// const std::string description = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat)+"nu"+std::to_string(nu);
// const std::string description_old = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat);
const std::string dir_data = "./data/";

std::string str( const bool x){
  if (x) return "True";
  return "False";
}
