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

const long Lx = 6 * 20;
const long Ly = Lx;

// -------------------

// const unsigned long Lx = 9;
// const unsigned long Ly = 12;

// const unsigned long Lx = 6 * 2;
// const unsigned long Ly = 6 * 4;

// const unsigned long Lx = 6 * 4;
// const unsigned long Ly = 6 * 8;

// const unsigned long Lx = 6 * 6;
// const unsigned long Ly = 6 * 12;

// const unsigned long Lx = 6 * 8;
// const unsigned long Ly = 6 * 16;

// const unsigned long Lx = 6 * 12;
// const unsigned long Ly = 2*Lx;

// const unsigned long Lx = 6 * 16;
// const unsigned long Ly = 6 * 32;

// const unsigned long Lx = 6 * 32;
// const unsigned long Ly = Lx * 3;


// const unsigned long Lx = 6 * 16;
// const unsigned long Ly = 6 * 16 * 3;


// -------------------


// const unsigned long Lx = 6 * 2;
// const unsigned long Ly = 6 * 2;

// const unsigned long Lx = 6 * 6;
// const unsigned long Ly = 6 * 6;




// const unsigned long Lx = 6 * 20; // 20
// const unsigned long Ly = Lx;


// const unsigned long Lx = 6 * 32;
// const unsigned long Ly = 6 * 32;

// const unsigned long Lx = 6 * 64;
// const unsigned long Ly = 6 * 64;

// const unsigned long Lx = 6 * 128;
// const unsigned long Ly = 6 * 128;

// const unsigned long Lx = 6 * 256;
// const unsigned long Ly = 6 * 256;

// const unsigned long Lx = 6 * 512;
// const unsigned long Ly = 6 * 512;
// const unsigned long Lx = 6 * 1024;
// const unsigned long Ly = 6 * 1024;

// const double m = 2.0;
// const unsigned long Lx = 6 * 512;
// const unsigned long Ly = 6 * 1024;

// const double alat = 0.01; // ell
const int nparallel = 12;

// const double Vy = 3.0*sqrt(3.0)/4.0 * alat*alat;
// const double my = m + 2.0/3.0 * 3.0/alat;

// const double Mu = 1.0;
// double Mu = 1.0;
// constexpr long double tt = -0.3; // 48.0l/(Lx*Ly*2.0l/3.0l);
// constexpr double ell1 = 1.0;
// constexpr double ell2 = 0.5;


// const double length = 1.0;
// const double theta = M_PI/3.0;
const double length = 1.0;
const double theta = 4.0*M_PI/9.0;
const double omega[2] = { length * std::cos(theta), length * std::sin(theta) };

double ell0[2]; // = {1.0, 0.0};
double ell2[2]; // = -omega;
double ell1[2]; // -ell2 - ell0

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

