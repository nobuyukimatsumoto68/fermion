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

// const unsigned long Lx = 3 * 1;
// const unsigned long Ly = 3 * 1;

// -------------------

// const unsigned long Lx = 6 * 1;
// const unsigned long Ly = 6 * 2;

// const unsigned long Lx = 6 * 2;
// const unsigned long Ly = 6 * 4;

// const unsigned long Lx = 6 * 4;
// const unsigned long Ly = 6 * 8;

// const unsigned long Lx = 6 * 6;
// const unsigned long Ly = 6 * 12;

// const unsigned long Lx = 6 * 8;
// const unsigned long Ly = 6 * 16;

const unsigned long Lx = 6 * 16;
const unsigned long Ly = 6 * 32;

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

const double alat = 0.01; // ell
const int nparallel = 1;


// const double Vy = 3.0*sqrt(3.0)/4.0 * alat*alat;
const double my = m + 2.0/3.0 * 3.0/alat;

// const double Mu = 1.0;
// double Mu = 1.0;

const double kappa = (2.0/3.0) * 2.0 / alat / my;
// const double kappa = 0.8;

const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
// const std::string description = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat)+"nu"+std::to_string(nu);
// const std::string description_old = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat);
const std::string dir_data = "./data/";

