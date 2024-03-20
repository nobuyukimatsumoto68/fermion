#pragma once
#include <array>
#include <string>


// ======================================


const bool is_periodic_orthogonal = false;

const int TWO = 2;
const int THREE = 3;
const int SIX = 6;

const int nu = 3;

const double m = 0.0;
// const int Lx = 3 * 1;
// const int Ly = 3 * 1;

const int Lx = 6 * 2;
const int Ly = 6 * 2;

// const int Lx = 6 * 6;
// const int Ly = 6 * 6;

// const int Lx = 6 * 32;
// const int Ly = 6 * 32;

// const int Lx = 6 * 64;
// const int Ly = 6 * 64;

// const int Lx = 6 * 128;
// const int Ly = 6 * 128;

// const int Lx = 6 * 256;
// const int Ly = 6 * 256;

// const int Lx = 6 * 512;
// const int Ly = 6 * 512;
// const int Lx = 6 * 1024;
// const int Ly = 6 * 1024;

// const double m = 2.0;
// const int Lx = 6 * 512;
// const int Ly = 6 * 1024;

const double alat = 0.01; // ell
const int nparallel = 1;


// const double Vy = 3.0*sqrt(3.0)/4.0 * alat*alat;
const double my = m + 2.0/3.0 * 3.0/alat;

// const double Mu = 1.0;
// double Mu = 1.0;

const double kappa = (2.0/3.0) * 2.0 / alat / my;
// const double kappa = 0.8;

const std::string description = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat)+"ortho"+std::to_string(is_periodic_orthogonal);
const std::string description_old = "m"+std::to_string(m)+"Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"alat"+std::to_string(alat);
const std::string dir_data = "./data/";

