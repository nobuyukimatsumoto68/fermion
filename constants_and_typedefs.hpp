#pragma once
#include <Eigen/Dense>

using Idx = long int;
using Complex = std::complex<double>;
using M2 = Eigen::Matrix2cd;
using V2 = Eigen::Vector2cd;

using Vect = Eigen::VectorXcd;
using Mat = Eigen::MatrixXcd;

using Pauli = std::array<M2, 4>;

// ======================================


const int TWO = 2;
const int SIX = 6;
const Complex I = Complex(0, 1);


const int Lx = 6 * 64;
const int Ly = 6 * 64;
const double alat = 0.01; // ell

const double m = 0.0;

// const double Vy = 3.0*sqrt(3.0)/4.0 * alat*alat;
const double my = m + 2.0/3.0 * 3.0/alat;
const double kappa = (2.0/3.0) * 2.0 / alat / my;


Pauli get_Pauli();
const Pauli sigma = get_Pauli();
