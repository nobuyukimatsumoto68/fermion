#pragma once

#include <Eigen/Dense>
#include <complex>

using Idx = long int;
using Complex = std::complex<double>;
// const Complex I = Complex(0, 1);

using M2 = Eigen::Matrix2cd;
using V2 = Eigen::Vector2cd;

using Vect = Eigen::VectorXcd;
using Mat = Eigen::MatrixXcd;

using Pauli = std::array<M2, 4>;
