// // ======== test is_site ===============
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     std::cout << is_site(x, y);
//   }
//   std::cout << std::endl;
// }




// // ======== test is_link ===============
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     std::cout << '[';
//     for(int mu=0; mu<SIX; mu++){
//       if(mu!=0) std::cout << ' ';
//       std::cout << is_link(x, y, mu);
//     }
//     std::cout << ']';
//   }
//   std::cout << std::endl;
// }



// // ======== test get_Pauli ===============
// Pauli sigma = get_Pauli();

// sigma[0] << 1,0,0,1;
// sigma[1] << 0,-I,I,0;
// sigma[2] << 0,1,1,0;
// sigma[3] << 1,0,0,-1;

// std::cout << sigma[0] << std::endl;
// std::cout << sigma[1] << std::endl;
// std::cout << sigma[2] << std::endl;
// std::cout << sigma[3] << std::endl;



// // ======== test cshift ===============
// const int nu=5;
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     int xp=0, yp=0;
//     cshift(xp, yp, x, y, nu);
//     std::cout << "(" << xp << " " << yp << ")";
//   }
//   std::cout << std::endl;
// }


// // ======== test get_e ===============
// for(int mu=0; mu<6; mu++){
//   std::cout << "mu = " << mu << std::endl;
//   std::cout << get_e( mu ) << std::endl;
// }


// // ======== test Wilson_projector ===============
// for(int mu=0; mu<6; mu++){
//   std::cout << "mu = " << mu << std::endl;
//   std::cout << Wilson_projector( mu ) << std::endl;
// }




// // ======== test Dirac operators ===============
// Eigen::MatrixXcd dirac = get_Dirac_matrix();

// // Eigen::ComplexEigenSolver<Eigen::MatrixXcd> esolver( dirac, false );
// // Eigen::VectorXcd evec = esolver.eigenvalues();
// // for(auto elem : evec){
// //   std::cout << elem.real() << " " << elem.imag() << std::endl;
// // }

// Eigen::VectorXcd vector1 = Eigen::VectorXcd::Random(2*Lx*Ly);
// Eigen::VectorXcd vector2 = vector1;

// std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

// vector1 = dirac*vector1;
// vector2 = multD_eigen(vector2);
// std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

// vector1 = dirac.adjoint() * vector1;
// vector2 = multDdagger_eigen(vector2);

// std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

// Eigen::VectorXcd init = Eigen::VectorXcd::Zero(2*Lx*Ly);
// Eigen::VectorXcd b0 = Eigen::VectorXcd::Random(2*Lx*Ly);
// Vect b = multDdagger_eigen(b0);

// Vect sol = CG(init, b);

// Vect test = A(sol) - b;
// std::cout << "test = " << test.norm() << std::endl;

// Eigen::FullPivLU<Eigen::MatrixXcd> lu(dirac);
// Vect sol_direct = lu.solve(b0);
// std::cout << "solve_diff" << (sol-sol_direct).norm() << std::endl;



