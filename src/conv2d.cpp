#include <iostream>

#include <armadillo>

#include "mesh2d.hpp"
#include "euler2d.hpp"

int main()
{
    Mesh2d mesh("bump0.gri");
    mesh.setupMatrices();

    Euler2d problem(mesh, 1.4, 1.0, 0.5, 1.0);
    problem.initialize();
    problem.setBoundaryStates();

    arma::mat R(4, mesh.nElemTot, arma::fill::none);
    arma::vec S(mesh.nElemTot, arma::fill::none);

    problem.computeResidual(problem.U_, R, S);

    R.save("residual.dat", arma::raw_ascii);
}
