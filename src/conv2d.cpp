#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <armadillo>

#include "mesh2d.hpp"
#include "euler2d.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        throw std::runtime_error("Need the name of the mesh");
    }

    Mesh2d mesh(argv[1]);
    mesh.setupMatrices();

    Euler2d problem(mesh, 1.4, 1.0, 0.5, 1.0);
    problem.initialize();
    problem.setBoundaryStates();

    std::cout << "Residuals:" << std::endl;

    double residual = 0.0;
    for (arma::uword i = 0; i < 1000; ++i)
    {
        residual = problem.timestep();
        std::cout << std::scientific << std::setprecision(15) << residual
                  << std::endl;
    }
    std::cout << std::scientific << std::setprecision(15) << residual
              << std::endl;

    return 0;
}
