#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <armadillo>

#include "mesh2d.hpp"
#include "euler2d.hpp"

int main()
{
    Mesh2d mesh("bump0.gri");
    mesh.setupMatrices();

    Euler2d problem(mesh, 1.4, 1.0, 0.5, 1.0);
    problem.initialize();
    problem.setFreeStreamBC();

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
