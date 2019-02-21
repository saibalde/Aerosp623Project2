#include <string>
#include <stdexcept>

#include <iostream>

#include <armadillo>

#include "Conv2D/Mesh.hpp"
#include "Conv2D/EulerDefaultBase.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        throw std::runtime_error("No choice of input mesh given");
    }

    int choice = std::atoi(argv[1]);

    if (choice < 0 || choice > 4)
    {
        throw std::runtime_error("Invalid choice of input mesh");
    }

    const std::string choiceString = std::to_string(choice);

    const std::string meshFile     = "bump" + choiceString + ".gri";
    const std::string residualFile = "SecondOrderPreserveResidual" + choiceString + ".dat";

    Mesh mesh;
    mesh.readFromFile(meshFile);
    mesh.computeMatrices();

    const double gamma = 1.4;
    const double R = 1.0;
    const double MInf = 0.5;
    const double pInf = 1.0;
    const double CFL = 0.5;

    EulerDefaultBase problem;
    problem.setMesh(mesh);
    problem.setParams(gamma, R, MInf, pInf, CFL);
    problem.initialize();

    const arma::uword numIter = 1000;
    problem.runSecondOrderSolver(numIter, residualFile);

    return 0;
}
