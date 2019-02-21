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

    EulerDefaultBase problem;
    problem.setMesh(mesh);
    problem.setGasConstant(1.0);
    problem.setSpecificHeatRatio(1.4);
    problem.setFreeFlowMachNumber(0.5);
    problem.setFreeFlowStaticPressure(1.0);
    problem.setCFLNumber(0.5);

    problem.setInitialState();

    const arma::uword numIter = 1000;
    problem.runSecondOrderSolver(numIter, residualFile);

    return 0;
}
