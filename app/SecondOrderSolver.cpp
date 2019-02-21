#include <string>
#include <stdexcept>

#include <cstdlib>

#include <armadillo>

#include "Conv2D/Mesh.hpp"
#include "Conv2D/EulerDefaultBase.hpp"

class Euler : public EulerDefaultBase
{
public:
    Euler() : EulerDefaultBase()
    {
    }

    ~Euler() = default;

    void computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &U, arma::vec &F, double &s) const;

    void computeRightFlux(const arma::vec &UInt, const arma::rowvec &n,
                          arma::vec &U, arma::vec &F, double &s) const;

    void computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                        arma::vec &U, arma::vec &F, double &s) const;

    void computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                         arma::vec &U, arma::vec &F, double &s) const;
};

void Euler::computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &U, arma::vec &F, double &s) const
{
    applyInvisidWallBC(UInt, n, U, F, s);
}

void Euler::computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                              arma::vec &U, arma::vec &F, double &s) const
{
    applyInvisidWallBC(UInt, n, U, F, s);
}

void Euler::computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &U, arma::vec &F, double &s) const
{
    applyInflowBC(UInt, n, U, F, s);
}

void Euler::computeRightFlux(const arma::vec &UInt, const arma::rowvec &n,
                             arma::vec &U, arma::vec &F, double &s) const
{
    applyOutflowBC(UInt, n, U, F, s);
}

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

    const std::string meshFile       = "bump" + choiceString + ".gri";
    const std::string residualFile   = "SecondOrderSolverResidual" + choiceString + ".dat";
    const std::string validationFile = "SecondOrderSolverValidation" + choiceString + ".dat";
    const std::string pressureFile   = "SecondOrderSolverPressureCoefficients" + choiceString + ".dat";
    const std::string machFile       = "SecondOrderSolverMachNumbers" + choiceString + ".vtk";
    const std::string solutionFile   = "SecondOrderSolverSolution" + choiceString + ".dat";

    const std::string initialFile    = "FirstOrderSolverSolution" + choiceString + ".dat";

    Mesh mesh;
    mesh.readFromFile(meshFile);
    mesh.computeMatrices();

    Euler problem;
    problem.setMesh(mesh);
    problem.setGasConstant(1.0);
    problem.setSpecificHeatRatio(1.4);
    problem.setFreeFlowMachNumber(0.5);
    problem.setFreeFlowStaticPressure(1.0);
    problem.setCFLNumber(0.5);

    problem.setInitialState(initialFile);

    const double tolerance = 1.0e-07;
    problem.runSecondOrderSolver(tolerance, residualFile);

    problem.writeValidationValuesTofile(validationFile);
    problem.writePressureCoefficientsToFile(pressureFile);
    problem.writeMachNumbersToFile(machFile);

    problem.writeStateToFile(solutionFile);

    return 0;
}
