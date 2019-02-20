#include <fstream>
#include <iomanip>
#include <stdexcept>

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
    const std::string machFile       = "SecondOrderSolverMachNumbers" + choiceString + ".dat";
    const std::string solutionFile   = "SecondOrderSolverSolution" + choiceString + ".dat";

    const std::string initialFile    = "FirstOrderSolverSolution" + choiceString + ".dat";

    Mesh mesh;
    mesh.readFromFile(meshFile);
    mesh.computeMatrices();

    const double gamma = 1.4;
    const double R = 1.0;
    const double MInf = 0.5;
    const double pInf = 1.0;
    const double CFL = 0.5;

    Euler problem;
    problem.setMesh(mesh);
    problem.setParams(gamma, R, MInf, pInf, CFL);
    problem.initialize(initialFile);

    const double tolerance = 1.0e-07;
    problem.secondOrderSolver(tolerance, residualFile);

    std::ofstream file(validationFile);
    file << "Lift coefficient = "
         << std::scientific << std::setprecision(15)
         << problem.liftCoefficient() << std::endl;
    file << "Drag coefficient = "
         << std::scientific << std::setprecision(15)
         << problem.dragCoefficient() << std::endl;
    file << "Entropy error = "
         << std::scientific << std::setprecision(15)
         << problem.entropyError() << std::endl;

    arma::vec cp;
    problem.pressureCoefficients(cp);
    cp.save(pressureFile, arma::raw_ascii);

    arma::vec M;
    problem.machNumbers(M);
    M.save(machFile, arma::raw_ascii);

    problem.writeStateToFile(solutionFile);

    return 0;
}
